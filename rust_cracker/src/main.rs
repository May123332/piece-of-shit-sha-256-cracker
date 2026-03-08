use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use sha2::{Sha256, Digest};
use sha1::Sha1;
use std::fs::File;
use std::io::{self, BufRead};
use std::process;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use ocl::{Device, Platform, Context, Queue, Program, Kernel, Buffer, MemFlags};

mod opencl_kernels;

// HIP FFI (enabled only when build.rs successfully compiles hip/sha256.hip)
#[cfg(has_hip)]
#[link(name = "hip_cracker", kind = "static")]
extern "C" {
    fn get_hip_device_info(info: *mut HipDeviceInfo) -> i32;
    fn launch_brute_force_hip(
        charset: *const u8, 
        charset_len: u64, 
        start_offset: u64, 
        len: u32,
        target_hash: *const u32, 
        result_found: *mut u32, 
        result_word: *mut u8,
        batch_size: u64,
        blocks: i32,
        threads: i32
    ) -> i32;
    fn cleanup_brute_force_hip();
}

#[cfg(has_hip)]
#[repr(C)]
struct HipDeviceInfo {
    available: i32,
    device_count: i32,
    device_id: i32,
    multiprocessor_count: i32,
    max_threads_per_block: i32,
    warp_size: i32,
    total_global_mem: u64,
    clock_rate_khz: i32,
    name: [u8; 256],
}

#[cfg(has_hip)]
impl Default for HipDeviceInfo {
    fn default() -> Self {
        Self {
            available: 0,
            device_count: 0,
            device_id: 0,
            multiprocessor_count: 0,
            max_threads_per_block: 0,
            warp_size: 0,
            total_global_mem: 0,
            clock_rate_khz: 0,
            name: [0u8; 256],
        }
    }
}

#[cfg(has_hip)]
impl HipDeviceInfo {
    fn name(&self) -> String {
        let end = self.name.iter().position(|&byte| byte == 0).unwrap_or(self.name.len());
        String::from_utf8_lossy(&self.name[..end]).trim().to_string()
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    target: String, 
    #[arg(short, long)]
    salt: Option<String>,
    #[arg(short, long, value_enum, default_value_t = Mode::Dictionary)]
    mode: Mode,
    #[arg(short, long)]
    wordlist: Option<String>,
    #[arg(short, long)]
    length: Option<usize>,
    #[arg(short, long, default_value = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*?")]
    charset: String,
    #[arg(long, default_value_t = false)]
    no_gpu: bool,
    #[arg(long)]
    total: Option<u64>,
    #[arg(long, default_value = "")]
    prefix: String,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum Mode {
    Dictionary,
    Brute,
}

#[derive(Clone, Copy, PartialEq)]
enum Algorithm {
    Md5,
    Sha1,
    Sha256,
}

static GLOBAL_COUNTER: AtomicU64 = AtomicU64::new(0);
const COUNTER_FLUSH_INTERVAL: u64 = 8192;
const HIP_ITEMS_PER_THREAD: u64 = 1;

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    init_rayon_pool();
    
    // Trim input
    let target_trimmed = args.target.trim();
    
    println!("DEBUG: Mode={:?}, Length={:?}, Prefix='{}'", args.mode, args.length, args.prefix);
    
    // 1. Detect Algorithm and Parse Hash
    let (algo, clean_hash_str) = if target_trimmed.starts_with("$SHA$") {
        let parts: Vec<&str> = target_trimmed.split('$').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 3 && parts[0] == "SHA" {
            (Algorithm::Sha256, parts[2].to_string())
        } else {
            eprintln!("Invalid $SHA$ format.");
            process::exit(1);
        }
    } else {
        let h = args.target.clone();
        if h.len() == 32 { (Algorithm::Md5, h) }
        else if h.len() == 40 { (Algorithm::Sha1, h) }
        else if h.len() == 64 { (Algorithm::Sha256, h) }
        else {
            eprintln!("Unknown hash format (len={})", h.len());
            process::exit(1);
        }
    };

    let target_bytes = hex::decode(&clean_hash_str).expect("Failed to decode target hash hex");

    let salt_str = if args.target.starts_with("$SHA$") {
        let parts: Vec<&str> = args.target.split('$').filter(|s| !s.is_empty()).collect();
        if parts.len() >= 2 {
            Some(parts[1].to_string())
        } else {
            None
        }
    } else {
        args.salt.clone()
    };
    
    let salt_bytes = salt_str.as_ref().map(|s| s.as_bytes().to_vec());

    println!("Target (Hex): {}", clean_hash_str);
    if let Some(s) = &salt_str {
        println!("Salt: {}", s);
    }

    // 2. GPU Setup
    let mut gpu_ready = false;
    let mut platform_id = None;
    let mut device_id = None;
    #[cfg(has_hip)]
    let mut hip_ready = false;
    #[cfg(has_hip)]
    let mut hip_device_info = HipDeviceInfo::default();

    if !args.no_gpu && algo == Algorithm::Sha256 {
        #[cfg(has_hip)]
        {
            let rc = unsafe { get_hip_device_info(&mut hip_device_info as *mut HipDeviceInfo) };
            if rc == 0 && hip_device_info.available == 1 {
                println!(
                    "GPU_DETECTED: {} ({} processor groups, wave {}, {} MiB VRAM)",
                    hip_device_info.name(),
                    hip_device_info.multiprocessor_count,
                    hip_device_info.warp_size,
                    hip_device_info.total_global_mem / (1024 * 1024)
                );
                hip_ready = true;
                gpu_ready = true;
            } else {
                eprintln!(
                    "ROCm HIP unavailable (error code {}). If this is a permissions issue, add your user to the render group.",
                    rc
                );
            }
        }
        
        if args.mode == Mode::Dictionary || !hip_ready {
            match std::panic::catch_unwind(Platform::list) {
                Ok(platforms) => {
                    for p in platforms {
                        if let Ok(devices) = Device::list_all(&p) {
                            if !devices.is_empty() {
                                if !gpu_ready {
                                     println!("GPU_DETECTED: {}", devices[0].name().unwrap_or_else(|_| "Unknown".to_string()));
                                     gpu_ready = true;
                                }
                                platform_id = Some(p);
                                device_id = Some(devices[0]);
                                break;
                            }
                        }
                    }
                }
                Err(_) => eprintln!("OpenCL platform probe unavailable on this session."),
            }
        }
    }
    
    // Stats Thread
    let found_flag = Arc::new(AtomicBool::new(false));
    let found_flag_clone = found_flag.clone();
    let total_ops = args.total;

    thread::spawn(move || {
        let mut last_count = 0;
        let mut last_time = Instant::now();
        loop {
            thread::sleep(Duration::from_secs(1));
            if found_flag_clone.load(Ordering::Relaxed) { break; }
            
            let current_count = GLOBAL_COUNTER.load(Ordering::Relaxed);
            let elapsed = last_time.elapsed().as_secs_f64();
            let delta = current_count - last_count;
            let speed = delta as f64 / elapsed;
            
            // Format Speed
            let speed_str = if speed > 1_000_000_000.0 {
                format!("{:.2} GH/s", speed / 1_000_000_000.0)
            } else if speed > 1_000_000.0 {
                format!("{:.2} MH/s", speed / 1_000_000.0)
            } else {
                format!("{:.2} KH/s", speed / 1_000.0)
            };

            // Format Total & Progress
            let total_str = if let Some(tot) = total_ops {
                 let pct = if tot > 0 { (current_count as f64 / tot as f64) * 100.0 } else { 0.0 };
                 fn fmt(n: u64) -> String {
                     let s = n.to_string();
                     let mut result = String::new();
                     let mut count = 0;
                     for c in s.chars().rev() {
                         if count > 0 && count % 3 == 0 {
                             result.insert(0, ',');
                         }
                         result.insert(0, c);
                         count += 1;
                     }
                     result
                 }
                 format!("{}|{}|{:.2}", fmt(current_count), fmt(tot), pct)
            } else {
                 format!("{}|0|0.00", current_count)
            };

            println!("STATS:{}|{}", speed_str, total_str);
            
            last_count = current_count;
            last_time = Instant::now();
        }
    });

    if gpu_ready {
        #[cfg(has_hip)]
        if args.mode == Mode::Brute && hip_ready {
            println!("ENGINE: ROCm HIP (Native Kernel Launch)");
            let res = run_hip_brute(
                &hip_device_info,
                &args,
                &target_bytes,
                salt_bytes.as_deref(),
                &found_flag,
            );
            
            if let Ok(Some(pwd)) = res {
                found_flag.store(true, Ordering::Relaxed);
                println!("MATCH_FOUND:{}", pwd);
                return Ok(());
            } else if res.is_ok() {
                // Completed no match
                return Ok(());
            } else {
                eprintln!("HIP Error: {:?}. Falling back to OpenCL...", res.err());
            }
        }

        // OpenCL Path (Dictionary or Fallback)
        if platform_id.is_some() {
            if args.mode == Mode::Dictionary {
                 println!("ENGINE: OpenCL (GPU Dictionary Mode)");
            } else {
                 println!("ENGINE: OpenCL (GPU Brute Fallback)");
            }
            
            let res = run_gpu_mode(
                platform_id.unwrap(), 
                device_id.unwrap(), 
                &args,
                &target_bytes,
                salt_bytes.as_deref()
            );
            match res {
                Ok(Some(pwd)) => {
                    found_flag.store(true, Ordering::Relaxed);
                    println!("MATCH_FOUND:{}", pwd);
                    return Ok(());
                }
                Ok(None) => return Ok(()),
                Err(e) => {
                    eprintln!("OpenCL Error: {}. Falling back to CPU.", e);
                }
            }
        }
    } 
    
    // CPU FALLBACK
    println!("ENGINE: Native AVX2 (Zero-Allocation Optimized)");

    match args.mode {
        Mode::Dictionary => {
            let path = args.wordlist.expect("Wordlist required for dictionary mode");
            let file = File::open(path)?;
            let reader = io::BufReader::new(file);
            
            reader.lines()
                .par_bridge()
                .try_for_each(|line_res| {
                    if found_flag.load(Ordering::Relaxed) { return None; }
                    
                    if let Ok(word) = line_res {
                         if check_candidate_stack(word.as_bytes(), &target_bytes, salt_bytes.as_deref(), algo) {
                             found_flag.store(true, Ordering::Relaxed);
                             println!("MATCH_FOUND:{}", word);
                             return None; 
                         }
                         GLOBAL_COUNTER.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(())
                });
        }
        Mode::Brute => {
            let max_len = args.length.unwrap_or(4);
            let charset: Vec<u8> = args.charset.bytes().collect();
            
            for len in 1..=max_len {
                if found_flag.load(Ordering::Relaxed) { break; }
                println!("Checking length {}...", len);
                
                charset.par_iter().for_each(|&first_char| {
                    if found_flag.load(Ordering::Relaxed) { return; }
                    let mut local_counter = 0u64;
                    
                    // Status Update
                    if len >= 4 {
                         println!("STATUS:Testing variations starting with '{}'...", first_char as char);
                    }

                    let mut buffer = [0u8; 128]; 
                    buffer[0] = first_char;
                    recursive_brute_stack(
                        &mut buffer,
                        1,
                        len,
                        &charset,
                        &target_bytes,
                        salt_bytes.as_deref(),
                        algo,
                        &found_flag,
                        &mut local_counter,
                    );
                    flush_global_counter(&mut local_counter);
                });
            }
        }
    }

    Ok(())
}

#[cfg(has_hip)]
fn run_hip_brute(
    device_info: &HipDeviceInfo,
    args: &Args,
    target_bytes: &[u8],
    salt: Option<&[u8]>, 
    found_flag: &Arc<AtomicBool>
) -> anyhow::Result<Option<String>> {
    struct HipCleanupGuard;
    impl Drop for HipCleanupGuard {
        fn drop(&mut self) {
            unsafe { cleanup_brute_force_hip() };
        }
    }

    let _cleanup_guard = HipCleanupGuard;
    let charset_bytes = args.charset.as_bytes();
    let max_len = args.length.unwrap_or(4);
    let prefix_len = args.prefix.len() as u32;

    if !args.prefix.is_empty() {
        anyhow::bail!("HIP brute-force mode does not yet support --prefix");
    }
    if salt.is_some() {
        anyhow::bail!("HIP brute-force mode does not yet support salted hashes");
    }
    if max_len > 16 {
        anyhow::bail!("HIP brute-force mode currently supports --length up to 16");
    }

    let mut target_u32 = [0u32; 8];
    for i in 0..8 {
        let j = i * 4;
        target_u32[i] = u32::from_be_bytes([target_bytes[j], target_bytes[j+1], target_bytes[j+2], target_bytes[j+3]]);
    }

    let mut found_host = [0u32; 1];
    let mut result_host = [0u8; 64];

    let suffix_start = if prefix_len == 0 { 1 } else { 
        if max_len as u32 > prefix_len { (max_len as u32 - prefix_len) as usize } else { 0 }
    };
    let suffix_end = if max_len as u32 >= prefix_len { (max_len as u32 - prefix_len) as usize } else { 0 };

    for len in suffix_start..=suffix_end {
        let total_combos = (charset_bytes.len() as u64).pow(len as u32);
        println!("STATUS:Checking suffix length {} ({} combinations)...", len, total_combos);
        let launch_config = hip_launch_config(device_info);
        println!(
            "HIP_TUNING: blocks={} threads={} batch={}",
            launch_config.blocks,
            launch_config.threads,
            launch_config.batch_size
        );
        let mut offset: u64 = 0;
        
        while offset < total_combos {
            if found_flag.load(Ordering::Relaxed) { return Ok(None); }
            
            if len >= 1 {
                let current_idx = offset;
                let divisor = (charset_bytes.len() as u64).pow(len as u32 - 1);
                let char_idx = (current_idx / divisor) % (charset_bytes.len() as u64);
                let current_char = charset_bytes[char_idx as usize] as char;
                println!("STATUS:Scanning range starting with '{}{}'...", args.prefix, current_char);
            }

            let current_batch = std::cmp::min(launch_config.batch_size, total_combos - offset);
            
            found_host[0] = 0;
            result_host.fill(0);

            let rc = unsafe {
                launch_brute_force_hip(
                    charset_bytes.as_ptr(),
                    charset_bytes.len() as u64,
                    offset,
                    len as u32,
                    target_u32.as_ptr(),
                    found_host.as_mut_ptr(),
                    result_host.as_mut_ptr(),
                    current_batch,
                    launch_config.blocks,
                    launch_config.threads
                )
            };
            if rc != 0 {
                anyhow::bail!("HIP kernel launch failed with error code {}", rc);
            }
            
            GLOBAL_COUNTER.fetch_add(current_batch, Ordering::Relaxed);
            
            if found_host[0] == 1 {
                let s = String::from_utf8_lossy(&result_host);
                let suffix = s.trim_matches(char::from(0));
                let full_pass = format!("{}{}", args.prefix, suffix);
                
                // CPU VERIFICATION (Safety Check)
                let mut hasher = Sha256::new();
                hasher.update(&full_pass);
                if let Some(_s) = salt {
                    // Explicitly unsupported in HIP mode for now.
                }
                let cpu_hash = hasher.finalize();
                
                // Compare with target_bytes
                if cpu_hash.as_slice() == target_bytes {
                    return Ok(Some(full_pass));
                } else {
                    eprintln!("WARN: GPU reported false positive: '{}'. Hardware error? Continuing...", full_pass);
                    found_host[0] = 0;
                }
            }
            
            offset += current_batch;
        }
    }

    Ok(None)
}

fn run_gpu_mode(
    platform: Platform, 
    device: Device, 
    args: &Args,
    target: &[u8],
    salt: Option<&[u8]>
) -> ocl::Result<Option<String>> {
    
    // 1. Setup OpenCL Context & Buffers (Common)
    let context = Context::builder().platform(platform).devices(device).build()?;
    let queue = Queue::new(&context, device, None)?;
    let program = Program::builder()
        .src(opencl_kernels::SHA256_KERNEL)
        .devices(device)
        .build(&context)?;

    // Common Buffers
    let target_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only().host_write_only())
        .len(32)
        .copy_host_slice(target)
        .build()?;
    
    let salt_len = salt.map(|s| s.len()).unwrap_or(0);
    let salt_buf = Buffer::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only().host_write_only())
        .len(if salt_len > 0 { salt_len } else { 1 })
        .copy_host_slice(if salt_len > 0 { salt.unwrap() } else { &[0] })
        .build()?;

    let result_found_buf = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_write())
        .len(1)
        .fill_val(0u32)
        .build()?;

    if args.mode == Mode::Dictionary {
        let path = args.wordlist.as_ref().unwrap();
        println!("STATUS:Loading wordlist to RAM...");
        let file = File::open(path).expect("File not found");
        let reader = io::BufReader::new(file);
        
        let mut flat_data = Vec::new();
        let mut offsets = Vec::new();
        offsets.push(0u32);
        
        let mut count = 0;
        for line in reader.lines() {
            if let Ok(word) = line {
                flat_data.extend_from_slice(word.as_bytes());
                offsets.push(flat_data.len() as u32);
                count += 1;
            }
        }
        println!("STATUS:Uploading {} words to VRAM...", count);
        
        let local_wg = 256;
        let aligned_word_count = ((count + local_wg - 1) / local_wg) * local_wg;
        let mut offsets_padded = offsets.clone();
        let last_offset = *offsets.last().unwrap();
        while offsets_padded.len() <= aligned_word_count + 1 { offsets_padded.push(last_offset); }

        let wordlist_buf = Buffer::builder().queue(queue.clone()).flags(MemFlags::new().read_only()).len(flat_data.len()).copy_host_slice(&flat_data).build()?;
        let offsets_buf = Buffer::builder().queue(queue.clone()).flags(MemFlags::new().read_only()).len(offsets_padded.len()).copy_host_slice(&offsets_padded).build()?;
        let result_index_buf = Buffer::<u32>::builder().queue(queue.clone()).flags(MemFlags::new().read_write()).len(1).fill_val(0u32).build()?;

        let max_wg = device.max_wg_size()?;
        let local_wg = if max_wg >= 256 { 256 } else { max_wg };

        let kernel = Kernel::builder()
            .program(&program)
            .name("dictionary_attack")
            .queue(queue.clone())
            .global_work_size(aligned_word_count)
            .local_work_size(local_wg)
            .arg(&wordlist_buf).arg(&offsets_buf).arg(count as u32)
            .arg(&target_buf).arg(&salt_buf).arg(salt_len as u32)
            .arg(&result_found_buf).arg(&result_index_buf)
            .build()?;

        unsafe { kernel.enq()?; }
        
        let mut found = vec![0u32; 1];
        result_found_buf.read(&mut found).enq()?;
        GLOBAL_COUNTER.store(count as u64, Ordering::Relaxed);

        if found[0] == 1 {
            let mut idx = vec![0u32; 1];
            result_index_buf.read(&mut idx).enq()?;
            let i = idx[0] as usize;
            let s = offsets[i] as usize;
            let e = offsets[i+1] as usize;
            return Ok(Some(String::from_utf8_lossy(&flat_data[s..e]).to_string()));
        }

    } else {
        // Brute-force OpenCL kernel path is not implemented in this build.
        return Err(ocl::Error::from("OpenCL brute-force fallback is not implemented"));
    }

    Ok(None)
}

fn recursive_brute_stack(
    buffer: &mut [u8],
    current_len: usize,
    target_len: usize,
    charset: &[u8],
    target_hash: &[u8],
    salt: Option<&[u8]>,
    algo: Algorithm,
    found_flag: &Arc<AtomicBool>,
    local_counter: &mut u64,
) {
    if found_flag.load(Ordering::Relaxed) { return; }
    if current_len == target_len {
        if check_candidate_stack(&buffer[0..current_len], target_hash, salt, algo) {
            found_flag.store(true, Ordering::Relaxed);
            println!("MATCH_FOUND:{}", String::from_utf8_lossy(&buffer[0..current_len]));
        }
        *local_counter += 1;
        flush_global_counter(local_counter);
        return;
    }
    for &c in charset {
        buffer[current_len] = c;
        recursive_brute_stack(buffer, current_len + 1, target_len, charset, target_hash, salt, algo, found_flag, local_counter);
        if current_len < 3 && found_flag.load(Ordering::Relaxed) { return; } 
    }
}

fn init_rayon_pool() {
    let threads = std::thread::available_parallelism().map(|parallelism| parallelism.get()).unwrap_or(1);
    let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();
}

fn flush_global_counter(local_counter: &mut u64) {
    if *local_counter >= COUNTER_FLUSH_INTERVAL {
        GLOBAL_COUNTER.fetch_add(*local_counter, Ordering::Relaxed);
        *local_counter = 0;
    }
}

#[cfg(has_hip)]
struct HipLaunchConfig {
    blocks: i32,
    threads: i32,
    batch_size: u64,
}

#[cfg(has_hip)]
fn hip_launch_config(device_info: &HipDeviceInfo) -> HipLaunchConfig {
    let wave_size = device_info.warp_size.max(32);
    let mut threads = device_info.max_threads_per_block.min(256).max(wave_size);
    threads -= threads % wave_size;
    if threads <= 0 {
        threads = wave_size;
    }

    let blocks_per_cu = 24;
    let blocks = (device_info.multiprocessor_count.max(1) * blocks_per_cu).max(1);
    let batch_size = blocks as u64 * threads as u64 * HIP_ITEMS_PER_THREAD * 16_384;

    HipLaunchConfig {
        blocks,
        threads,
        batch_size,
    }
}

#[inline(always)]
fn check_candidate_stack(cand: &[u8], target: &[u8], salt: Option<&[u8]>, algo: Algorithm) -> bool {
    // 1. Raw
    if hash_matches(cand, target, algo) { return true; }
    
    if let Some(s) = salt {
        let mut buf = [0u8; 256];
        let c_len = cand.len();
        let s_len = s.len();
        
        if c_len + s_len > 256 { return false; } 

        // 2. Pass + Salt
        buf[..c_len].copy_from_slice(cand);
        buf[c_len..c_len+s_len].copy_from_slice(s);
        if hash_matches(&buf[..c_len+s_len], target, algo) { return true; }

        // 3. Salt + Pass
        buf[..s_len].copy_from_slice(s);
        buf[s_len..s_len+c_len].copy_from_slice(cand);
        if hash_matches(&buf[..s_len+c_len], target, algo) { return true; }

        // 4. AuthMe (Sha256)
        if let Algorithm::Sha256 = algo {
            let mut hasher = Sha256::new();
            hasher.update(cand);
            let inner_digest = hasher.finalize();
            
            let mut hex_buf = [0u8; 64];
            hex::encode_to_slice(inner_digest, &mut hex_buf).ok();

            if 64 + s_len <= 256 {
                buf[..64].copy_from_slice(&hex_buf);
                buf[64..64+s_len].copy_from_slice(s);
                if hash_matches(&buf[..64+s_len], target, algo) { return true; }
            }
        }
    }
    false
}

#[inline(always)]
fn hash_matches(input: &[u8], target: &[u8], algo: Algorithm) -> bool {
    match algo {
        Algorithm::Md5 => md5::compute(input).as_slice() == target,
        Algorithm::Sha1 => {
            let mut hasher = Sha1::new();
            hasher.update(input);
            hasher.finalize().as_slice() == target
        },
        Algorithm::Sha256 => {
            let mut hasher = Sha256::new();
            hasher.update(input);
            hasher.finalize().as_slice() == target
        },
    }
}
