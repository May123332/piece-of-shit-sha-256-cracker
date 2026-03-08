use sha2::{Sha256, Digest};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::fs;
use std::process::Command;

// Configuration
const DB_PATH: &str = "/mnt/backup/shadowbreaker_cache/lookup_v2.db";
const WORDLIST_PATH: &str = "/mnt/Data/gemini/sha_decryptor/english-95-extended.txt";
const TARGET_SIZE_GB: u64 = 500;
const RAM_BUFFER_PATH: &str = "/dev/shm/shadow_chunk.csv";

fn main() -> anyhow::Result<()> {
    println!("--- SHADOWBREAKER CACHE BUILDER (RAM-BUFFER MODE) ---");
    
    // 1. Setup DB Schema (Once)
    // Removed "OFF" pragmas to prevent corruption on HDD
    let setup_sql = "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        CREATE TABLE IF NOT EXISTS hashes (hash TEXT PRIMARY KEY, password TEXT);
    ";
    Command::new("sqlite3").arg(DB_PATH).arg(setup_sql).status()?;

    // 2. Load Base Words
    println!("Loading base wordlist...");
    let file = File::open(WORDLIST_PATH)?;
    let reader = io::BufReader::new(file);
    let base_words: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();
    println!("Loaded {} base words.", base_words.len());

    let symbols = vec!['!', '@', '#', '$', '%', '&', '*', '?', '.', '_'];
    let start_time = Instant::now();
    let mut total_inserted: u64 = 0;

    // Process in massive chunks (e.g. 5000 base words -> ~50 Million variations)
    // 50M rows * 100 bytes = ~5GB CSV. Fits in 128GB RAM easily.
    let chunk_size = 5000; 

    for (chunk_idx, chunk) in base_words.chunks(chunk_size).enumerate() {
        // Stop if size reached
        if fs::metadata(DB_PATH).map(|m| m.len()).unwrap_or(0) > TARGET_SIZE_GB * 1024 * 1024 * 1024 {
            println!("REACHED 500GB LIMIT. STOPPING.");
            break;
        }

        println!("Generating Chunk #{} in RAM...", chunk_idx);
        
        // Generate CSV Data in Parallel
        let mut csv_rows: Vec<String> = Vec::new();
        let mutex = Arc::new(Mutex::new(&mut csv_rows));
        
        chunk.par_iter().for_each(|word| {
            let mut local_rows = Vec::with_capacity(10000);
            
            // 1. Word + Number (0..9999)
            for i in 0..10000 {
                let p = format!("{}{}", word, i);
                let h = hash(&p);
                local_rows.push(format!("{},{}", h, p));
            }
            
            // 2. Word + Symbol + Number (0..99)
            for s in &symbols {
                for i in 0..100 {
                    let p = format!("{}{}{}", word, s, i);
                    let h = hash(&p);
                    local_rows.push(format!("{},{}", h, p));
                }
            }
            
            let mut g = mutex.lock().unwrap();
            g.extend(local_rows);
        });
        
        let row_count = csv_rows.len();
        println!("  -> Generated {} rows. Writing to /dev/shm...", row_count);

        // Write to RAM Disk
        {
            let mut f = File::create(RAM_BUFFER_PATH)?;
            for row in csv_rows {
                writeln!(f, "{}", row)?;
            }
        }
        
        println!("  -> Bulk Importing to HDD (sqlite3 mode)...");
        
        // Write SQL script to file to avoid CLI argument parsing issues
        let sql_script_path = "/dev/shm/import_script.sql";
        let sql_content = format!(
            ".mode csv\n.import \"{}\" hashes",
            RAM_BUFFER_PATH
        );
        
        {
            let mut f = File::create(sql_script_path)?;
            f.write_all(sql_content.as_bytes())?;
        }
        
        // Run sqlite3 with input redirection
        // sqlite3 DB_PATH < import_script.sql
        let status = Command::new("bash")
            .arg("-c")
            .arg(format!("sqlite3 \"{}\" < \"{}\"", DB_PATH, sql_script_path))
            .status()?;
            
        if !status.success() {
            eprintln!("SQLite Import Failed!");
        }
        
        // Cleanup RAM
        fs::remove_file(RAM_BUFFER_PATH).ok();
        fs::remove_file(sql_script_path).ok();
        
        total_inserted += row_count as u64;
        let elapsed = start_time.elapsed().as_secs_f64();
        let rate = total_inserted as f64 / elapsed;
        
        println!("  -> Chunk Complete. Total: {} | Speed: {:.0} H/s", total_inserted, rate);
    }
    
    println!("Indexing...");
    Command::new("sqlite3").arg(DB_PATH).arg("CREATE INDEX IF NOT EXISTS idx_hash ON hashes(hash);").status()?;

    Ok(())
}

fn hash(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s);
    hex::encode(hasher.finalize())
}
