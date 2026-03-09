use std::env;
use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn find_hipcc(rocm_path: &Path) -> Option<PathBuf> {
    let candidates = [
        env::var_os("HIPCC").map(PathBuf::from),
        Some(rocm_path.join("bin/hipcc")),
        env::var_os("PATH").and_then(|path| {
            env::split_paths(&path)
                .map(|dir| dir.join("hipcc"))
                .find(|candidate| candidate.is_file())
        }),
    ];

    candidates.into_iter().flatten().find(|path| path.is_file())
}

fn main() {
    println!("cargo:rerun-if-changed=hip/sha256.hip");
    println!("cargo:rerun-if-env-changed=HIP_OFFLOAD_ARCH");
    println!("cargo:rerun-if-env-changed=HIPCC");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rustc-check-cfg=cfg(has_hip)");

    let rocm_path = PathBuf::from(env::var_os("ROCM_PATH").unwrap_or_else(|| OsString::from("/opt/rocm")));
    let Some(hipcc_path) = find_hipcc(&rocm_path) else {
        println!(
            "cargo:warning=hipcc not found in HIPCC, {}/bin, or PATH. ROCm HIP support disabled.",
            rocm_path.display()
        );
        return;
    };

    if Command::new(&hipcc_path).arg("--version").output().is_err() {
        println!(
            "cargo:warning=Failed to execute {}. ROCm HIP support disabled.",
            hipcc_path.display()
        );
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let obj_path = out_dir.join("sha256_hip.o");
    let lib_path = out_dir.join("libhip_cracker.a");
    let arch = env::var("HIP_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1030".to_string());
    let include_dir = rocm_path.join("include");

    let compile_status = Command::new(&hipcc_path)
        .arg("-O3")
        .arg("-ffast-math")
        .arg("-march=native")
        .arg("-mtune=native")
        .arg("-I")
        .arg(&include_dir)
        .arg(format!("--offload-arch={arch}"))
        .arg("-c")
        .arg("hip/sha256.hip")
        .arg("-o")
        .arg(&obj_path)
        .status();

    match compile_status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            println!(
                "cargo:warning=hipcc failed to compile HIP kernel (exit code: {status}). ROCm HIP support disabled."
            );
            return;
        }
        Err(err) => {
            println!(
                "cargo:warning=Failed to invoke hipcc ({err}). ROCm HIP support disabled."
            );
            return;
        }
    }

    let ar_tool = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let archive_status = Command::new(ar_tool)
        .arg("crs")
        .arg(&lib_path)
        .arg(&obj_path)
        .status();

    match archive_status {
        Ok(status) if status.success() => {}
        Ok(status) => {
            println!(
                "cargo:warning=Failed to archive HIP object file (exit code: {status}). ROCm HIP support disabled."
            );
            return;
        }
        Err(err) => {
            println!(
                "cargo:warning=Failed to invoke archiver ({err}). ROCm HIP support disabled."
            );
            return;
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=hip_cracker");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rustc-link-search=native={}", rocm_path.join("lib").display());
    println!("cargo:rustc-link-search=native={}", rocm_path.join("lib64").display());
    println!("cargo:rustc-cfg=has_hip");
}
