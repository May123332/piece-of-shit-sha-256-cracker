use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=hip/sha256.hip");
    println!("cargo:rerun-if-env-changed=HIP_OFFLOAD_ARCH");
    println!("cargo:rustc-check-cfg=cfg(has_hip)");

    if Command::new("hipcc").arg("--version").output().is_err() {
        println!("cargo:warning=hipcc not found. ROCm HIP support disabled.");
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR must be set"));
    let obj_path = out_dir.join("sha256_hip.o");
    let lib_path = out_dir.join("libhip_cracker.a");
    let arch = env::var("HIP_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1030".to_string());

    let compile_status = Command::new("hipcc")
        .arg("-O3")
        .arg("-ffast-math")
        .arg("-march=native")
        .arg("-mtune=native")
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
    println!("cargo:rustc-link-search=native=/opt/rocm/lib");
    println!("cargo:rustc-link-search=native=/opt/rocm/lib64");
    println!("cargo:rustc-cfg=has_hip");
}
