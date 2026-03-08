# ShadowBreaker Pro

ShadowBreaker Pro is a local desktop front end around a Rust core that can process MD5, SHA-1, and SHA-256 targets using dictionary mode, brute mode, optional GPU acceleration, and an optional distributed worker setup.

## Project Layout

- `shadow_breaker_pro.py`: CustomTkinter desktop GUI.
- `ShadowBreakerPro.desktop`: desktop launcher for Linux desktops.
- `rust_cracker/`: Rust core engine used by the GUI.
- `rust_cracker/hip/sha256.hip`: ROCm HIP SHA-256 GPU kernel.
- `rust_cracker/src/opencl_kernels.rs`: OpenCL kernels used by the Rust core.
- `cache_builder/`: SQLite cache generator for precomputed lookups.
- `grid_server.js`: Socket.IO orchestration server for distributed jobs.
- `index.js`: Node worker/client for grid participation.
- `english-95-extended.txt`: default large wordlist bundled with the project.
- `common_passwords.txt`: smaller bundled wordlist.

## Main Components

### GUI

The GUI is built with `customtkinter` and launches the Rust binary from `rust_cracker/target/release/rust_cracker`.

Features include:

- target hash input
- dictionary and brute-force modes
- GPU enable/disable toggle
- live speed, progress, and status output
- local SQLite cache lookup before launching the Rust core
- AMD and NVIDIA GPU detection in the sidebar
- optional grid worker mode for remote orchestration

### Rust Core

The Rust engine is the main execution backend.

Current behavior:

- detects hash type by input length or `$SHA$...` format
- uses ROCm HIP for SHA-256 brute mode when available
- falls back to OpenCL for dictionary mode when applicable
- falls back to CPU when GPU backends are unavailable or disabled
- prints machine-readable status lines consumed by the GUI

### Cache Builder

`cache_builder/` creates a large SQLite lookup database from a base wordlist and generated variations.

The GUI checks this cache first if `/mnt/backup/shadowbreaker_cache/lookup_v2.db` exists.

### Grid Mode

The Node-based pieces provide basic orchestration for distributing work:

- `grid_server.js` runs the coordinator
- `index.js` acts as a worker client
- the GUI can register as a worker through Socket.IO

## Requirements

### Python GUI

The GUI expects the local virtual environment at `.venv` and currently imports:

- `customtkinter`
- `requests`
- `python-socketio`
- `tkinter`

### Rust Core

- `cargo`
- `rustc`
- ROCm/HIP toolchain for the HIP backend
- OpenCL runtime if you want the OpenCL path available

### AMD GPU Notes

On AMD systems, tools such as `rocminfo` and `rocm-smi` may require the correct device permissions. If GPU detection fails in the GUI or Rust backend, check:

- your user is in the `render` group
- you started a fresh login session after changing group membership
- `rocminfo` works from your shell

## Setup

### 1. Python environment

The GUI is configured to use:

```bash
/mnt/Data/gemini/sha_decryptor/.venv/bin/python
```

If the environment already exists, install or refresh the GUI dependencies with:

```bash
.venv/bin/pip install customtkinter requests 'python-socketio[client]'
```

### 2. Build the Rust engine

From the project root:

```bash
cd rust_cracker
cargo build --release
```

The release binary will be written to:

```text
rust_cracker/target/release/rust_cracker
```

### 3. Optional HIP target override

The build script defaults to `gfx1030`. To override that:

```bash
HIP_OFFLOAD_ARCH=gfx1030 cargo build --release
```

## Running

### Launch the GUI

```bash
/mnt/Data/gemini/sha_decryptor/.venv/bin/python /mnt/Data/gemini/sha_decryptor/shadow_breaker_pro.py
```

### Use the desktop entry

The included desktop file launches the same command:

```text
ShadowBreakerPro.desktop
```

### Run the Rust core directly

Examples:

```bash
cd rust_cracker
./target/release/rust_cracker --target <hash> --mode dictionary --wordlist ../common_passwords.txt
```

```bash
cd rust_cracker
./target/release/rust_cracker --target <hash> --mode brute --length 4 --charset abc123
```

## Runtime Output

The Rust core emits lines that the GUI parses, including:

- `GPU_DETECTED:...`
- `ENGINE:...`
- `STATUS:...`
- `STATS:<speed>|<current>|<total>|<percent>`
- `MATCH_FOUND:<value>`

If you are debugging GUI behavior, these lines are the first thing to inspect.

## Troubleshooting

### GUI does not launch

Check these first:

- `shadow_breaker_pro.py` uses the `.venv` interpreter in the shebang
- `.venv` exists and has the required Python packages
- `python /path/to/shadow_breaker_pro.py` starts without import errors

### GUI says `GPU: Not Detected`

On NVIDIA, verify `nvidia-smi` works.

On AMD, verify:

```bash
rocminfo
rocm-smi --showproductname --showmeminfo vram --json
```

The GUI now tries NVIDIA first, then `rocm-smi`, then `rocminfo`.

### Rust core builds but GPU path is unavailable

Check:

- `hipcc --version`
- ROCm libraries exist under `/opt/rocm/lib` or `/opt/rocm/lib64`
- `rocminfo` can see the GPU

### OpenCL issues

If OpenCL is unavailable, the project may still run through HIP or CPU depending on mode.

## Files You Will Likely Touch Most

- `shadow_breaker_pro.py`
- `rust_cracker/src/main.rs`
- `rust_cracker/hip/sha256.hip`
- `rust_cracker/build.rs`
- `ShadowBreakerPro.desktop`

## Current Known State

Based on the current workspace:

- the GUI launcher path has been fixed to use `.venv`
- the GUI dependencies have been installed into `.venv`
- AMD GPU detection in the GUI has been added through ROCm tools
- the Rust release build completes successfully
- the HIP backend detects the RX 6800 and the release binary runs
