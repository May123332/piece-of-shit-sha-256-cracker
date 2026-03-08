#!/mnt/Data/gemini/sha_decryptor/.venv/bin/python
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import hashlib
import threading
import itertools
import string
import time
import os
import sys
import subprocess
import signal
import sqlite3
import json
import shutil
import socketio
import requests

# Set Theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class NetworkWorker:
    def __init__(self, server_url, worker_name, log_callback, on_status_change):
        self.server_url = server_url
        self.worker_name = worker_name
        self.log = log_callback
        self.on_status_change = on_status_change
        self.sio = socketio.Client()
        self.running = False
        self.current_proc = None

        # Callbacks
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('task', self.on_task)

    def start(self):
        try:
            self.running = True
            self.log(f"Connecting to {self.server_url}...")
            self.sio.connect(self.server_url)
        except Exception as e:
            self.log(f"Connection failed: {e}")
            self.on_status_change("Connection Failed")
            self.running = False

    def stop(self):
        self.running = False
        if self.sio.connected:
            self.sio.disconnect()
        self.kill_proc()
        self.on_status_change("Disconnected")

    def kill_proc(self):
        if self.current_proc:
            try:
                self.current_proc.terminate()
                self.current_proc.wait()
            except:
                pass
            self.current_proc = None

    def on_connect(self):
        self.log("Connected to Grid Orchestrator!")
        self.on_status_change("Connected / Idle")
        
        # Register capabilities
        specs = {
            'name': self.worker_name,
            'cores': os.cpu_count(),
            'platform': sys.platform,
            'hasLLM': False, 
            'capabilities': ['hash-crack']
        }
        self.sio.emit('register', specs)

    def on_disconnect(self):
        self.log("Disconnected from server.")
        self.on_status_change("Disconnected")

    def on_task(self, task):
        if not self.running: return
        
        self.log(f"Received Task: {task.get('type')} (Job {task.get('jobId')})")
        
        if task.get('type') == 'hash-crack':
            self.handle_hash_crack(task)
        else:
            self.log(f"Ignoring unknown task type: {task.get('type')}")

    def handle_hash_crack(self, task):
        payload = task.get('payload', {})
        job_id = task.get('jobId')
        
        target_hash = payload.get('targetHash')
        charset = payload.get('charset', 'abcdefghijklmnopqrstuvwxyz0123456789')
        length = payload.get('length', 4)
        prefix = payload.get('prefix', '')
        
        self.on_status_change(f"Working on Job #{job_id} (Prefix: '{prefix}')")
        self.log(f"Starting Job #{job_id}: Len {length}, Prefix '{prefix}'")

        rust_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rust_cracker/target/release/rust_cracker")
        
        cmd = [
            rust_bin, 
            "--target", target_hash,
            "--mode", "brute",
            "--length", str(length),
            "--charset", charset,
            "--prefix", prefix
        ]

        try:
            start_time = time.time()
            self.current_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            found_secret = None
            while self.running:
                line = self.current_proc.stdout.readline()
                if not line and self.current_proc.poll() is not None:
                    break
                if line:
                    line = line.strip()
                    if line.startswith("MATCH_FOUND:"):
                        found_secret = line.split(":", 1)[1]
                        self.current_proc.terminate()
                        break
            
            duration = (time.time() - start_time) * 1000
            result = {'found': found_secret is not None, 'secret': found_secret}
            
            self.log(f"Job #{job_id} Finished. Found: {result['found']} ({duration:.0f}ms)")
            self.sio.emit('result', {'jobId': job_id, 'data': result})
            self.on_status_change("Connected / Idle")

        except Exception as e:
            self.log(f"Job Execution Error: {e}")
            self.on_status_change("Job Failed")
        finally:
            self.kill_proc()


class ShadowBreakerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.title("ShadowBreaker Pro")
        self.geometry("950x750")
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Handle Window Close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Variables
        self.hash_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Dictionary")
        self.wordlist_path = tk.StringVar(value="/mnt/Data/gemini/sha_decryptor/english-95-extended.txt")
        self.status_var = tk.StringVar(value="Ready to initialize.")
        self.is_running = False
        self.brute_len = tk.IntVar(value=4)
        self.charset_var = tk.StringVar(value=string.ascii_letters + string.digits + "!@#$%^&*?")
        self.gpu_enabled = tk.BooleanVar(value=True)
        self.vram_info = tk.StringVar(value="VRAM: Checking...")
        
        # Network Vars
        self.net_server_url = tk.StringVar(value="http://localhost:3000")
        self.net_worker_name = tk.StringVar(value=f"Worker-{os.getpid()}")
        self.net_status = tk.StringVar(value="Disconnected")
        self.net_worker = None
        
        # Grid Controller Vars
        self.grid_workers = []
        self.grid_jobs = []
        self.grid_socket = None
        
        # Database Connection (Persistent)
        self.db_conn = None
        self.connect_db()
        
        # Process handle
        self.process = None
        self.sound_process = None

        self.create_sidebar()
        self.create_main_view()
        self.create_console()
        
        # Start VRAM monitor
        self.update_vram()

    def connect_db(self):
        db_path = "/mnt/backup/shadowbreaker_cache/lookup_v2.db"
        if os.path.exists(db_path):
            try:
                self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
                # Optimize read performance
                self.db_conn.execute("PRAGMA journal_mode = WAL")
                self.db_conn.execute("PRAGMA synchronous = NORMAL")
                self.db_conn.execute("PRAGMA cache_size = -100000") # 100MB RAM Cache
                self.db_conn.execute("PRAGMA query_only = 1") # Safety
            except:
                self.db_conn = None

    def on_closing(self):
        self.stop_cracking()
        if self.db_conn:
            self.db_conn.close()
        if self.net_worker:
            self.net_worker.stop()
        self.create_main_view()
        self.create_console()
        
        # Start VRAM monitor
        self.update_vram()

    def update_vram(self):
        info = self._probe_nvidia_vram() or self._probe_rocm_vram() or self._probe_rocminfo_gpu()
        if info:
            self.vram_info.set(info["text"])
            self.vram_lbl.configure(text_color=info["color"])
        else:
            self.vram_info.set("GPU: Not Detected")
            self.vram_lbl.configure(text_color="#ef4444")

        self.after(5000, self.update_vram)

    def _gpu_text_color(self, free_mib):
        if free_mib > 4096:
            return "#10b981"
        if free_mib > 1024:
            return "#f59e0b"
        return "#ef4444"

    def _probe_nvidia_vram(self):
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return None
        try:
            result = subprocess.check_output(
                [nvidia_smi, "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
            if not result:
                return None
            name, free_raw, total_raw = [part.strip() for part in result.split(",", 2)]
            free_mib = int(free_raw)
            total_mib = int(total_raw)
            percent_free = (free_mib / total_mib) * 100 if total_mib else 0
            return {
                "text": f"VRAM Free: {free_mib} MiB ({percent_free:.1f}%) [{name}]",
                "color": self._gpu_text_color(free_mib),
            }
        except Exception:
            return None

    def _probe_rocm_vram(self):
        rocm_smi = shutil.which("rocm-smi")
        if not rocm_smi and os.path.exists("/opt/rocm/bin/rocm-smi"):
            rocm_smi = "/opt/rocm/bin/rocm-smi"
        if not rocm_smi:
            return None
        try:
            result = subprocess.check_output(
                [rocm_smi, "--showproductname", "--showmeminfo", "vram", "--json"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            json_start = result.find("{")
            if json_start == -1:
                return None
            payload = json.loads(result[json_start:])
            if not payload:
                return None
            card = next(iter(payload.values()))
            total_bytes = int(card.get("VRAM Total Memory (B)", 0) or 0)
            used_bytes = int(card.get("VRAM Total Used Memory (B)", 0) or 0)
            free_bytes = max(total_bytes - used_bytes, 0)
            free_mib = free_bytes // (1024 * 1024)
            total_mib = total_bytes // (1024 * 1024)
            percent_free = (free_bytes / total_bytes) * 100 if total_bytes else 0
            gpu_name = card.get("Card Series", "AMD GPU")
            return {
                "text": f"VRAM Free: {free_mib} MiB ({percent_free:.1f}%) [{gpu_name}]",
                "color": self._gpu_text_color(free_mib),
            }
        except Exception:
            return None

    def _probe_rocminfo_gpu(self):
        rocminfo = shutil.which("rocminfo")
        if not rocminfo and os.path.exists("/opt/rocm/bin/rocminfo"):
            rocminfo = "/opt/rocm/bin/rocminfo"
        if not rocminfo:
            return None
        try:
            result = subprocess.check_output([rocminfo], text=True, stderr=subprocess.STDOUT)
            for line in result.splitlines():
                if "Marketing Name:" in line:
                    gpu_name = line.split(":", 1)[1].strip()
                    if gpu_name and "Intel" not in gpu_name and "CPU" not in gpu_name:
                        return {"text": f"GPU: {gpu_name}", "color": "#10b981"}
        except Exception:
            return None
        return None

    def on_closing(self):
        self.stop_cracking()
        if self.net_worker: self.net_worker.stop()
        if self.grid_socket and self.grid_socket.connected: self.grid_socket.disconnect()
        self.stop_sound()
        try:
            if self.process:
                self.process.terminate()
                self.process.kill()
        except: pass
        self.destroy()
        sys.exit(0)

    def create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="SHADOW\nBREAKER", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.ver_label = ctk.CTkLabel(self.sidebar_frame, text="v4.6 // GRID MASTER", font=ctk.CTkFont(size=12), text_color="gray")
        self.ver_label.grid(row=1, column=0, padx=20, pady=(0, 20))

        self.vram_lbl = ctk.CTkLabel(self.sidebar_frame, textvariable=self.vram_info, font=ctk.CTkFont(size=11, weight="bold"))
        self.vram_lbl.grid(row=2, column=0, padx=20, pady=0)
        
        self.gpu_switch = ctk.CTkSwitch(self.sidebar_frame, text="GPU Acceleration", variable=self.gpu_enabled)
        self.gpu_switch.grid(row=3, column=0, padx=20, pady=10)

        self.stats_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="#374151")
        self.stats_frame.grid(row=4, column=0, padx=10, pady=(10,0), sticky="ew")
        self.speed_lbl = ctk.CTkLabel(self.stats_frame, text="0 H/s", font=ctk.CTkFont(size=13, weight="bold"), text_color="#10b981")
        self.speed_lbl.pack(pady=(5,0))
        self.total_lbl = ctk.CTkLabel(self.stats_frame, text="Total: 0", font=ctk.CTkFont(size=11), text_color="gray")
        self.total_lbl.pack(pady=(0,5))
        self.progress_val = ctk.CTkProgressBar(self.stats_frame, height=8, progress_color="#10b981")
        self.progress_val.pack(fill="x", padx=10, pady=(0, 10))
        self.progress_val.set(0)

        self.mode_label = ctk.CTkLabel(self.sidebar_frame, text="OPERATION MODE", anchor="w", font=ctk.CTkFont(size=12, weight="bold"))
        self.mode_label.grid(row=5, column=0, padx=20, pady=(20, 0))
        
        self.app_mode = tk.StringVar(value="Local Cracker")
        self.mode_menu_main = ctk.CTkOptionMenu(self.sidebar_frame, values=["Local Cracker", "Grid Worker", "Grid Controller"], 
                                         command=self.change_app_mode, variable=self.app_mode)
        self.mode_menu_main.grid(row=6, column=0, padx=20, pady=10)

        self.config_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.config_frame.grid(row=7, column=0, padx=10, pady=10, sticky="nsew")
        
        # Sub-frames for configs
        self.local_config_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        self.local_mode_menu = ctk.CTkOptionMenu(self.local_config_frame, values=["Dictionary", "Brute Force"], command=self.change_local_mode, variable=self.mode_var)
        self.wordlist_btn = ctk.CTkButton(self.local_config_frame, text="Load Wordlist", command=self.browse_wordlist, fg_color="#4b5563")
        self.brute_slider = ctk.CTkSlider(self.local_config_frame, from_=1, to=12, number_of_steps=11, variable=self.brute_len, command=self.update_slider_lbl)
        self.brute_val_lbl = ctk.CTkLabel(self.local_config_frame, text="4", font=ctk.CTkFont(size=14, weight="bold"))

        self.net_config_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        self.server_entry = ctk.CTkEntry(self.net_config_frame, textvariable=self.net_server_url)
        self.name_entry = ctk.CTkEntry(self.net_config_frame, textvariable=self.net_worker_name)
        self.net_btn = ctk.CTkButton(self.net_config_frame, text="CONNECT", command=self.toggle_network)

        self.grid_config_frame = ctk.CTkFrame(self.config_frame, fg_color="transparent")
        self.grid_server_entry = ctk.CTkEntry(self.grid_config_frame, textvariable=self.net_server_url)
        self.grid_btn = ctk.CTkButton(self.grid_config_frame, text="LOGIN AS MASTER", fg_color="#10b981", command=self.toggle_grid_controller)

        self.update_sidebar_config()

    def create_main_view(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.header = ctk.CTkLabel(self.main_frame, text="Decryption Dashboard", font=ctk.CTkFont(size=28, weight="bold"))
        self.header.pack(anchor="w", pady=(0, 20))

        self.local_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.local_view.pack(fill="both", expand=True)
        self.input_card = ctk.CTkFrame(self.local_view)
        self.input_card.pack(fill="x", pady=10)
        self.hash_entry = ctk.CTkEntry(self.input_card, height=40, font=("Consolas", 14), textvariable=self.hash_var)
        self.hash_entry.pack(fill="x", padx=15, pady=15)
        self.hash_entry.bind("<Control-a>", lambda e: (self.hash_entry.select_range(0, 'end'), self.hash_entry.icursor('end'), 'break')[2])

        self.action_frame = ctk.CTkFrame(self.local_view, fg_color="transparent")
        self.action_frame.pack(fill="x", pady=20)
        self.start_btn = ctk.CTkButton(self.action_frame, text="INITIATE DECRYPTION", height=50, command=self.start_cracking, fg_color="#7c3aed")
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.stop_btn = ctk.CTkButton(self.action_frame, text="ABORT", height=50, command=self.stop_cracking, fg_color="#ef4444", state="disabled")
        self.stop_btn.pack(side="right", fill="x", expand=True, padx=(10, 0))

        self.grid_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.workers_count_lbl = ctk.CTkLabel(self.grid_view, text="Connected Workers: 0", font=ctk.CTkFont(size=14, weight="bold"))
        self.workers_count_lbl.pack(anchor="w", pady=10)
        
        self.grid_input_card = ctk.CTkFrame(self.grid_view)
        self.grid_input_card.pack(fill="x", pady=10)
        self.grid_hash_entry = ctk.CTkEntry(self.grid_input_card, placeholder_text="Target Hash...", height=35)
        self.grid_hash_entry.pack(fill="x", padx=15, pady=10)
        
        self.grid_start_btn = ctk.CTkButton(self.grid_input_card, text="START GRID ATTACK", height=40, fg_color="#10b981", command=self.start_grid_job)
        self.grid_start_btn.pack(fill="x", padx=15, pady=15)

        self.worker_list_box = ctk.CTkTextbox(self.grid_view, height=200, fg_color="#1e1e1e")
        self.worker_list_box.pack(fill="both", expand=True, pady=10)

        self.progress_bar = ctk.CTkProgressBar(self.main_frame)
        self.progress_bar.pack(fill="x", pady=(10, 0))
        self.progress_bar.set(0)
        self.status_lbl = ctk.CTkLabel(self.main_frame, textvariable=self.status_var, text_color="gray")
        self.status_lbl.pack(anchor="e", pady=5)

    def create_console(self):
        self.console_frame = ctk.CTkFrame(self.main_frame)
        self.console_frame.pack(fill="both", expand=True, pady=(20, 0))
        self.log_box = ctk.CTkTextbox(self.console_frame, font=("Consolas", 12), text_color="#10b981", fg_color="#1e1e1e")
        self.log_box.pack(fill="both", expand=True, padx=5, pady=5)

    def change_app_mode(self, mode):
        self.local_view.pack_forget()
        self.grid_view.pack_forget()
        self.update_sidebar_config()
        if mode == "Grid Worker":
            self.header.configure(text="Grid Compute Node")
            self.local_view.pack(fill="both", expand=True)
            self.input_card.pack_forget()
            self.start_btn.configure(state="disabled", text="WAITING FOR TASKS")
        elif mode == "Grid Controller":
            self.header.configure(text="Grid Master Control")
            self.grid_view.pack(fill="both", expand=True)
        else:
            self.header.configure(text="Decryption Dashboard")
            self.local_view.pack(fill="both", expand=True)
            self.input_card.pack(fill="x", pady=10, before=self.action_frame)
            self.start_btn.configure(state="normal", text="INITIATE DECRYPTION")

    def update_sidebar_config(self):
        for widget in self.config_frame.winfo_children(): widget.pack_forget()
        mode = self.app_mode.get()
        if mode == "Grid Worker":
            self.net_config_frame.pack(fill="x")
            self.server_entry.pack(fill="x", pady=5)
            self.name_entry.pack(fill="x", pady=5)
            self.net_btn.pack(fill="x", pady=10)
        elif mode == "Grid Controller":
            self.grid_config_frame.pack(fill="x")
            self.grid_server_entry.pack(fill="x", pady=5)
            self.grid_btn.pack(fill="x", pady=10)
        else:
            self.local_config_frame.pack(fill="x")
            self.local_mode_menu.pack(fill="x", pady=5)
            if self.mode_var.get() == "Dictionary": self.wordlist_btn.pack(fill="x", pady=5)
            else:
                self.brute_slider.pack(fill="x", pady=5)
                self.brute_val_lbl.pack()

    def toggle_network(self):
        if self.net_worker and self.net_worker.running:
            self.net_worker.stop()
            self.net_btn.configure(text="CONNECT", fg_color="#7c3aed")
        else:
            self.net_worker = NetworkWorker(self.net_server_url.get(), self.net_worker_name.get(), self.log, lambda msg: self.net_status.set(msg))
            threading.Thread(target=self.net_worker.start, daemon=True).start()
            self.net_btn.configure(text="DISCONNECT", fg_color="#ef4444")

    def toggle_grid_controller(self):
        if self.grid_socket and self.grid_socket.connected:
            self.grid_socket.disconnect()
            self.grid_btn.configure(text="LOGIN AS MASTER", fg_color="#10b981")
        else:
            url = self.net_server_url.get()
            self.log(f"Connecting to Master at {url}...")
            self.grid_socket = socketio.Client()
            @self.grid_socket.on('worker-update')
            def on_workers(data):
                self.grid_workers = data
                self.update_grid_ui()
            @self.grid_socket.on('job-result')
            def on_result(data):
                if data.get('data', {}).get('found'):
                    res = data['data']['secret']
                    self.log(f"!!! GRID FOUND: {res} !!!", "success")
                    self.play_sound(True)
                    messagebox.showinfo("GRID SUCCESS", f"Found: {res}")
                    self.stop_sound()
            try:
                self.grid_socket.connect(url)
                self.grid_btn.configure(text="SIGNOUT", fg_color="#ef4444")
            except Exception as e: self.log(f"Error: {e}")

    def update_grid_ui(self):
        self.workers_count_lbl.configure(text=f"Connected Workers: {len(self.grid_workers)}")
        self.worker_list_box.delete("1.0", "end")
        for w in self.grid_workers:
            self.worker_list_box.insert("end", f"[{'WORKING' if w.get('currentTask') else 'IDLE'}] {w['name']} ({w.get('specs',{}).get('cores','?')} Cores)\n")

    def start_grid_job(self):
        url = self.net_server_url.get() + "/api/job"
        target = self.grid_hash_entry.get().strip()
        if not target: return
        try:
            res = requests.post(url, json={"type": "hash-crack", "payload": {"targetHash": target, "charset": self.charset_var.get(), "length": self.grid_len_var.get(), "algorithm": "sha256"}})
            if res.status_code == 200: self.log(f"Grid Job Sent: {res.json().get('jobId')}")
        except Exception as e: self.log(f"Error: {e}")

    def update_slider_lbl(self, value): self.brute_val_lbl.configure(text=f"{int(value)}")
    def change_local_mode(self, mode): self.mode_var.set(mode); self.update_sidebar_config()
    def browse_wordlist(self):
        f = filedialog.askopenfilename(); 
        if f: self.wordlist_path.set(f); self.log(f"Loaded: {os.path.basename(f)}")
    def log(self, message, msg_type="info"):
        self.log_box.insert("end", f"[{time.strftime('%H:%M:%S')}] {message}\n")
        if self.log_box.yview()[1] == 1.0: self.log_box.see("end")

    def play_sound(self, success=True):
        self.stop_sound()
        try:
            s = "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga" if success else "/usr/share/sounds/freedesktop/stereo/dialog-error.oga"
            if os.path.exists(s):
                if success: self.sound_process = subprocess.Popen(f"while true; do paplay '{s}'; done", shell=True, preexec_fn=os.setsid)
                else: subprocess.Popen(["paplay", s])
        except: pass

    def stop_sound(self):
        if self.sound_process:
            try: os.killpg(os.getpgid(self.sound_process.pid), signal.SIGTERM); self.sound_process = None
            except: pass

    def check_precomputed(self, target_hash):
        if not self.db_conn:
            self.connect_db()
            if not self.db_conn: return None
        
        try:
            # Check for $SHA$ format
            clean_hash = target_hash.split("$")[3] if target_hash.startswith("$SHA$") else target_hash
            
            # Set a busy timeout for this query to avoid hanging if DB is locked by writer
            # But python sqlite3 doesn't have per-query timeout easily without busy_handler
            # We rely on WAL mode allowing concurrent reads.
            
            cursor = self.db_conn.cursor()
            cursor.execute("SELECT password FROM hashes WHERE hash = ?", (clean_hash.lower(),))
            result = cursor.fetchone()
            
            return result[0] if result else None
        except Exception as e: 
            # If DB is locked or busy, just fail fast
            self.log(f"Cache check skipped: {e}")
            return None

    def start_cracking(self):
        if self.is_running: return
        t = self.hash_var.get().strip()
        if not t: return
        self.is_running = True
        self.start_btn.configure(state="disabled"); self.stop_btn.configure(state="normal")
        self.progress_bar.start(); self.log("--- STARTING RUST ENGINE ---")
        threading.Thread(target=self.run_process, args=(t,), daemon=True).start()

    def stop_cracking(self):
        self.is_running = False; self.status_var.set("Aborting...")
        if self.process: self.process.terminate()

    def reset_ui(self):
        self.is_running = False; self.start_btn.configure(state="normal"); self.stop_btn.configure(state="disabled")
        self.progress_bar.stop(); self.progress_bar.set(0)

    def run_process(self, raw_hash):
        self.log("Checking cache..."); res = self.check_precomputed(raw_hash)
        if res: self.play_sound(True); self.success(res, "Cache"); self.reset_ui(); return
        
        rb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rust_cracker/target/release/rust_cracker")
        cmd = [rb, "--target", raw_hash]
        if not self.gpu_enabled.get(): cmd.append("--no-gpu")
        
        if self.mode_var.get() == "Dictionary":
            wp = self.wordlist_path.get(); cmd.extend(["--mode", "dictionary", "--wordlist", wp])
            try:
                out = subprocess.check_output(["wc", "-l", wp], text=True)
                cmd.extend(["--total", str(int(out.split()[0]))])
            except: pass
        else:
            l = self.brute_len.get(); cs = self.charset_var.get(); cmd.extend(["--mode", "brute", "--length", str(l), "--charset", cs])
            total = sum([len(cs)**i for i in range(1, l+1)])
            cmd.extend(["--total", str(min(total, 18446744073709551615))])

        try:
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            match = False
            while self.is_running:
                line = self.process.stdout.readline()
                if not line and self.process.poll() is not None: break
                if line:
                    line = line.strip()
                    if line.startswith("MATCH_FOUND:"):
                        pwd = line.split(":", 1)[1]; match = True
                        self.process.terminate(); self.process.wait()
                        self.play_sound(True); self.success(pwd, "Rust Engine"); return
                    elif line.startswith("GPU_DETECTED:"):
                        gpu_name = line.split(":", 1)[1].strip()
                        self.vram_info.set(f"GPU: {gpu_name}")
                        self.vram_lbl.configure(text_color="#10b981")
                        self.log(f"[CORE] {line}")
                    elif line.startswith("ENGINE:"):
                        self.status_var.set(line.split(":", 1)[1].strip())
                        self.log(f"[CORE] {line}")
                    elif line.startswith("STATS:"):
                        p = line.split(":", 1)[1].split("|")
                        if len(p) >= 4:
                            self.speed_lbl.configure(text=p[0])
                            cur = int(p[1].replace(',','')); tot = int(p[2].replace(',','')); pct = float(p[3])
                            def f(n): return f"{n/1e9:.1f}B" if n>1e9 else f"{n/1e6:.1f}M" if n>1e6 else str(n)
                            self.total_lbl.configure(text=f"{f(cur)} / {f(tot)} ({pct:.1f}%)")
                            self.progress_val.set(pct/100.0)
                    elif line.startswith("STATUS:") or line.startswith("Checking length"):
                        self.log(line)
                    else:
                        self.log(f"[CORE] {line}")
            if not match and self.is_running: self.log("Not found."); self.play_sound(False)
        except Exception as e: self.log(f"Error: {e}")
        finally: self.reset_ui(); self.status_var.set("Idle")

    def success(self, pwd, method):
        self.log(f"MATCH FOUND: {pwd} ({method})", "success")
        self.status_var.set(f"FOUND: {pwd}")
        try:
            h = self.hash_var.get().strip()
            sh = "".join([c for c in h if c.isalnum()])[:32]
            with open(f"cracked_{sh}.txt", "w") as f:
                f.write(f"Date: {time.ctime()}\nHash: {h}\nPass: {pwd}\nMethod: {method}\n")
        except: pass
        messagebox.showinfo("CRACKED", f"Password: {pwd}")
        self.stop_sound()

if __name__ == "__main__":
    app = ShadowBreakerApp()
    app.mainloop()