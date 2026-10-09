"""Minimal laptop-side monitor for the ORION Pi: stats and service logs over SSH.

Nothing is installed on the Pi. Two long-lived SSH sessions are used: one runs a
shell loop that reads /proc and /sys with shell builtins, the other follows journalctl.
"""

from __future__ import annotations

import argparse
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

STATS_SCRIPT = r"""
i=0
while :; do
  echo @@
  read -r l < /proc/stat; echo "$l"
  while read -r k v _; do
    case $k in MemTotal:|MemAvailable:|SwapTotal:|SwapFree:) echo "$k $v";; esac
  done < /proc/meminfo
  read -r t < /sys/class/thermal/thermal_zone0/temp && echo "temp $t"
  read -r f < /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq && echo "freq $f"
  if read -r th 2>/dev/null < /sys/devices/platform/soc/soc:firmware/get_throttled; then
    echo "throttled 0x$th"
  else
    vcgencmd get_throttled 2>/dev/null
  fi
  read -r a b c _ < /proc/loadavg; echo "load $a $b $c"
  read -r u _ < /proc/uptime; echo "uptime $u"
  if [ $((i % 5)) -eq 0 ]; then s=$(systemctl is-active "$UNIT" 2>/dev/null); fi
  echo "service $s"
  echo '##'
  i=$((i + 1))
  sleep "$INTERVAL"
done
"""

THROTTLE_BITS = {
    0: "under-voltage",
    1: "freq capped",
    2: "throttled",
    3: "soft temp limit",
}
MAX_LOG_LINES = 2000
LLM_MODELS = (
    ("qwen3.5-2b-q4", "Qwen3.5 2B Q4_K_M", "LLM/Qwen3.5-2B-Q4_K_M.gguf"),
    ("qwen3.5-2b-q8", "Qwen3.5 2B Q8_0", "LLM/Qwen3.5-2B-Q8_0.gguf"),
    ("qwen3-1.7b-q8", "Qwen3 1.7B Q8_0", "LLM/Qwen3-1.7B-Q8_0.gguf"),
    ("phi4-mini-q4", "Phi-4 mini Q4_K_M", "LLM/Phi-4-mini-instruct-Q4_K_M.gguf"),
    ("gemma4-e2b-q8", "Gemma 4 E2B Q8_0", "LLM/gemma-4-E2B-Q8_0.gguf"),
)
REMOTE_PYTHON = "/opt/orion/.venv-voice/bin/orion-model"


def ssh_base(target: str) -> list[str]:
    return [
        "ssh", "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=2",
        target,
    ]


def spawn(cmd: list[str], stdin_text: str | None = None) -> subprocess.Popen:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=flags,
    )
    if stdin_text is not None:
        # Bypass text-mode newline translation; Windows would send CRLF to the remote sh.
        proc.stdin.buffer.write(stdin_text.encode("utf-8"))
        proc.stdin.close()
    return proc


def decode_throttled(value: int) -> tuple[str, str]:
    now = [name for bit, name in THROTTLE_BITS.items() if value & (1 << bit)]
    past = [name for bit, name in THROTTLE_BITS.items() if value & (1 << (bit + 16))]
    return (", ".join(now) or "ok"), (", ".join(past) or "none")


class Monitor:
    def __init__(
        self,
        root: tk.Tk,
        target: str,
        unit: str,
        interval: float,
        model_root: str,
        database_path: str,
        selection_path: str,
        local_model_root: Path,
    ) -> None:
        self.root = root
        self.target = target
        self.unit = unit
        self.interval = interval
        self.model_root = model_root
        self.database_path = database_path
        self.selection_path = selection_path
        self.local_model_root = local_model_root
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.procs: list[subprocess.Popen] = []
        self.stopping = threading.Event()
        self.prev_cpu: list[int] | None = None
        self.paused = tk.BooleanVar(value=False)
        self.metric_history = {
            "CPU": deque(maxlen=360),
            "RAM": deque(maxlen=360),
            "Temp": deque(maxlen=360),
        }
        self.graph_metric = tk.StringVar(value="CPU")
        self.chart_hover_x: float | None = None
        self.log_buffer: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.log_filter = tk.StringVar(value="All")
        self.log_search = tk.StringVar()
        self.model_choices: dict[str, str] = {}
        self.model_sizes: dict[str, int] = {}
        self.model_refreshing = False
        self.model_switching = False

        root.title(f"ORION Pi monitor - {target}")
        root.geometry("1040x760")
        root.minsize(780, 560)
        root.protocol("WM_DELETE_WINDOW", self.close)

        header = ttk.Frame(root, padding=(12, 10))
        header.pack(fill="x")
        self.status = ttk.Label(header, text="Connecting to Pi...", font=("Segoe UI", 10, "bold"))
        self.status.pack(side="left")
        ttk.Button(header, text="Copy snapshot", command=self.copy_snapshot).pack(side="right")

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        overview = ttk.Frame(notebook, padding=10)
        logs = ttk.Frame(notebook, padding=10)
        notebook.add(overview, text="Overview")
        notebook.add(logs, text="Service logs")

        stats = ttk.LabelFrame(overview, text="Live system", padding=10)
        stats.pack(fill="x")
        self.fields: dict[str, ttk.Label] = {}
        names = ["CPU", "RAM", "Swap", "Temp", "Clock", "Load", "Throttle", "Throttle (since boot)", "Uptime", "Service"]
        for idx, name in enumerate(names):
            r, c = divmod(idx, 2)
            ttk.Label(stats, text=f"{name}:", width=22, anchor="e").grid(row=r, column=c * 2, sticky="e")
            value = ttk.Label(stats, text="-", width=34, anchor="w")
            value.grid(row=r, column=c * 2 + 1, sticky="w", padx=(6, 18), pady=2)
            self.fields[name] = value

        model_frame = ttk.LabelFrame(overview, text="Local LLM", padding=8)
        model_frame.pack(fill="x", pady=(10, 0))
        self.model_picker = ttk.Combobox(model_frame, state="readonly", width=38)
        self.model_picker.pack(side="left")
        self.switch_button = ttk.Button(
            model_frame, text="Switch and restart", command=self.confirm_model_switch,
            state="disabled",
        )
        self.switch_button.pack(side="left", padx=8)
        ttk.Button(model_frame, text="Refresh", command=self.refresh_models).pack(side="left")
        self.model_status = ttk.Label(model_frame, text="Checking Pi model files...")
        self.model_status.pack(side="left", padx=12)

        chart_bar = ttk.Frame(overview, padding=(0, 12, 0, 6))
        chart_bar.pack(fill="x")
        ttk.Label(chart_bar, text="History").pack(side="left")
        self.metric_picker = ttk.Combobox(
            chart_bar, textvariable=self.graph_metric, values=("CPU", "RAM", "Temp"),
            state="readonly", width=10,
        )
        self.metric_picker.pack(side="left", padx=8)
        self.metric_picker.bind("<<ComboboxSelected>>", lambda _event: self.draw_chart())
        ttk.Button(chart_bar, text="Clear history", command=self.clear_history).pack(side="right")

        self.chart = tk.Canvas(overview, height=260, background="#f4f6f4", highlightthickness=1,
                               highlightbackground="#cbd3cc")
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _event: self.draw_chart())
        self.chart.bind("<Motion>", self.chart_hover)
        self.chart.bind("<Leave>", self.chart_leave)

        log_bar = ttk.Frame(logs)
        log_bar.pack(fill="x", pady=(0, 8))
        ttk.Label(log_bar, text="Filter").pack(side="left")
        search = ttk.Entry(log_bar, textvariable=self.log_search, width=28)
        search.pack(side="left", padx=(6, 12))
        search.bind("<KeyRelease>", lambda _event: self.render_logs())
        ttk.Label(log_bar, text="Level").pack(side="left")
        level_picker = ttk.Combobox(
            log_bar, textvariable=self.log_filter, values=("All", "Warnings", "Errors"),
            state="readonly", width=12,
        )
        level_picker.pack(side="left", padx=6)
        level_picker.bind("<<ComboboxSelected>>", lambda _event: self.render_logs())
        ttk.Checkbutton(log_bar, text="Pause", variable=self.paused,
                command=self.toggle_log_pause).pack(side="right", padx=8)
        ttk.Button(log_bar, text="Clear", command=self.clear_logs).pack(side="right")

        frame = ttk.Frame(logs)
        frame.pack(fill="both", expand=True)
        self.log = tk.Text(frame, wrap="none", font=("Consolas", 9), bg="#111", fg="#ddd", insertbackground="#ddd")
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.log.pack(side="left", fill="both", expand=True)
        self.log.tag_configure("err", foreground="#f66")
        self.log.tag_configure("warn", foreground="#fc6")

        threading.Thread(target=self.run_stats, daemon=True).start()
        threading.Thread(target=self.run_logs, daemon=True).start()
        self.refresh_models()
        self.root.after(200, self.drain)

    def supervise(self, name: str, make_proc, handle_line) -> None:
        while not self.stopping.is_set():
            proc = make_proc()
            self.procs.append(proc)
            for line in proc.stdout:
                if self.stopping.is_set():
                    break
                handle_line(line.rstrip("\n"))
            proc.wait()
            self.procs.remove(proc)
            if self.stopping.is_set():
                return
            self.events.put(("status", f"{name} disconnected (exit {proc.returncode}); retrying in 5s"))
            self.stopping.wait(5)

    def run_stats(self) -> None:
        script = f"UNIT='{self.unit}'; INTERVAL={self.interval:g}\n" + STATS_SCRIPT
        block: list[str] = []
        in_block = False

        def handle(line: str) -> None:
            nonlocal block, in_block
            if line == "@@":
                block, in_block = [], True
            elif line == "##" and in_block:
                self.events.put(("stats", block))
                in_block = False
            elif in_block:
                block.append(line)
            else:
                self.events.put(("log", f"[stats] {line}"))

        self.supervise("stats", lambda: spawn(ssh_base(self.target) + ["sh", "-s"], script), handle)

    def run_logs(self) -> None:
        cmd = ssh_base(self.target) + [
            "journalctl", "-u", self.unit, "-f", "-n", "200", "-o", "short-iso", "--no-pager",
        ]
        self.supervise("logs", lambda: spawn(cmd), lambda line: self.events.put(("log", line)))

    def refresh_models(self) -> None:
        if self.model_refreshing or self.model_switching:
            return
        self.model_refreshing = True
        self.model_status.configure(text="Checking model files...")
        threading.Thread(target=self.run_model_inventory, daemon=True).start()

    def run_model_inventory(self) -> None:
        remote_command = f"find {shlex.quote(self.model_root)} -type f -printf '%P\\t%s\\n'"
        try:
            proc = spawn(ssh_base(self.target) + [remote_command])
            self.procs.append(proc)
            output = [line.rstrip("\n") for line in proc.stdout]
            return_code = proc.wait()
        except OSError as error:
            self.events.put(("models", ([], {}, f"Model scan failed: {error}")))
            return
        finally:
            if "proc" in locals() and proc in self.procs:
                self.procs.remove(proc)
        remote_sizes: dict[str, int] = {}
        if return_code != 0:
            detail = "Could not read model inventory over SSH: " + " ".join(output[-3:])
            self.events.put(("models", ([], {}, detail)))
            return

        for line in output:
            relative, separator, raw_size = line.partition("\t")
            if separator and raw_size.isdigit():
                remote_sizes[relative] = int(raw_size)

        rows = []
        available: dict[str, str] = {}
        sizes: dict[str, int] = {}
        for model_id, display_name, relative_path in LLM_MODELS:
            remote_size = remote_sizes.get(relative_path)
            local_path = self.local_model_root / relative_path
            try:
                local_size = local_path.stat().st_size
            except OSError:
                local_size = None
            if remote_size is None:
                state = "missing on Pi"
            elif local_size is None:
                state = "local reference missing"
            elif remote_size != local_size:
                state = f"size mismatch ({remote_size:,}/{local_size:,} bytes)"
            else:
                state = f"ready ({remote_size / (1024 ** 3):.1f} GiB)"
                label = f"{model_id}  |  {display_name}"
                available[label] = model_id
                sizes[model_id] = local_size
            rows.append((model_id, state))
        self.events.put(("models", (rows, (available, sizes), "")))

    def show_models(self, payload) -> None:
        rows, inventory, error = payload
        self.model_refreshing = False
        if error:
            self.model_status.configure(text=error)
            self.switch_button.configure(state="disabled")
            return
        self.model_choices, self.model_sizes = inventory
        labels = list(self.model_choices)
        self.model_picker.configure(values=labels)
        if labels:
            if self.model_picker.get() not in self.model_choices:
                self.model_picker.set(labels[0])
            self.switch_button.configure(state="normal")
        else:
            self.model_picker.set("")
            self.switch_button.configure(state="disabled")
        ready_count = len(self.model_choices)
        self.model_status.configure(text=f"{ready_count}/{len(rows)} models size-verified")
        self.append_logs([f"[models] {model_id}: {state}" for model_id, state in rows])

    def confirm_model_switch(self) -> None:
        label = self.model_picker.get()
        model_id = self.model_choices.get(label)
        if model_id is None or self.model_switching:
            return
        if not messagebox.askyesno(
            "Switch LLM model",
            f"Select {model_id} and restart {self.unit}? Active requests will stop.",
            parent=self.root,
        ):
            return
        password = simpledialog.askstring(
            "Pi sudo password", "Enter password to switch model and restart ORION:",
            show="*", parent=self.root,
        )
        if password is None:
            return
        self.model_switching = True
        self.switch_button.configure(state="disabled")
        self.model_status.configure(text=f"Switching to {model_id}; restarting ORION...")
        threading.Thread(
            target=self.run_model_switch, args=(model_id, password), daemon=True
        ).start()

    def run_model_switch(self, model_id: str, password: str) -> None:
        relative_path = next(path for candidate, _name, path in LLM_MODELS if candidate == model_id)
        remote_file = f"{self.model_root}/{relative_path}"
        expected_size = self.model_sizes[model_id]
        sudo_script = (
            "IFS= read -r ORION_SUDO_PASSWORD || exit 1; "
            "printf '%s\\n' \"$ORION_SUDO_PASSWORD\" | sudo -S -p '' -v || exit $?; "
            "unset ORION_SUDO_PASSWORD; "
            "if sudo -n grep -Eq '^[[:space:]]*ORION_LLM_MODEL_ID[[:space:]]*=' /etc/orion/orion.env; then "
            "echo 'ORION_LLM_MODEL_ID in /etc/orion/orion.env overrides UI selection; remove override first.' >&2; exit 3; fi; "
            f"if ! sudo -n test \"$(sudo -n stat -c %s {shlex.quote(remote_file)})\" = {expected_size}; then "
            "echo 'Model file size changed since inventory; refresh model list.' >&2; exit 4; fi; "
            f"sudo -n -u orion-service env HOME=/var/lib/orion "
            f"ORION_DATABASE_PATH={shlex.quote(self.database_path)} "
            f"ORION_MODEL_SELECTION_PATH={shlex.quote(self.selection_path)} "
            f"ORION_MODEL_ROOT={shlex.quote(self.model_root)} "
            f"{REMOTE_PYTHON} use {shlex.quote(model_id)} && "
            f"sudo -n systemctl restart {shlex.quote(self.unit)} && "
            f"sudo -n systemctl is-active --quiet {shlex.quote(self.unit)}"
        )
        try:
            proc = spawn(ssh_base(self.target) + [sudo_script], password + "\n")
            self.procs.append(proc)
            output = [line.rstrip("\n") for line in proc.stdout]
            return_code = proc.wait()
        except OSError as error:
            self.events.put(("model_switch", (False, model_id, f"Could not start SSH: {error}")))
            return
        finally:
            if "proc" in locals() and proc in self.procs:
                self.procs.remove(proc)
        if return_code == 0:
            detail = f"Selected {model_id}; ORION restarted"
            self.events.put(("model_switch", (True, model_id, detail)))
        else:
            detail = "Model switch failed: " + (" ".join(output[-4:]) or f"SSH exit {return_code}")
            self.events.put(("model_switch", (False, model_id, detail)))

    def drain(self) -> None:
        try:
            self.process_events()
        except Exception as exc:
            self.status.configure(text=f"update error: {exc!r}")
        finally:
            if not self.stopping.is_set():
                self.root.after(250, self.drain)

    def process_events(self) -> None:
        latest_stats = None
        logs: list[str] = []
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "stats":
                    latest_stats = payload
                elif kind == "log":
                    logs.append(payload)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "models":
                    self.show_models(payload)
                elif kind == "model_switch":
                    success, model_id, detail = payload
                    self.model_switching = False
                    self.switch_button.configure(state="normal" if self.model_choices else "disabled")
                    self.model_status.configure(text=detail)
                    if success:
                        self.status.configure(text=f"Selected {model_id}; ORION restarted")
                        self.refresh_models()
                    else:
                        messagebox.showerror("Model switch failed", detail, parent=self.root)
        except queue.Empty:
            pass
        if latest_stats is not None:
            self.update_stats(latest_stats)
        if logs:
            self.append_logs(logs)

    def append_logs(self, lines: list[str]) -> None:
        for line in lines:
            self.log_buffer.append(line)
        if not self.paused.get():
            self.render_logs()

    def toggle_log_pause(self) -> None:
        if not self.paused.get():
            self.render_logs()

    def render_logs(self) -> None:
        query = self.log_search.get().casefold().strip()
        level = self.log_filter.get()
        visible = []
        for line in self.log_buffer:
            low = line.casefold()
            is_error = any(word in low for word in ("error", "traceback", "exception", "critical"))
            is_warning = "warn" in low
            if level == "Errors" and not is_error:
                continue
            if level == "Warnings" and not (is_warning or is_error):
                continue
            if query and query not in low:
                continue
            visible.append(line)

        at_bottom = self.log.yview()[1] >= 0.999
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        for line in visible:
            low = line.casefold()
            tag = "err" if any(word in low for word in ("error", "traceback", "exception", "critical")) else (
                "warn" if "warn" in low else ""
            )
            self.log.insert("end", line + "\n", tag)
        self.log.configure(state="disabled")
        if at_bottom:
            self.log.see("end")

    def clear_logs(self) -> None:
        self.log_buffer.clear()
        self.render_logs()

    def clear_history(self) -> None:
        for samples in self.metric_history.values():
            samples.clear()
        self.draw_chart()

    def copy_snapshot(self) -> None:
        snapshot = [f"ORION Pi monitor: {self.target}", f"Captured: {time.strftime('%Y-%m-%d %H:%M:%S')}"]
        snapshot.extend(f"{name}: {field.cget('text')}" for name, field in self.fields.items())
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(snapshot))
        self.status.configure(text="Snapshot copied to clipboard")

    def record_metric(self, name: str, value: float) -> None:
        self.metric_history[name].append((time.time(), value))

    def draw_chart(self) -> None:
        if not hasattr(self, "chart"):
            return
        self.chart.delete("all")
        width = max(self.chart.winfo_width(), 300)
        height = max(self.chart.winfo_height(), 180)
        left, right, top, bottom = 58, width - 18, 24, height - 34
        metric = self.graph_metric.get()
        samples = list(self.metric_history[metric])
        max_value = 100.0
        unit = "%" if metric in {"CPU", "RAM"} else "°C"
        color = {"CPU": "#087f8c", "RAM": "#386cb0", "Temp": "#d65b43"}[metric]

        for tick in range(5):
            y = top + (bottom - top) * tick / 4
            value = max_value * (4 - tick) / 4
            self.chart.create_line(left, y, right, y, fill="#d8dfd9")
            self.chart.create_text(left - 8, y, text=f"{value:.0f}", anchor="e", fill="#536159")
        self.chart.create_text(12, top, text=unit, anchor="nw", fill="#536159")

        if not samples:
            self.chart.create_text(width / 2, height / 2, text="Waiting for samples", fill="#536159")
            return

        points = []
        for index, (_stamp, value) in enumerate(samples):
            x = left + (right - left) * index / max(len(samples) - 1, 1)
            y = bottom - (bottom - top) * min(max(value, 0.0), max_value) / max_value
            points.extend((x, y))
        if len(points) >= 4:
            self.chart.create_line(*points, fill=color, width=2, smooth=True)
        else:
            x, y = points
            self.chart.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline=color)
        self.chart.create_text(left, height - 12, text=time.strftime("%H:%M:%S", time.localtime(samples[0][0])),
                               anchor="w", fill="#536159")
        self.chart.create_text(right, height - 12, text=time.strftime("%H:%M:%S", time.localtime(samples[-1][0])),
                               anchor="e", fill="#536159")

        if self.chart_hover_x is not None:
            index = round((self.chart_hover_x - left) / max(right - left, 1) * (len(samples) - 1))
            index = min(max(index, 0), len(samples) - 1)
            stamp, value = samples[index]
            x = left + (right - left) * index / max(len(samples) - 1, 1)
            y = bottom - (bottom - top) * min(max(value, 0.0), max_value) / max_value
            label = f"{time.strftime('%H:%M:%S', time.localtime(stamp))}  {value:.1f}{unit}"
            self.chart.create_line(x, top, x, bottom, fill="#7d8980", dash=(3, 3))
            self.chart.create_oval(x - 4, y - 4, x + 4, y + 4, fill=color, outline="white", width=1)
            self.chart.create_text(min(max(x, left + 72), right - 72), top + 12, text=label,
                                   fill="#202923", font=("Segoe UI", 9, "bold"))

    def chart_hover(self, event) -> None:
        self.chart_hover_x = event.x
        self.draw_chart()

    def chart_leave(self, _event) -> None:
        self.chart_hover_x = None
        self.draw_chart()

    def set(self, name: str, text: str, color: str | None = None) -> None:
        self.fields[name].configure(text=text, foreground=color or "")

    def update_stats(self, lines: list[str]) -> None:
        data: dict[str, list[str]] = {}
        for line in lines:
            # vcgencmd prints "throttled=0x0"
            parts = line.replace("=", " ", 1).split()
            if parts:
                data[parts[0].rstrip(":")] = parts[1:]

        if "cpu" in data:
            cpu = [int(x) for x in data["cpu"][:8]]
            if self.prev_cpu:
                delta = [a - b for a, b in zip(cpu, self.prev_cpu)]
                total = sum(delta)
                idle = delta[3] + delta[4]
                pct = 100 * (total - idle) / total if total else 0.0
                self.set("CPU", f"{pct:5.1f} %", "red" if pct > 90 else None)
                self.record_metric("CPU", pct)
            self.prev_cpu = cpu

        if "MemTotal" in data and "MemAvailable" in data:
            total = int(data["MemTotal"][0])
            used = total - int(data["MemAvailable"][0])
            pct = 100 * used / total
            self.set("RAM", f"{used / 1024:.0f} / {total / 1024:.0f} MiB ({pct:.0f} %)", "red" if pct > 90 else None)
            self.record_metric("RAM", pct)
        if "SwapTotal" in data and "SwapFree" in data:
            total = int(data["SwapTotal"][0])
            used = total - int(data["SwapFree"][0])
            self.set("Swap", f"{used / 1024:.0f} / {total / 1024:.0f} MiB" if total else "off")

        if "temp" in data:
            temp = int(data["temp"][0]) / 1000
            color = "red" if temp >= 80 else ("orange" if temp >= 70 else None)
            self.set("Temp", f"{temp:.1f} °C", color)
            self.record_metric("Temp", temp)
        if "freq" in data:
            self.set("Clock", f"{int(data['freq'][0]) / 1000:.0f} MHz")
        if "load" in data:
            self.set("Load", " ".join(data["load"]))

        raw = (data.get("throttled") or [""])[0]
        match = re.search(r"0x([0-9a-fA-F]+)", raw)
        if match:
            now, past = decode_throttled(int(match.group(1), 16))
            self.set("Throttle", now, None if now == "ok" else "red")
            self.set("Throttle (since boot)", past, None if past == "none" else "orange")
        else:
            self.set("Throttle", "n/a")

        if "uptime" in data:
            secs = int(float(data["uptime"][0]))
            d, rem = divmod(secs, 86400)
            h, rem = divmod(rem, 3600)
            self.set("Uptime", f"{d}d {h}h {rem // 60}m")
        if "service" in data:
            state = (data["service"] or ["unknown"])[0]
            self.set("Service", state, None if state == "active" else "red")

        self.status.configure(text=f"Live  ·  updated {time.strftime('%H:%M:%S')}")
        self.draw_chart()

    def close(self) -> None:
        self.stopping.set()
        for proc in list(self.procs):
            proc.kill()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight ORION Pi monitor over SSH.")
    parser.add_argument("--target", default="orion.local", help="SSH host (default: orion.local)")
    parser.add_argument("--unit", default="orion.service", help="systemd unit to follow")
    parser.add_argument("--interval", type=float, default=2.0, help="stats poll seconds (default: 2)")
    parser.add_argument("--model-root", default="/opt/orion/models", help="Pi model root")
    parser.add_argument("--database-path", default="/var/lib/orion/orion.sqlite3", help="Pi database path")
    parser.add_argument("--selection-path", help="Pi model-selection.json path; default: next to database")
    parser.add_argument(
        "--local-model-root",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "ORION-MODELS",
        help="local depot used to verify Pi model file sizes",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", args.unit):
        parser.error("--unit must be a systemd unit name ending in .service")
    if not 0.5 <= args.interval <= 60:
        parser.error("--interval must be between 0.5 and 60")
    if args.target.startswith("-"):
        parser.error("--target must be a host name")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", args.model_root):
        parser.error("--model-root must be an absolute path without shell-special characters")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", args.database_path):
        parser.error("--database-path must be an absolute path without shell-special characters")
    if args.selection_path is None:
        args.selection_path = args.database_path.rsplit("/", 1)[0] + "/model-selection.json"
    if not re.fullmatch(r"/[A-Za-z0-9._/-]+", args.selection_path):
        parser.error("--selection-path must be an absolute path without shell-special characters")

    root = tk.Tk()
    Monitor(
        root,
        args.target,
        args.unit,
        args.interval,
        args.model_root,
        args.database_path,
        args.selection_path,
        args.local_model_root,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
