"""Minimal laptop-side monitor for the ORION Pi: stats and service logs over SSH.

Nothing is installed on the Pi. Two long-lived SSH sessions are used: one runs a
shell loop that reads /proc and /sys with shell builtins, the other follows journalctl.
"""

from __future__ import annotations

import argparse
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

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
    def __init__(self, root: tk.Tk, target: str, unit: str, interval: float) -> None:
        self.root = root
        self.target = target
        self.unit = unit
        self.interval = interval
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.procs: list[subprocess.Popen] = []
        self.stopping = threading.Event()
        self.prev_cpu: list[int] | None = None
        self.paused = tk.BooleanVar(value=False)

        root.title(f"ORION Pi monitor - {target}")
        root.geometry("900x600")
        root.protocol("WM_DELETE_WINDOW", self.close)

        stats = ttk.Frame(root, padding=8)
        stats.pack(fill="x")
        self.fields: dict[str, ttk.Label] = {}
        names = ["CPU", "RAM", "Swap", "Temp", "Clock", "Load", "Throttle", "Throttle (since boot)", "Uptime", "Service"]
        for idx, name in enumerate(names):
            r, c = divmod(idx, 2)
            ttk.Label(stats, text=f"{name}:", width=20, anchor="e").grid(row=r, column=c * 2, sticky="e")
            value = ttk.Label(stats, text="-", width=32, anchor="w")
            value.grid(row=r, column=c * 2 + 1, sticky="w", padx=(4, 16))
            self.fields[name] = value

        bar = ttk.Frame(root, padding=(8, 0))
        bar.pack(fill="x")
        self.status = ttk.Label(bar, text="connecting...")
        self.status.pack(side="left")
        ttk.Button(bar, text="Clear", command=lambda: self.log.delete("1.0", "end")).pack(side="right")
        ttk.Checkbutton(bar, text="Pause logs", variable=self.paused).pack(side="right", padx=8)

        frame = ttk.Frame(root, padding=8)
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
        except queue.Empty:
            pass
        if latest_stats is not None:
            self.update_stats(latest_stats)
        if logs and not self.paused.get():
            self.append_logs(logs)

    def append_logs(self, lines: list[str]) -> None:
        at_bottom = self.log.yview()[1] >= 0.999
        for line in lines:
            low = line.lower()
            tag = "err" if ("error" in low or "traceback" in low or "exception" in low) else (
                "warn" if "warn" in low else ""
            )
            self.log.insert("end", line + "\n", tag)
        excess = int(self.log.index("end-1c").split(".")[0]) - MAX_LOG_LINES
        if excess > 0:
            self.log.delete("1.0", f"{excess + 1}.0")
        if at_bottom:
            self.log.see("end")

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
            self.prev_cpu = cpu

        if "MemTotal" in data and "MemAvailable" in data:
            total = int(data["MemTotal"][0])
            used = total - int(data["MemAvailable"][0])
            pct = 100 * used / total
            self.set("RAM", f"{used / 1024:.0f} / {total / 1024:.0f} MiB ({pct:.0f} %)", "red" if pct > 90 else None)
        if "SwapTotal" in data and "SwapFree" in data:
            total = int(data["SwapTotal"][0])
            used = total - int(data["SwapFree"][0])
            self.set("Swap", f"{used / 1024:.0f} / {total / 1024:.0f} MiB" if total else "off")

        if "temp" in data:
            temp = int(data["temp"][0]) / 1000
            color = "red" if temp >= 80 else ("orange" if temp >= 70 else None)
            self.set("Temp", f"{temp:.1f} °C", color)
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

        self.status.configure(text=f"updated {time.strftime('%H:%M:%S')}")

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
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.@-]+\.service", args.unit):
        parser.error("--unit must be a systemd unit name ending in .service")
    if not 0.5 <= args.interval <= 60:
        parser.error("--interval must be between 0.5 and 60")
    if args.target.startswith("-"):
        parser.error("--target must be a host name")

    root = tk.Tk()
    Monitor(root, args.target, args.unit, args.interval)
    root.mainloop()


if __name__ == "__main__":
    main()
