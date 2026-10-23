# -*- coding: utf-8 -*-
"""本地大模型控制器 —— llama.cpp / Qwen3.6-35B-A3B 一键启动/退出

设计要点：
- 启动用 llama-server.exe，工作目录设为 llama.cpp 根目录（CUDA DLL 在那里）
- 参数 -ngl 99 -ncmoe 24 是实测最优（53.5 tok/s）
- 停止用 taskkill /F /T 整棵进程树，确保显存立即释放
- 后台线程做健康检查，不卡 GUI
"""
import os
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = os.environ.get("LLAMA_ROOT", r"D:\AI\llama.cpp")
EXE = os.path.join(ROOT, "llama-server.exe")
MODEL = os.environ.get("LLAMA_MODEL", r"D:\AI\models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
LOG = os.path.join(ROOT, "_stage", "server.log")
ERR = os.path.join(ROOT, "_stage", "server.err.log")
PORT = 8080
URL = f"http://127.0.0.1:{PORT}"
NCMOE = "24"
CTX = "32768"

CREATE_NO_WINDOW = 0x08000000


# ---------- 工具函数 ----------

def run_capture(args, timeout=15):
    """跑外部命令并返回文本。

    注意：本机控制台代码页是 936(GBK)，tasklist/nvidia-smi 的输出不是 UTF-8。
    先用 OEM 代码页解码，失败再退 UTF-8，最后兜底 replace，绝不抛异常。
    """
    try:
        p = subprocess.run(args, capture_output=True, timeout=timeout,
                           creationflags=CREATE_NO_WINDOW)
    except Exception:
        return ""
    raw = p.stdout or b""
    for enc in ("gbk", "cp936", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def find_server_pids():
    """用 tasklist 找 llama-server.exe 的 PID（不需要额外依赖）。"""
    out = run_capture(
        ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"])
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) >= 2 and parts[0].lower() == "llama-server.exe":
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def http_ok(timeout=3.0):
    """检查 server 的 /health 是否返回 200。"""
    try:
        import urllib.request
        with urllib.request.urlopen(URL + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_memory():
    """返回 (已用MiB, 总共MiB)，失败返回 None。"""
    out = run_capture(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader,nounits"], timeout=10).strip().splitlines()
    try:
        last = out[-1] if out else ""
        used, total = [int(x.strip()) for x in last.split(",")[:2]]
        return used, total
    except Exception:
        return None


def notify(title, msg):
    """弹一个 Windows 提示框（不依赖第三方库）。"""
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, msg, title, 0x40)
    except Exception:
        pass


# ---------- 主界面 ----------

class App:
    def __init__(self, root):
        self.root = root
        self.busy = False
        # 工作线程 -> 主线程的通信队列。tkinter 不是线程安全的，
        # 后台线程绝不直接调 root.after()/改控件，一律投递到这里。
        import queue
        self.ui_queue = queue.Queue()

        root.title("本地大模型控制器 — Qwen3.6-35B-A3B")
        root.geometry("620x400")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        try:
            root.iconbitmap(default="")
        except Exception:
            pass

        # 顶部标题
        header = tk.Frame(root, bg="#1f2937", height=64)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="本地大模型控制器", bg="#1f2937", fg="white",
                 font=("Microsoft YaHei UI", 15, "bold")).pack(side="left", padx=18)
        tk.Label(header, text="RTX 5070 Ti · Qwen3.6-35B-A3B · llama.cpp",
                 bg="#1f2937", fg="#9ca3af",
                 font=("Microsoft YaHei UI", 9)).pack(side="left")

        # 状态区
        body = tk.Frame(root, padx=20, pady=16)
        body.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="检查中…")
        self.status_lbl = tk.Label(body, textvariable=self.status_var,
                                   font=("Microsoft YaHei UI", 20, "bold"),
                                   fg="#6b7280", anchor="w")
        self.status_lbl.pack(fill="x")

        self.detail_var = tk.StringVar(value="")
        tk.Label(body, textvariable=self.detail_var, font=("Consolas", 9),
                 fg="#6b7280", anchor="w", justify="left").pack(fill="x", pady=(6, 0))

        ttk.Separator(body, orient="horizontal").pack(fill="x", pady=14)

        # 按钮区
        btns = tk.Frame(body)
        btns.pack(fill="x")

        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            pass

        # 颜色：正常启用 vs 灰色禁用（tk.Button 的 disabled 状态不会自动变灰，手动管）
        self.C_START_ON, self.C_START_OFF = "#16a34a", "#d1d5db"
        self.C_STOP_ON, self.C_STOP_OFF = "#dc2626", "#d1d5db"
        self.FG_ON, self.FG_OFF = "white", "#6b7280"

        self.btn_start = tk.Button(btns, text="▶  一键启动", command=self.start,
                                   font=("Microsoft YaHei UI", 12, "bold"),
                                   bg=self.C_START_ON, fg=self.FG_ON,
                                   activebackground="#15803d",
                                   activeforeground="white", relief="flat",
                                   width=14, height=2, cursor="hand2")
        self.btn_start.pack(side="left", padx=(0, 10))

        self.btn_stop = tk.Button(btns, text="■  一键退出", command=self.stop,
                                  font=("Microsoft YaHei UI", 12, "bold"),
                                  bg=self.C_STOP_OFF, fg=self.FG_OFF,
                                  activebackground="#b91c1c",
                                  activeforeground="white", relief="flat",
                                  width=14, height=2, cursor="hand2")
        self.btn_stop.pack(side="left", padx=(0, 10))

        self.btn_web = tk.Button(btns, text="打开模型接口", command=self.open_web,
                                 font=("Microsoft YaHei UI", 10),
                                 bg="#e5e7eb", fg="#111827", relief="flat",
                                 width=14, height=2, cursor="hand2")
        self.btn_web.pack(side="left")

        def _paint(btn, enabled, on_color):
            """启用时上色，禁用时置灰，并同步 state。"""
            btn.config(state="normal" if enabled else "disabled",
                       bg=on_color if enabled else "#d1d5db",
                       fg=self.FG_ON if enabled else self.FG_OFF,
                       cursor="hand2" if enabled else "arrow")

        self.paint = _paint

        # 进度条
        self.prog = ttk.Progressbar(body, mode="indeterminate")
        self.prog.pack(fill="x", pady=(16, 0))

        # 底部提示
        self.hint_var = tk.StringVar(
            value=f"参数：-ngl 99 -ncmoe {NCMOE} -c {CTX} -fa on（实测最优档）")
        tk.Label(root, textvariable=self.hint_var, font=("Microsoft YaHei UI", 8),
                 fg="#9ca3af", anchor="w").pack(fill="x", padx=20, pady=(0, 10))

        self.refresh()
        self.drain_queue()

    # ---------- 工作线程 -> 主线程 ----------

    def post(self, fn, *args):
        """线程安全：把回调投递给主线程执行。"""
        self.ui_queue.put((fn, args))

    def drain_queue(self):
        """只在主线程跑：取出后台线程投递的回调并执行。"""
        try:
            while True:
                fn, args = self.ui_queue.get_nowait()
                try:
                    fn(*args)
                except Exception:
                    pass
        except Exception:
            pass
        self.root.after(100, self.drain_queue)

    # ---------- 状态刷新 ----------

    def refresh(self):
        if self.busy:
            self.root.after(800, self.refresh)
            return

        pids = find_server_pids()
        running = bool(pids)
        ready = http_ok() if running else False

        if ready:
            self.status_var.set("● 运行中")
            self.status_lbl.config(fg="#16a34a")
        elif running:
            self.status_var.set("● 启动中…")
            self.status_lbl.config(fg="#d97706")
        else:
            self.status_var.set("○ 已停止")
            self.status_lbl.config(fg="#6b7280")

        parts = []
        if running:
            parts.append(f"PID: {', '.join(map(str, pids))}")
        else:
            parts.append("PID: —")
        mem = gpu_memory()
        if mem:
            parts.append(f"显存: {mem[0]} / {mem[1]} MiB")
        parts.append(f"接口: {URL}")
        self.detail_var.set("   |   ".join(parts))

        # 按钮可用性（禁用时置灰）
        self.paint(self.btn_start, not (running or self.busy), self.C_START_ON)
        self.paint(self.btn_stop, running and not self.busy, self.C_STOP_ON)

        self.root.after(2000, self.refresh)

    def set_busy(self, on, text=""):
        self.busy = on
        if on:
            self.prog.start(12)
            if text:
                self.status_var.set(text)
        else:
            self.prog.stop()
        self.paint(self.btn_start, not (on or find_server_pids()), self.C_START_ON)
        self.paint(self.btn_stop, not on, self.C_STOP_ON)

    # ---------- 启动 ----------

    def start(self):
        if find_server_pids():
            return
        if not os.path.exists(EXE):
            messagebox.showerror("找不到程序", f"未找到：\n{EXE}")
            return
        if not os.path.exists(MODEL):
            messagebox.showerror("找不到模型", f"未找到：\n{MODEL}")
            return

        self.set_busy(True, "● 启动中…（加载模型，约 10 秒）")

        def work():
            try:
                os.makedirs(os.path.dirname(LOG), exist_ok=True)
                fout = open(LOG, "a", encoding="utf-8", errors="replace")
                ferr = open(ERR, "a", encoding="utf-8", errors="replace")
                args = [
                    EXE, "-m", MODEL,
                    "-ngl", "99", "-ncmoe", NCMOE,
                    "-c", CTX,
                    "-fa", "on",
                    "-ctk", "q8_0", "-ctv", "q8_0",
                    "-t", "16",
                    "--host", "127.0.0.1", "--port", str(PORT),
                    "--no-webui",
                ]
                subprocess.Popen(args, cwd=ROOT, stdout=fout, stderr=ferr,
                                 creationflags=CREATE_NO_WINDOW)
                # 等就绪，最多 120 秒
                for _ in range(120):
                    time.sleep(1)
                    if http_ok(timeout=2):
                        self.post(self._started_ok)
                        return
                    if not find_server_pids():
                        break
                self.post(self._started_fail)
            except Exception as e:
                msg = str(e)
                self.post(self._started_fail, msg)

        threading.Thread(target=work, daemon=True).start()

    def _started_ok(self):
        self.set_busy(False)
        self.status_var.set("● 运行中")
        self.status_lbl.config(fg="#16a34a")
        notify("启动成功", "本地大模型已就绪。\n\n在 DSH 里选择模型：\n"
                          "Qwen3.6-35B-A3B (本地)")

    def _started_fail(self, err=""):
        self.set_busy(False)
        tail = ""
        try:
            with open(ERR, "r", encoding="utf-8", errors="replace") as f:
                tail = "".join(f.readlines()[-12:])
        except Exception:
            pass
        messagebox.showerror("启动失败",
                             (f"{err}\n\n" if err else "") +
                             f"最后日志：\n{tail or '(无)'}\n\n完整日志：\n{ERR}")

    # ---------- 停止 ----------

    def stop(self, skip_confirm=False):
        pids = find_server_pids()
        if not pids:
            return
        if not skip_confirm and not messagebox.askyesno(
                "确认退出",
                f"停止本地大模型服务？\n\n将结束进程 {pids}，"
                f"并释放约 11.6 GB 显存。"):
            return
        self.set_busy(True, "● 正在退出…")

        def work():
            for pid in pids:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, creationflags=CREATE_NO_WINDOW)
            for _ in range(20):
                time.sleep(0.5)
                if not find_server_pids():
                    break
            self.post(self._stopped)

        threading.Thread(target=work, daemon=True).start()

    def _stopped(self):
        self.set_busy(False)
        self.status_var.set("○ 已停止")
        self.status_lbl.config(fg="#6b7280")
        mem = gpu_memory()
        if mem:
            self.hint_var.set(f"已释放显存 — 当前占用 {mem[0]} / {mem[1]} MiB，可以开游戏了")
        if getattr(self, "_closing", False):
            self.root.destroy()

    # ---------- 关闭窗口 ----------

    def on_close(self):
        """关窗口时如果模型还在跑，问一下要不要一起停掉。"""
        if find_server_pids():
            ans = messagebox.askyesnocancel(
                "关闭控制器",
                "本地大模型仍在运行。\n\n"
                "选「是」= 停止模型并关闭（释放显存）\n"
                "选「否」= 保持模型运行，只关闭这个窗口\n"
                "选「取消」= 什么都不做")
            if ans is None:
                return                      # 取消
            if ans is True:
                # 先显示退出中，等真正停干净再由 _stopped() 关窗口，
                # 否则用户看不到任何反馈，会以为没生效。
                self._closing = True
                self.btn_start.config(state="disabled")
                self.set_busy(True, "● 正在退出…（释放显存）")
                self.stop(skip_confirm=True)
                return
            # ans is False -> 保持模型运行，直接关窗口
        self.root.destroy()

    # ---------- 打开接口 ----------

    def open_web(self):
        import webbrowser
        webbrowser.open(URL + "/v1/models")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
