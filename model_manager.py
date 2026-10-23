# -*- coding: utf-8 -*-
"""本地大模型管理器 —— 一个程序管理所有 GGUF 模型的启动/停止。

功能：
  1. 自动扫描模型目录（默认 D:\\AI\\models），识别 .gguf
  2. 读 GGUF 头部自动识别架构（MoE / dense / 非文本模型）
  3. 按实时空闲显存自动推荐 -ncmoe / -ngl / 上下文
  4. 每个模型独立保存参数，一键启动 / 一键停止
  5. 显示运行状态与显存占用

设计约束：
  - 12GB 显存一次只能跑一个大模型 → 同一时刻只允许一个服务在跑
  - tkinter 非线程安全 → 后台线程只投递队列，主线程消费
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
from tkinter import ttk, messagebox, filedialog

import gguf_meta as gm

APP_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
           else os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(APP_DIR, "models.json")
LLAMA_ROOT = os.environ.get("LLAMA_ROOT", r"D:\AI\llama.cpp")
EXE = os.path.join(LLAMA_ROOT, "llama-server.exe")
DEFAULT_SCAN = [os.environ.get("LLAMA_MODELS_DIR", r"D:\AI\models")]
CREATE_NO_WINDOW = 0x08000000
LOG_DIR = os.path.join(LLAMA_ROOT, "_stage")


# ---------------- 外部命令 ----------------

def run_capture(args, timeout=15):
    """跑命令并解码 —— 本机控制台是 GBK(936)，不能假定 UTF-8。"""
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


def server_pids():
    out = run_capture(["tasklist", "/FI", "IMAGENAME eq llama-server.exe",
                       "/FO", "CSV", "/NH"])
    pids = []
    for line in out.splitlines():
        parts = [x.strip('"') for x in line.split('","')]
        if len(parts) >= 2 and parts[0].lower() == "llama-server.exe":
            try:
                pids.append(int(parts[1]))
            except ValueError:
                pass
    return pids


def gpu_mem():
    """(已用MiB, 总MiB) 或 None"""
    out = run_capture(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                       "--format=csv,noheader,nounits"], timeout=8)
    for line in reversed(out.strip().splitlines()):
        try:
            u, t = [int(x.strip()) for x in line.split(",")[:2]]
            return u, t
        except Exception:
            continue
    return None


def http_ok(port, timeout=2.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def which_port():
    """找出当前 llama-server 监听端口（从命令行参数里读不到，就用配置兜底）。"""
    out = run_capture(["netstat", "-ano", "-p", "TCP"])
    for line in out.splitlines():
        if "LISTENING" in line and "127.0.0.1:" in line:
            parts = line.split()
            if len(parts) >= 5 and parts[3] == "LISTENING":
                try:
                    pid = int(parts[4])
                except ValueError:
                    continue
                if pid in server_pids():
                    try:
                        return int(parts[1].rsplit(":", 1)[1])
                    except Exception:
                        pass
    return None


# ---------------- 配置 ----------------

def default_config():
    return {"llama_root": LLAMA_ROOT, "scan_dirs": list(DEFAULT_SCAN),
            "models": [], "last_selected": ""}


def load_config():
    cfg = default_config()
    if os.path.exists(CONFIG):
        try:
            with open(CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg.update(data or {})
        except Exception:
            pass
    cfg.setdefault("models", [])
    cfg.setdefault("scan_dirs", list(DEFAULT_SCAN))
    return cfg


def save_config(cfg):
    tmp = CONFIG + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CONFIG)


# ---------------- 模型扫描与识别 ----------------

NON_TEXT_ARCH = {"flux", "sd", "sdxl", "stable-diffusion", "t5", "clip", "vae",
                 "qwen_image", "qwen-image", "wan", "hunyuan", "ltxv", "aura"}


def analyze(path):
    """读 GGUF 头部，返回 (info, ok, 说明)"""
    try:
        meta = gm.read_gguf_meta(path)
        info = gm.summarize(meta)
    except Exception as e:
        return None, False, f"读取失败：{e}"

    arch = (info.get("arch") or "").lower()
    # 生图/编码器类模型：有 arch 但没有上下文长度，llama.cpp 不能当对话模型跑
    if arch in NON_TEXT_ARCH or (info.get("ctx_train") or 0) == 0:
        return info, False, f"非文本模型（arch={info['arch']}），llama.cpp 不支持"
    return info, True, ""


def scan(cfg):
    """扫描目录，返回合并后的模型列表（保留已有设置）。"""
    by_path = {m["path"].lower(): m for m in cfg.get("models", [])}
    found = []

    for d in cfg.get("scan_dirs", []):
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".gguf"):
                continue
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            info, ok, why = analyze(p)
            entry = by_path.get(p.lower())
            if entry is None:
                entry = {
                    "path": p,
                    "port": 8080,
                    "ctx": 0,          # 0 = 用自动推荐
                    "ncmoe": -1,       # -1 = 用自动推荐
                    "ngl": -1,
                    "extra": "",
                }
            entry["name"] = (info or {}).get("name") or os.path.splitext(fn)[0]
            entry["file_gb"] = (info or {}).get("_file_gb") or round(
                os.path.getsize(p) / 1024**3, 2)
            entry["arch"] = (info or {}).get("arch", "?")
            entry["layers"] = (info or {}).get("layers", 0)
            entry["experts"] = (info or {}).get("experts", 0)
            entry["is_moe"] = bool((info or {}).get("is_moe"))
            entry["ctx_train"] = (info or {}).get("ctx_train", 0)
            entry["supported"] = ok
            entry["why"] = why
            found.append(entry)

    found.sort(key=lambda m: m.get("file_gb", 0), reverse=True)
    return found


# ---------------- GUI ----------------

class Manager:
    def __init__(self, root):
        self.root = root
        self.cfg = load_config()
        self.q = queue.Queue()
        self.busy = False
        self.models = []
        self.current = None
        self.srv_port = None

        root.title("本地大模型管理器")
        root.geometry("880x620")
        root.minsize(820, 560)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_ui()
        self.reload_models()
        self.tick()

    # ---------- 界面 ----------
    def _build_ui(self):
        # 顶部
        head = tk.Frame(self.root, bg="#111827", height=56)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text="  本地大模型管理器", bg="#111827", fg="white",
                 font=("Microsoft YaHei UI", 14, "bold")).pack(side="left")
        self.gpu_lbl = tk.Label(head, text="", bg="#111827", fg="#9ca3af",
                                font=("Consolas", 9))
        self.gpu_lbl.pack(side="right", padx=16)

        # 主体：左列表 + 右详情
        body = tk.Frame(self.root, padx=10, pady=8)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="模型列表（点击选择）", anchor="w",
                 font=("Microsoft YaHei UI", 9), fg="#6b7280").pack(fill="x")

        cols = ("name", "size", "arch", "status")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=16)
        self.tree.heading("name", text="模型")
        self.tree.heading("size", text="大小")
        self.tree.heading("arch", text="架构")
        self.tree.heading("status", text="状态")
        self.tree.column("name", width=200, anchor="w")
        self.tree.column("size", width=70, anchor="e")
        self.tree.column("arch", width=90, anchor="w")
        self.tree.column("status", width=90, anchor="w")
        self.tree.pack(fill="both", expand=True, pady=(4, 0))
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.tag_configure("running", foreground="#16a34a")
        self.tree.tag_configure("bad", foreground="#9ca3af")

        # 右侧详情
        right = tk.Frame(body, width=340, padx=12)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(right, text="模型详情", anchor="w",
                 font=("Microsoft YaHei UI", 9), fg="#6b7280").pack(fill="x")

        self.info_lbl = tk.Label(right, text="（未选择）", anchor="nw", justify="left",
                                 font=("Consolas", 9), fg="#374151",
                                 wraplength=320, height=7)
        self.info_lbl.pack(fill="x", pady=(4, 8))

        params = tk.Frame(right)
        params.pack(fill="x")
        self.entries = {}
        for i, (key, label, hint) in enumerate([
            ("ctx", "上下文", "0=自动"),
            ("ncmoe", "ncmoe", "-1=自动"),
            ("ngl", "ngl", "-1=自动"),
            ("port", "端口", ""),
            ("extra", "额外参数", ""),
        ]):
            tk.Label(params, text=label, anchor="w", width=10,
                     font=("Microsoft YaHei UI", 9)).grid(row=i, column=0, sticky="w", pady=2)
            e = tk.Entry(params, font=("Consolas", 10), width=18)
            e.grid(row=i, column=1, sticky="we", pady=2)
            tk.Label(params, text=hint, font=("Microsoft YaHei UI", 8),
                     fg="#9ca3af").grid(row=i, column=2, sticky="w", padx=(6, 0))
            self.entries[key] = e
        params.columnconfigure(1, weight=1)

        tk.Button(right, text="自动推荐参数", command=self.auto_params,
                  font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(8, 0))
        tk.Button(right, text="保存该模型设置", command=self.save_current,
                  font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=(4, 0))

        # 底部按钮
        bar = tk.Frame(self.root, pady=8, padx=10)
        bar.pack(fill="x")

        self.btn_start = tk.Button(bar, text="▶  启动所选模型", command=self.start_selected,
                                   font=("Microsoft YaHei UI", 11, "bold"),
                                   bg="#16a34a", fg="white", relief="flat",
                                   width=16, height=2, cursor="hand2")
        self.btn_start.pack(side="left")

        self.btn_stop = tk.Button(bar, text="■  停止服务", command=self.stop_server,
                                  font=("Microsoft YaHei UI", 11, "bold"),
                                  bg="#d1d5db", fg="#6b7280", relief="flat",
                                  width=12, height=2, cursor="arrow", state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        for text, fn in [("重新扫描", self.reload_models),
                         ("添加模型文件", self.add_model),
                         ("打开模型目录", lambda: os.startfile(self.cfg["scan_dirs"][0])
                          if self.cfg["scan_dirs"] else None)]:
            tk.Button(bar, text=text, command=fn, font=("Microsoft YaHei UI", 9),
                      relief="flat", bg="#e5e7eb", width=12).pack(side="left", padx=(8, 0))

        # 状态栏
        self.status = tk.StringVar(value="就绪")
        tk.Label(self.root, textvariable=self.status, anchor="w",
                 font=("Microsoft YaHei UI", 9), fg="#6b7280").pack(
            fill="x", padx=12, pady=(0, 8))

    # ---------- 数据 ----------
    def reload_models(self):
        self.status.set("正在扫描模型目录…")
        self.root.update_idletasks()
        try:
            self.models = scan(self.cfg)
        except Exception as e:
            messagebox.showerror("扫描失败", str(e))
            self.models = []
        self.cfg["models"] = self.models
        for m in self.models:
            m.setdefault("port", 8080)
            m.setdefault("ctx", 0)
            m.setdefault("ncmoe", -1)
            m.setdefault("ngl", -1)
            m.setdefault("extra", "")
        save_config(self.cfg)
        self.refresh_tree()
        # 默认选中第一个可用模型，用户进来就能直接点启动
        children = self.tree.get_children()
        if children:
            want = self.cfg.get("last_selected")
            target = children[0]
            if want:
                for i in children:
                    try:
                        if self.models[int(i)]["path"] == want:
                            target = i
                            break
                    except (ValueError, IndexError):
                        pass
            self.tree.selection_set(target)
            self.tree.focus(target)
        n_ok = sum(1 for m in self.models if m.get("supported"))
        self.status.set(f"扫描完成：{len(self.models)} 个文件，其中 {n_ok} 个可用于对话")

    def refresh_tree(self):
        sel = self.current["path"] if self.current else None
        self.tree.delete(*self.tree.get_children())
        running = bool(server_pids())
        for i, m in enumerate(self.models):
            if running:
                status, tag = "● 运行中", "running"
            elif not m.get("supported"):
                status, tag = "不支持", "bad"
            else:
                status, tag = "已停止", ""
            self.tree.insert("", "end", iid=str(i),
                             values=(m.get("name", "?"),
                                     f"{m.get('file_gb', 0):.1f} GB",
                                     m.get("arch", "?"),
                                     status),
                             tags=(tag,) if tag else ())
            if sel and m["path"] == sel:
                self.tree.selection_set(str(i))

    def on_select(self, _evt=None):
        sel = self.tree.selection()
        if not sel:
            return
        self.current = self.models[int(sel[0])]
        m = self.current
        lines = [
            f"文件：{os.path.basename(m['path'])}",
            f"大小：{m.get('file_gb', 0):.2f} GB",
            f"架构：{m.get('arch', '?')}",
            f"层数：{m.get('layers', '?')}    专家：{m.get('experts', 0) or '无'}",
            f"原生上下文：{m.get('ctx_train', 0) or '?'}",
        ]
        if not m.get("supported"):
            lines.append(f"⚠ {m.get('why', '')}")
        self.info_lbl.config(text="\n".join(lines))
        for key in self.entries:
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, str(m.get(key, "")))
        # 记住上次选择
        self.cfg["last_selected"] = m["path"]
        try:
            save_config(self.cfg)
        except Exception:
            pass
        self.paint()

    def auto_params(self):
        if not self.current:
            messagebox.showinfo("提示", "先选一个模型")
            return
        m = self.current
        info, ok, why = analyze(m["path"])
        if not ok:
            messagebox.showwarning("不适用", why or "该文件不是对话模型")
            return
        mem = gpu_mem()
        free_gb = (mem[1] - mem[0]) / 1024 if mem else 11.1
        rec = gm.recommend(info, free_vram_gb=free_gb)
        self.entries["ctx"].delete(0, "end")
        self.entries["ctx"].insert(0, str(rec["ctx"]))
        self.entries["ncmoe"].delete(0, "end")
        self.entries["ncmoe"].insert(0, str(rec["ncmoe"] if rec["ncmoe"] is not None else -1))
        self.entries["ngl"].delete(0, "end")
        self.entries["ngl"].insert(0, str(rec["ngl"]))
        self.save_current(silent=True)
        messagebox.showinfo("自动推荐", rec["reason"] + f"\n\n上下文：{rec['ctx']}")

    def save_current(self, silent=False):
        if not self.current:
            return
        try:
            m = self.current
            m["ctx"] = int(self.entries["ctx"].get() or 0)
            m["ncmoe"] = int(self.entries["ncmoe"].get() or -1)
            m["ngl"] = int(self.entries["ngl"].get() or -1)
            m["port"] = int(self.entries["port"].get() or 8080)
            m["extra"] = self.entries["extra"].get().strip()
            save_config(self.cfg)
            if not silent:
                self.status.set(f"已保存：{m['name']}")
        except ValueError:
            messagebox.showerror("参数错误", "数字字段必须是整数")

    # ---------- 启动 ----------
    def build_args(self, m):
        mem = gpu_mem()
        free_gb = (mem[1] - mem[0]) / 1024 if mem else 11.1
        info, ok, why = analyze(m["path"])
        rec = gm.recommend(info, free_vram_gb=free_gb) if ok else {
            "ctx": 8192, "ncmoe": None, "ngl": 99}

        ctx = m["ctx"] or rec["ctx"]
        ncmoe = m["ncmoe"] if m["ncmoe"] >= 0 else rec["ncmoe"]
        ngl = m["ngl"] if m["ngl"] >= 0 else rec["ngl"]

        args = [EXE, "-m", m["path"],
                "-ngl", str(ngl),
                "-c", str(ctx),
                "-fa", "on",
                "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", "16",
                "--host", "127.0.0.1", "--port", str(m["port"]),
                "--no-webui"]
        if ncmoe is not None:
            args += ["-ncmoe", str(ncmoe)]
        if m.get("extra"):
            args += m["extra"].split()
        return args, ctx, ncmoe, ngl

    def start_selected(self):
        if self.busy:
            return
        if not self.current:
            messagebox.showinfo("提示", "先在列表里选一个模型")
            return
        m = self.current
        if not m.get("supported"):
            messagebox.showwarning("不支持", m.get("why") or "该文件不能作为对话模型")
            return
        if server_pids():
            messagebox.showwarning("已有服务", "已经有一个模型在运行，先停止它。\n\n"
                                             "12GB 显存同时只能跑一个。")
            return
        if not os.path.exists(EXE):
            messagebox.showerror("缺少 llama.cpp", f"未找到：\n{EXE}")
            return

        args, ctx, ncmoe, ngl = self.build_args(m)
        self.set_busy(True, f"正在启动 {m['name']}…（ctx={ctx}, ncmoe={ncmoe}, ngl={ngl}）")

        def work():
            try:
                os.makedirs(LOG_DIR, exist_ok=True)
                fout = open(os.path.join(LOG_DIR, "server.log"), "a",
                            encoding="utf-8", errors="replace")
                ferr = open(os.path.join(LOG_DIR, "server.err.log"), "a",
                            encoding="utf-8", errors="replace")
                subprocess.Popen(args, cwd=LLAMA_ROOT, stdout=fout, stderr=ferr,
                                 creationflags=CREATE_NO_WINDOW)
                for _ in range(180):
                    time.sleep(1)
                    if http_ok(m["port"]):
                        self.q.put(("started", m["name"], m["port"]))
                        return
                    if not server_pids():
                        break
                self.q.put(("start_failed",))
            except Exception as e:
                self.q.put(("start_failed", str(e)))

        threading.Thread(target=work, daemon=True).start()

    # ---------- 停止 ----------
    def stop_server(self, skip_confirm=False):
        pids = server_pids()
        if not pids:
            return
        if not skip_confirm and not messagebox.askyesno(
                "确认停止", f"停止模型服务？\n\n进程 {pids}，将释放约 11.6 GB 显存。"):
            return
        self.set_busy(True, "正在停止…")

        def work():
            for pid in pids:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, creationflags=CREATE_NO_WINDOW)
            for _ in range(30):
                time.sleep(0.5)
                if not server_pids():
                    break
            self.q.put(("stopped",))

        threading.Thread(target=work, daemon=True).start()

    # ---------- 状态机 ----------
    def set_busy(self, on, text=""):
        self.busy = on
        if text:
            self.status.set(text)
        self.paint()

    def paint(self):
        pids = server_pids()
        running = bool(pids)
        can_start = (not running) and (not self.busy) and bool(self.current)
        self.btn_start.config(
            state="normal" if can_start else "disabled",
            bg="#16a34a" if can_start else "#d1d5db",
            fg="white" if can_start else "#6b7280",
            cursor="hand2" if can_start else "arrow")
        can_stop = running and not self.busy
        self.btn_stop.config(
            state="normal" if can_stop else "disabled",
            bg="#dc2626" if running else "#d1d5db",
            fg="white" if running else "#6b7280",
            cursor="hand2" if can_stop else "arrow")

    def tick(self):
        """主线程定时器：消费队列 + 刷新状态。"""
        try:
            while True:
                item = self.q.get_nowait()
                self.handle(item)
        except queue.Empty:
            pass

        mem = gpu_mem()
        self.gpu_lbl.config(
            text=(f"显存 {mem[0]}/{mem[1]} MiB" if mem else "显存 ?"))
        if not self.busy:
            self.paint()
            self.refresh_tree_light()
        self.root.after(1000, self.tick)

    def refresh_tree_light(self):
        """只更新状态列，不重建整棵树（避免闪烁与选中丢失）。"""
        running = bool(server_pids())
        for i in self.tree.get_children():
            try:
                m = self.models[int(i)]
            except (ValueError, IndexError):
                continue
            if running:
                s, tag = "● 运行中", "running"
            elif not m.get("supported"):
                s, tag = "不支持", "bad"
            else:
                s, tag = "已停止", ""
            vals = list(self.tree.item(i, "values"))
            if len(vals) == 4 and vals[3] != s:
                vals[3] = s
                self.tree.item(i, values=vals, tags=(tag,) if tag else ())

    def handle(self, item):
        kind = item[0]
        if kind == "started":
            _, name, port = item
            self.srv_port = port
            self.set_busy(False, f"✅ {name} 已就绪 → http://127.0.0.1:{port}/v1")
        elif kind == "start_failed":
            err = item[1] if len(item) > 1 else ""
            self.set_busy(False, "启动失败")
            tail = ""
            try:
                with open(os.path.join(LOG_DIR, "server.err.log"), "r",
                          encoding="utf-8", errors="replace") as f:
                    tail = "".join(f.readlines()[-10:])
            except Exception:
                pass
            messagebox.showerror("启动失败",
                                 (f"{err}\n\n" if err else "") + tail)
        elif kind == "stopped":
            self.set_busy(False, "已停止，显存已释放")

    # ---------- 其他 ----------
    def add_model(self):
        p = filedialog.askopenfilename(
            title="选择 GGUF 模型",
            filetypes=[("GGUF 模型", "*.gguf"), ("所有文件", "*.*")])
        if not p:
            return
        d = os.path.dirname(p)
        if d not in self.cfg["scan_dirs"]:
            self.cfg["scan_dirs"].append(d)
            save_config(self.cfg)
        self.reload_models()
        for i, m in enumerate(self.models):
            if m["path"].lower() == p.lower():
                self.tree.selection_set(str(i))
                break

    def on_close(self):
        if server_pids():
            ans = messagebox.askyesnocancel(
                "关闭管理器",
                "模型服务仍在运行。\n\n"
                "是 = 停止模型并关闭\n否 = 保持运行，只关界面\n取消 = 不关闭")
            if ans is None:
                return
            if ans:
                self.stop_server(skip_confirm=True)
                def wait_then_die():
                    for _ in range(60):
                        if not server_pids():
                            break
                        time.sleep(0.5)
                    self.root.after(0, self.root.destroy)
                threading.Thread(target=wait_then_die, daemon=True).start()
                return
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    Manager(root)
    root.mainloop()
