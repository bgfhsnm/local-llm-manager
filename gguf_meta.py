# -*- coding: utf-8 -*-
"""GGUF 元数据解析 + 自动启动参数推荐。

不依赖 llama.cpp，直接读文件头。用于：
  1. 识别模型架构（MoE / dense）
  2. 判断层数、专家数
  3. 据此推荐 -ncmoe（MoE）或 -ngl（dense）与上下文长度
"""
import os
import struct

# GGUF 值类型
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)

_FIXED = {
    _U8: ("<B", 1), _I8: ("<b", 1), _U16: ("<H", 2), _I16: ("<h", 2),
    _U32: ("<I", 4), _I32: ("<i", 4), _F32: ("<f", 4), _BOOL: ("<?", 1),
    _U64: ("<Q", 8), _I64: ("<q", 8), _F64: ("<d", 8),
}

MAX_STR = 1 << 20          # 单个字符串上限，防坏文件
MAX_ARRAY = 1 << 22        # 数组元素上限


class _Reader:
    def __init__(self, f, size):
        self.f = f
        self.size = size
        self.pos = 0

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("unexpected EOF")
        self.pos += n
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def string(self):
        n = self.u64()
        if n > MAX_STR:
            raise ValueError(f"string too long: {n}")
        return self.read(n).decode("utf-8", "replace")

    def value(self, t):
        if t in _FIXED:
            fmt, sz = _FIXED[t]
            return struct.unpack(fmt, self.read(sz))[0]
        if t == _STR:
            return self.string()
        if t == _ARR:
            et = self.u32()
            cnt = self.u64()
            if cnt > MAX_ARRAY:
                raise ValueError(f"array too long: {cnt}")
            out = []
            for _ in range(cnt):
                out.append(self.value(et))
            return out
        raise ValueError(f"unknown type {t}")

    def skip(self, t):
        """跳过值而不物化（数组可能巨大）。"""
        if t in _FIXED:
            self.read(_FIXED[t][1])
        elif t == _STR:
            n = self.u64()
            self.read(n)
        elif t == _ARR:
            et = self.u32()
            cnt = self.u64()
            for _ in range(cnt):
                self.skip(et)
        else:
            raise ValueError(f"unknown type {t}")


def read_gguf_meta(path, max_kv=200):
    """读 GGUF 头部，返回元数据 dict（只保留标量/短字符串，数组跳过）。"""
    size = os.path.getsize(path)
    meta = {"_file_size": size, "_file_gb": round(size / 1024**3, 2)}
    with open(path, "rb") as f:
        head = f.read(4)
        if head != b"GGUF":
            raise ValueError("不是 GGUF 文件")
        r = _Reader(f, size)
        r.pos = 4
        version = r.u32()
        tcount = r.u64()
        kcount = r.u64()
        meta["_gguf_version"] = version
        meta["_tensors"] = tcount

        n = 0
        while n < kcount and n < max_kv:
            key = r.string()
            t = r.u32()
            if t in _FIXED or t == _STR:
                try:
                    v = r.value(t)
                    if isinstance(v, str) and len(v) > 400:
                        v = v[:400] + "…"
                    meta[key] = v
                except (EOFError, ValueError):
                    break
            else:
                # 数组（如 tokenizer 词表）跳过，不占用内存
                try:
                    r.skip(t)
                except (EOFError, ValueError):
                    break
            n += 1
    return meta


def summarize(meta):
    """从元数据提取关键架构信息。"""
    arch = meta.get("general.architecture", "?")
    name = meta.get("general.name", "?")
    prefix = f"{arch}."

    def get(k, default=None):
        return meta.get(prefix + k, default)

    info = {
        "arch": arch,
        "name": name,
        "file_gb": meta.get("_file_gb", 0),
        "layers": get("block_count") or 0,
        "kv_heads": get("attention.head_count_kv"),
        "head_dim": get("attention.key_length") or get("attention.value_length"),
        "experts": get("expert_count") or 0,
        "experts_used": get("expert_used_count") or 0,
        "ctx_train": get("context_length") or 0,
        "is_moe": bool((get("expert_count") or 0) > 1),
        "embedding": get("embedding_length"),
    }
    return info


def recommend(info, free_vram_gb=11.1, kv_gb=0.4, overhead_gb=0.8):
    """根据模型结构 + 实时空闲显存，推荐启动参数。

    校准依据（本机实测，Qwen3.6-35B-A3B 20.82GB / 40层 / 256专家）：
      用两组实测反解内存构成：
        ncmoe=30（10层专家上GPU）-> server 占 7.70 GB
        ncmoe=24（16层专家上GPU）-> server 占 10.60 GB
      解得：专家总量 19.33GB（占文件 92.9%），固定开销 2.87GB
            （非专家权重 1.67 + KV 0.4 + 运行时 0.8）
      验证：空闲 11.1GB 时推荐 ncmoe=24，与实测最优档一致（53.5 tok/s）。
    """
    layers = info.get("layers") or 32
    file_gb = info.get("file_gb") or 8.0
    ctx = 32768

    if info.get("is_moe"):
        non_expert = file_gb * 0.08
        expert_total = file_gb * 0.92
        per_layer = expert_total / max(layers, 1)

        budget = free_vram_gb - overhead_gb - kv_gb - non_expert
        gpu_layers = int(budget // per_layer) if per_layer > 0 else 0
        gpu_layers = max(0, min(gpu_layers, layers))
        # 留 1 层安全余量，避免踩到显存边界（实测多放 4 层就会崩）
        gpu_layers = max(0, gpu_layers - 1)
        ncmoe = layers - gpu_layers

        need = non_expert + gpu_layers * per_layer + kv_gb + overhead_gb
        reason = (f"MoE {layers}层/{info.get('experts')}专家：{gpu_layers}层专家上GPU"
                  f"（预计占 {need:.1f}GB / 空闲 {free_vram_gb:.1f}GB），{ncmoe}层留CPU")
        return {"ngl": 99, "ncmoe": ncmoe, "ctx": ctx, "reason": reason,
                "est_vram_gb": round(need, 1)}

    # dense：按层数比例分配，留 8% 余量
    budget = free_vram_gb - overhead_gb - kv_gb
    ngl = int(layers * budget / max(file_gb, 0.1) * 0.92)
    ngl = max(0, min(ngl, layers))
    ctx = 16384 if file_gb > 12 else 32768
    reason = f"dense {layers}层：{ngl}/{layers}层放GPU（模型 {file_gb:.1f}GB）"
    return {"ngl": ngl, "ncmoe": None, "ctx": ctx, "reason": reason}


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        m = read_gguf_meta(p)
        i = summarize(m)
        r = recommend(i)
        print(f"{os.path.basename(p)}")
        print(f"  arch={i['arch']} layers={i['layers']} experts={i['experts']} "
              f"used={i['experts_used']} ctx_train={i['ctx_train']} size={i['file_gb']}GB")
        print(f"  推荐: {r}")
