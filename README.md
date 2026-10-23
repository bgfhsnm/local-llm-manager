# local-llm-manager（本地大模型管理器）

一个 Windows 桌面小工具，用**一个窗口**管理本机所有 GGUF 大模型的启动与停止：

- 自动扫描模型目录，列出全部 `.gguf`
- 读 GGUF 头部**自动识别架构**（MoE / dense / 非文本模型），不靠文件名猜
- 按**实时空闲显存**自动推荐 `-ncmoe` / `-ngl` / 上下文长度
- 每个模型独立保存参数，一键启动 / 一键停止
- 实时显示运行状态、端口、显存占用

为「**12GB 显存只能同时跑一个大模型**」这个现实约束设计：同一时刻只允许一个服务在跑，启动新模型前会提示先停旧的。

## 为什么需要它

跑本地大模型最常见的两个坑，这个工具都替你算掉了：

1. **MoE 模型的 `-ncmoe` 该设多少？** 设小了 OOM，设大了白白浪费速度。
   本工具从 GGUF 元数据反解出专家层占比，再按当前空闲显存算出一个不爆显存的档位。
2. **参数该记在哪？** `llama-server` 的命令行长到没法手敲。本工具按模型路径分别持久化，下次点开就是上次的配置。

参数推荐公式（用本机实测反解校准，非拍脑袋）：

```
MoE 专家权重总量 ≈ 文件大小 × 0.929      # 该模型 92.9% 参数在专家层
固定开销（非专家权重 + KV cache + 运行时） ≈ 2.87 GB
可用显存 = 空闲显存 - 固定开销 - 安全余量
可留在显存里的专家层数 → 反推 -ncmoe
```

在 RTX 5070 Ti Laptop（12GB，可用 11.91GB）、空闲 11.1GB 时，推荐 `ncmoe=24`，与手工压测找到的最优值一致。

## 功能细节

| 模块 | 作用 |
|---|---|
| `gguf_meta.py` | 直接解析 GGUF 头部二进制：架构、层数、专家数、训练上下文、是否为 MoE、是否支持文本 |
| `model_manager.py` | GUI 主体（tkinter）：模型列表、参数编辑、启动/停止、显存与日志显示 |
| `ds_llm_controller.py` | 单模型极简版控制器（一个模型一个按钮），适合只跑一个模型的场景 |

**非文本模型会被自动标记「不支持」**：ComfyUI 里的 `flux` / `qwen-image` / `t5` 是生图组件，`llama.cpp` 跑不了，工具会识别出来并灰掉，不会让你白等半天。

**停止用 `taskkill /F /T` 杀整棵进程树**，显存立刻释放，不会留僵尸进程占着显存。

## 环境要求

- Windows 10/11
- Python 3.10+（源码运行）
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 `llama-server.exe`（建议带 CUDA 的预编译版）
- 若干 `.gguf` 模型文件

## 使用

### 源码运行

```powershell
pip install -r requirements.txt   # 本工具无第三方依赖，仅 tkinter 标准库
python model_manager.py
```

### 环境变量（可选，换机器时用）

| 变量 | 默认值 | 作用 |
|---|---|---|
| `LLAMA_ROOT` | `D:\AI\llama.cpp` | llama.cpp 根目录（`llama-server.exe` 所在处） |
| `LLAMA_MODELS_DIR` | `D:\AI\models` | 默认扫描的模型目录 |
| `LLAMA_MODEL` | `D:\AI\models\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | `ds_llm_controller.py` 用的模型 |

模型参数持久化在程序同目录的 `models.json`（首次运行自动生成，可从 `models.example.json` 复制一份改）。

### 打包成 exe

```powershell
pyinstaller --noconfirm --onefile --windowed --name 本地大模型管理器 model_manager.py gguf_meta.py
```

> **注意**：PyInstaller onefile 打出来的 exe 放在用户目录（如 `C:\Users\<你>\Documents\...`）下运行可能报
> `Could not create temporary directory!`——某些杀软（如火绒）会拦截未知 exe 在用户目录建临时目录。
> 把 exe 放到 `D:\` 或桌面等路径即可正常运行。

## 目录结构

```
local-llm-manager/
├── model_manager.py         # GUI 主体（模型列表 + 参数 + 启停）
├── gguf_meta.py             # GGUF 头部解析
├── ds_llm_controller.py     # 单模型极简控制器
├── models.example.json      # 配置样例
└── README.md
```

## 已知限制

- **仅 Windows**：依赖 `taskkill`、`CREATE_NO_WINDOW` 等 Windows 特性
- 只管理 `llama-server.exe`，不支持 vLLM / Ollama 等其它后端
- 显存读数来自 `nvidia-smi`，多卡机器目前只统计第一张卡

## 许可

MIT，见 [LICENSE](LICENSE)。
