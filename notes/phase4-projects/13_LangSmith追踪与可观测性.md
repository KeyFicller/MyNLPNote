# 第33课：LangSmith 追踪与可观测性

**项目**: LangChain 应用调试与监控  
**技术栈**: LangChain, langchain-deepseek, LangSmith  
**示例代码**: `examples/llm-apps/12_langsmith.py`  
**前置课程**: 第32课 LangChain 进阶 — DeepSeek 接入与多种调用方式  
**环境与运行**：DeepSeek 与虚拟环境见 [第32课 §1 环境配置](12_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/llm-apps/12_langsmith.py`

---

## 课程概述

第32课掌握了 Chat Model 的初始化与调用；本课聚焦 **可观测性（Observability）**：当 Chain / Agent 变复杂后，如何记录每一次 LLM 调用的输入、输出、耗时与错误，并在 LangSmith 控制台中可视化追踪。

示例在 DeepSeek 直连的基础上，通过**环境变量开启追踪**，并在 `invoke` 时传入 `run_name`、`tags` 等元数据，便于在 LangSmith 中筛选与对比。

**学习目标：**
1. 理解 LangSmith 在 LangChain 生态中的定位
2. 掌握 LangSmith 相关环境变量的含义与配置方式
3. 会在 `invoke` 时传入 `config`，为单次 Run 命名和打标签
4. 能在 LangSmith 控制台查看 Trace、分析延迟与 Token 消耗

---

## 1. 为什么需要 LangSmith？

### 1.1 直接调 LLM 的调试困境

```
用户提问 → Chain（Prompt + Retriever + LLM）→ 回答
                ↑
         中间出了错，是哪一步？
         延迟 8 秒，是检索慢还是模型慢？
         回答跑偏，当时喂给模型的 context 是什么？
```

仅靠 `print(response.content)` 无法回答上述问题。生产级 Agent / RAG 需要**全链路追踪**。

### 1.2 LangSmith 是什么

LangSmith 是 LangChain 官方提供的 **LLM 应用开发平台**，核心能力包括：

| 能力 | 说明 |
|------|------|
| **Tracing** | 自动记录每次 Run 的输入、输出、子步骤、耗时 |
| **Debugging** | 在 Web UI 中逐步查看 Chain / Tool 调用链 |
| **Evaluation** | 对数据集批量跑分、对比不同 Prompt / 模型 |
| **Monitoring** | 监控延迟、错误率、Token 用量 |

```
┌──────────────┐     自动上报 Trace      ┌─────────────────┐
│  LangChain   │ ──────────────────────→ │   LangSmith     │
│  本地脚本    │   (环境变量开启)          │   Web 控制台    │
└──────────────┘                         └─────────────────┘
       ↑                                          │
  invoke / stream / Agent                         ↓
                                          查看 Run 树、重放、评测
```

---

## 2. LangSmith 额外配置

除第32课中的 DeepSeek 配置外，本课还需以下 LangSmith 环境变量：

| 变量 | 必需 | 说明 |
|------|------|------|
| `LANGSMITH_API_KEY` | ✅（追踪） | LangSmith 控制台生成的 API Key |
| `LANGSMITH_PROJECT` | 推荐 | 项目名，Trace 归档到该项目下 |
| `LANGSMITH_TRACING` | ✅（追踪） | 设为 `true` 开启追踪 |
| `LANGSMITH_ENDPOINT` | 可选 | 自定义 LangSmith 端点（企业部署） |

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_pt_...
export LANGSMITH_PROJECT=allinai-demo
```

> **说明**：旧版文档使用 `LANGCHAIN_TRACING_V2=true`，LangChain 新版本统一为 `LANGSMITH_TRACING`。本仓库示例采用后者。

### 2.1 获取 LangSmith API Key

1. 注册 [LangSmith](https://smith.langchain.com/)
2. 进入 **Settings → API Keys**
3. 创建 Key，写入 `LANGSMITH_API_KEY`

脚本会先打印当前 LangSmith 配置，再执行一次带元数据的 `invoke` 调用。成功后可在 LangSmith 控制台对应 Project 中看到新的 Trace。

---

## 3. 示例代码结构

### 3.1 配置读取 — 工厂函数

示例将环境变量读取封装为独立函数，与第32课风格一致：

```python
def _api_base() -> str | None:
    return os.getenv("DEEPSEEK_BASE_URL")

def _api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")

def _langsmith_config() -> dict:
    return {
        "project": os.getenv("LANGSMITH_PROJECT"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT"),
        "tracing": os.getenv("LANGSMITH_TRACING"),
    }

def _chat_deepseek() -> ChatDeepSeek:
    kwargs: dict = {"model": MODEL, "api_key": _api_key()}
    if base := _api_base():
        kwargs["api_base"] = base
    return ChatDeepSeek(**kwargs)
```

**要点**：
- `_langsmith_config()` 仅用于**打印诊断**，不负责写入环境；LangChain 在 import / 运行时自动读取同名环境变量
- 追踪是否生效，取决于运行前是否已 `export LANGSMITH_TRACING=true`

### 3.2 带元数据的 invoke

```python
def demo_api_specifications() -> None:
    llm = _chat_deepseek()
    config = {
        "run_name": "Hello LangSmith",
        "tags": ["langchain", "deepseek"],
    }
    response = llm.invoke(
        "Introduce yourself with single sentence.",
        config=config,
    )
    print(response.content)
```

| config 字段 | 作用 |
|-------------|------|
| `run_name` | 在 LangSmith UI 中显示的运行名称，便于识别 |
| `tags` | 标签列表，可按 `langchain`、`deepseek` 等筛选 Trace |

`config` 遵循 LangChain 的 **RunnableConfig** 约定，同样适用于 `stream`、`batch`、`ainvoke` 及 Chain / Agent 调用。

### 3.3 main 流程

```python
def main() -> None:
    if not _api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    config = _langsmith_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    demo_api_specifications()
```

启动时先检查 DeepSeek Key，再打印 LangSmith 四项配置，便于确认追踪环境是否就绪。

---

## 4. 追踪原理

### 4.1 自动插桩

当 `LANGSMITH_TRACING=true` 且 `LANGSMITH_API_KEY` 有效时，LangChain 会在底层为每次 Runnable 调用创建 **Run** 记录，无需改业务逻辑：

```
llm.invoke("...")
    ↓
LangChain 创建 Run（含 input / output / latency）
    ↓
异步上报至 LangSmith API
    ↓
LangSmith Project 中出现 Trace
```

### 4.2 Run 与 Trace 的关系

```
Trace（一次用户请求）
├── Run: ChatDeepSeek          ← run_name: "Hello LangSmith"
│   ├── input: "Introduce yourself..."
│   └── output: AIMessage(...)
└── metadata: tags, project, timestamps
```

复杂 Chain 会产生**嵌套 Run 树**，例如 RAG：`RetrievalQA` → `Retriever` → `ChatModel`，每层均可展开查看。

### 4.3 config 还能传什么

除 `run_name`、`tags` 外，常用字段：

| 字段 | 说明 |
|------|------|
| `metadata` | 任意键值对，如 `{"user_id": "u123"}` |
| `run_id` | 自定义 Run UUID（高级场景） |
| `callbacks` | 额外回调处理器 |

```python
config = {
    "run_name": "客服-订单查询",
    "tags": ["prod", "customer-service"],
    "metadata": {"session_id": "abc-123"},
}
chain.invoke({"question": "我的订单到哪了？"}, config=config)
```

---

## 5. 在 LangSmith 控制台查看

1. 打开 [smith.langchain.com](https://smith.langchain.com/)
2. 左侧选择 **Projects** → 进入 `LANGSMITH_PROJECT` 对应项目
3. 找到 `run_name` 为 **Hello LangSmith** 的最新 Run
4. 点击查看：
   - **Input / Output** — 完整 prompt 与回复
   - **Latency** — 端到端耗时
   - **Tokens** — 若 API 返回 usage 则可见
   - **Tags** — `langchain`, `deepseek`

```
Projects → allinai-demo → Runs
                              │
                              ├─ Hello LangSmith  [langchain] [deepseek]
                              │     Input:  Introduce yourself...
                              │     Output: I am ...
                              │     Latency: 1.2s
                              └─ ...
```

---

## 6. 与前面课程的关系

```
第32课 LangChain 进阶     →  ChatDeepSeek 初始化与 invoke ✅ 基础
第33课 LangSmith 追踪     →  为每次调用加上可观测性 ✅ 你在这里
第28课 Function Calling   →  Tool 调用链可在 LangSmith 中逐步查看
第26课 RAG + Agent        →  检索 + 生成多步 Run 树，追踪价值最大
第27课 Gradio 部署        →  生产环境可对接 LangSmith 监控延迟与错误
```

**典型演进路径**：先跑通 Model → 加 Tracing 调试 Chain → 上线后用 LangSmith 监控与评测。

---

## 7. 常见问题

### Q1: 运行了脚本但 LangSmith 里没有 Trace？

检查清单：
1. `LANGSMITH_TRACING` 是否为字符串 `true`（不是 `True` 或 `1`）
2. `LANGSMITH_API_KEY` 是否有效、未过期
3. 环境变量是否在**运行 Python 之前**已 export（`.env` 需被加载）
4. 网络是否能访问 LangSmith API（企业环境可能需配置代理）

### Q2: `_langsmith_config()` 打印了值，为什么仍无 Trace？

打印只说明**读取到了**环境变量；若 `tracing` 为 `None` 或非 `true`，LangChain 不会上报。需在 shell 或 `.env` 中显式设置 `LANGSMITH_TRACING=true`。

### Q3: 不想每次手动 export，怎么办？

在项目入口统一加载 `.env`：

```python
from dotenv import load_dotenv
load_dotenv()  # 在 import langchain 相关模块之前调用
```

或使用 IDE 的 `settings.json` / 运行配置注入环境变量。

### Q4: 本地开发会泄露 Prompt 吗？

Trace 会上传到 LangSmith 云端（或自建 Endpoint）。**敏感数据**（用户 PII、密钥）应：
- 使用独立 Project 区分 dev / prod
- 在 `metadata` 中避免写入机密
- 企业场景可部署私有 LangSmith 并设置 `LANGSMITH_ENDPOINT`

### Q5: 与 OpenTelemetry / 自建日志的区别？

| 方案 | 特点 |
|------|------|
| **LangSmith** | 与 LangChain 深度集成，LLM 输入输出、Token、Run 树开箱即用 |
| **OpenTelemetry** | 通用分布式追踪，需额外适配 LLM span |
| **print / logging** | 零依赖，但难以还原嵌套 Chain 结构 |

LangSmith 适合 LangChain 技术栈的快速调试；混合架构可 OTel + LangSmith 并存。

---

## 8. 动手练习

1. **首次 Trace**：配置四项 LangSmith 环境变量，运行脚本，在控制台找到 `Hello LangSmith` Run
2. **改 run_name**：将 `run_name` 改为你的名字，再次运行，确认 UI 中可区分两次 Run
3. **加 metadata**：在 `config` 中增加 `metadata={"lesson": 33}`，在 LangSmith 中查看是否展示
4. **对比无追踪**：临时 `unset LANGSMITH_TRACING`，观察脚本行为（应仍能打印回复，但无 Trace）
5. **延伸**：把第32课某条 Chain 或 RAG 示例加上相同 `config`，观察多步 Run 树结构

---

## 9. 参考

- 示例代码：`examples/llm-apps/12_langsmith.py`
- LangSmith 文档：[Tracing](https://docs.smith.langchain.com/observability/how_to_guides/tracing)
- 前置笔记：`notes/phase4-projects/12_LangChain进阶与DeepSeek接入.md`
- LangChain RunnableConfig：[配置 Runnable](https://python.langchain.com/docs/concepts/runnables/#configurable-fields)

---

*完成本课后，你已能为 LangChain 应用开启全链路追踪：环境变量控制开关，`config` 标注每次 Run。这是从「能跑」到「能查、能优」的关键一步。*
