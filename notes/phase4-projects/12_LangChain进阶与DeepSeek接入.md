# 第32课：LangChain 进阶 — DeepSeek 接入与多种调用方式

**项目**: LangChain Chat Model 进阶用法  
**技术栈**: LangChain, langchain-deepseek, langchain-openai, asyncio  
**示例代码**: `examples/llm-apps/11_langchain_advanced.py`  
**前置课程**: 第19课 LangChain 基础与 RAG、第25课 Agent 接入真实 LLM API

---

## 课程概述

第19课重点在 Chain、Prompt、RAG 等**组件组合**；本课聚焦 **Chat Model 本身**：如何用 LangChain 接入 DeepSeek、三种初始化方式有何区别，以及 `invoke` / `stream` / `batch` / `ainvoke` 等调用模式。

示例采用**交互式菜单**，运行后按需选择 1～9 单项演示，不会一次性跑完全部代码。

**学习目标：**
1. 掌握 DeepSeek 的三种 LangChain 接入方式（原生 / OpenAI 兼容 / 标准 init）
2. 理解 Chat Model 支持的多种输入格式（字符串、字典列表、Message 对象）
3. 会使用流式、批量、异步三种高级调用方式
4. 了解 `AIMessage` 响应结构与 rich 调试输出

---

## 1. 环境配置

### 1.1 依赖

```bash
source activate_env.sh
pip install langchain langchain-deepseek langchain-openai rich
```

### 1.2 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | 可选 | 自定义 API 地址（代理或兼容网关） |

可在项目根目录 `.env` 或 `.vscode/settings.json` 中配置。未设置 `DEEPSEEK_API_KEY` 时脚本会警告，调用仍会失败。

### 1.3 运行

```bash
python examples/llm-apps/11_langchain_advanced.py
```

菜单选项：

```
  1. [API Specifications] ChatDeepSeek 直连
  2. [API Compatibility] ChatOpenAI 兼容模式
  3. [Standard Model init] init_chat_model 标准初始化
  4. [Invoke Input] 字符串输入
  5. [Invoke Input] 字典消息列表（多轮）
  6. [Invoke Input] Message 对象 + rich 打印
  7. [Stream Response] 流式响应
  8. [Batch Response] 批量响应
  9. [Ainvoke Input] 异步调用
  a. 全部运行
  q. 退出
```

---

## 2. 三种模型初始化方式

DeepSeek 提供 OpenAI 兼容 API，LangChain 因此有多种接入路径。本课统一使用模型 `deepseek-v4-pro`。

```
                    ┌─────────────────────┐
                    │   DeepSeek API      │
                    └──────────┬──────────┘
           ┌───────────────────┼───────────────────┐
           ↓                   ↓                   ↓
   ChatDeepSeek          ChatOpenAI          init_chat_model
   (langchain-deepseek)  (langchain-openai)  (langchain.chat_models)
   原生集成               OpenAI 兼容模式      统一入口 "deepseek:模型名"
```

### 2.1 ChatDeepSeek — 原生集成（Demo 1）

```python
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base=os.getenv("DEEPSEEK_BASE_URL"),  # 可选
)
response = llm.invoke("Introduce yourself with single sentence.")
print(response.content)
```

**适用场景**：明确使用 DeepSeek，需要官方集成包提供的特性（如流式、异步的完整支持）。

### 2.2 ChatOpenAI — 兼容模式（Demo 2）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-v4-pro",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),  # 注意参数名是 base_url
)
response = llm.invoke("1 + 1 = ?.")
```

**适用场景**：已有基于 `ChatOpenAI` 的代码，只需改 `base_url` 和 `api_key` 即可切换至 DeepSeek，迁移成本最低。

| 对比项 | ChatDeepSeek | ChatOpenAI |
|--------|--------------|------------|
| 包 | `langchain-deepseek` | `langchain-openai` |
| 自定义 URL 参数 | `api_base` | `base_url` |
| 语义 | DeepSeek 专用 | 任意 OpenAI 兼容端点 |

### 2.3 init_chat_model — 标准初始化（Demo 3）

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="deepseek:deepseek-v4-pro",  # provider:model 格式
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    api_base=os.getenv("DEEPSEEK_BASE_URL"),  # 可选
)
response = llm.invoke("Red + Blue = ?")
```

**适用场景**：LangChain 1.x 推荐的**统一入口**。通过 `provider:model` 字符串切换厂商，便于配置驱动和多模型切换，后续 Chain / Agent 代码无需改动 import。

---

## 3. Invoke 输入格式

Chat Model 的 `invoke()` 接受多种输入，LangChain 会在内部统一转换为消息列表。

### 3.1 字符串 — 最简单（Demo 4）

```python
response = llm.invoke("What is the capital of France?")
print(response.content)
```

等价于单条 `HumanMessage`，适合单轮问答。

### 3.2 字典消息列表 — OpenAI 风格（Demo 5）

```python
messages = [
    {"role": "system", "content": "You are a stupid calculator..."},
    {"role": "user", "content": "1 + 1 = ?"},
    {"role": "assistant", "content": "3"},
    {"role": "user", "content": "What i have asked you?"},
]
response = llm.invoke(messages)
```

与 OpenAI SDK 的 `messages` 格式一致，便于从原生 API 代码迁移。示例故意让模型「算错」，再追问「我问了什么」，用于观察**多轮上下文**是否被正确理解。

### 3.3 Message 对象 — LangChain 原生（Demo 6）

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="You are a stupid calculator..."),
    HumanMessage(content="1 + 1 = ?"),
    AIMessage(content="3"),
    HumanMessage(content="What i have asked you?"),
]
response = llm.invoke(messages)
```

| 类型 | 对应 role | 用途 |
|------|-----------|------|
| `SystemMessage` | system | 系统指令 |
| `HumanMessage` | user | 用户输入 |
| `AIMessage` | assistant | 模型历史回复 |

Message 对象携带更多元数据（如 `tool_calls`、`response_metadata`），Chain / Agent 内部均使用此格式。Demo 6 还用 `rich.print` 打印完整 `AIMessage`，便于调试：

```python
from rich import print as rprint
rprint(response)  # 可看到 content、usage 等字段
```

---

## 4. 高级调用方式

### 4.1 流式响应 — stream（Demo 7）

```python
for chunk in llm.stream("Introduce yourself with single sentence."):
    print(chunk.content, end="", flush=True)  # chunk 可能是 AIMessageChunk
```

**特点**：
- 逐 token / 逐块返回，降低首字延迟，适合聊天 UI
- 返回 `AIMessageChunk`，可拼接为完整回复
- Gradio 聊天机器人（第27课）底层常用此模式

### 4.2 批量响应 — batch（Demo 8）

```python
responses = llm.batch([
    "Introduce yourself with single sentence.",
    "What is the capital of France?",
])
for response in responses:
    print(response.content)
```

**特点**：
- 一次传入多个独立 prompt，LangChain 可并发请求
- 各 prompt **互不共享上下文**，适合离线批处理、评测集跑分
- 与 `invoke` 多轮对话不同：batch 是 N 个独立任务

### 4.3 异步调用 — ainvoke（Demo 9）

```python
import asyncio

async def main():
    llm = ChatDeepSeek(...)
    task = asyncio.create_task(llm.ainvoke("Introduce yourself..."))
    # 等待期间可执行其他协程
    await asyncio.sleep(1)
    response = await task
    print(response.content)

asyncio.run(main())
```

**特点**：
- 不阻塞事件循环，适合 FastAPI、aiohttp 等异步 Web 服务
- 可与多个 `ainvoke` / `astream` 配合 `asyncio.gather` 并发
- Demo 9 在等待期间打印倒计时，演示「异步等待时可做别的事」

---

## 5. 调用方式选型

| 方法 | 同步/异步 | 典型场景 |
|------|-----------|----------|
| `invoke` | 同步 | 脚本、Jupyter、简单 Chain |
| `stream` | 同步迭代 | 聊天界面、长文生成实时展示 |
| `batch` | 同步（内部可并发） | 批量翻译、评测、数据标注 |
| `ainvoke` | 异步 | Web 服务、高并发 Agent |
| `astream` | 异步迭代 | 异步 Web + 流式 UI |

```
用户请求
   ↓
需要实时逐字输出？ ──是──→ stream / astream
   ↓ 否
多个独立问题？     ──是──→ batch
   ↓ 否
在 async 框架中？  ──是──→ ainvoke
   ↓ 否
invoke
```

---

## 6. 代码结构速览

示例将配置与演示分离，便于复用：

```python
# 工厂函数 — 统一读取环境变量
_chat_deepseek()   # → ChatDeepSeek
_chat_openai()     # → ChatOpenAI
_init_chat_model() # → init_chat_model

# 演示函数 — 每个对应菜单一项
demo_api_specifications()
demo_invoke_string()
demo_stream_response()
...

# 菜单驱动 — main() 循环 input，支持单选 / 全跑 / 退出
```

异步 demo 通过 `inspect.iscoroutinefunction` 判断，自动 `asyncio.run()`，同步 demo 直接调用。

---

## 7. 与前面课程的关系

```
第19课 LangChain 基础     →  Chain / Prompt / RAG 组件
第25课 接入真实 LLM API    →  OpenAI SDK、多平台 base_url
第32课 LangChain 进阶     →  Chat Model 初始化与 invoke/stream/batch/ainvoke ✅ 你在这里
第27课 Gradio 部署         →  stream 的实际 UI 应用场景
第28课 Function Calling    →  AIMessage.tool_calls 字段
```

本课打牢 **Model 层** 后，再学 Agent、RAG Chain 时，只需替换 `llm` 实例，调用方式保持一致。

---

## 8. 常见问题

### Q1: ChatDeepSeek 和 ChatOpenAI 选哪个？

- 新项目、要统一多厂商 → `init_chat_model("deepseek:...")`
- 已有 OpenAI 代码迁移 → `ChatOpenAI` + `base_url`
- 需要 DeepSeek 专用能力 → `ChatDeepSeek`

### Q2: 字典 messages 和 Message 对象可以混用吗？

不建议在同一列表中混用。选定一种风格，全项目保持一致；LangChain Chain 内部最终会转为 Message 对象。

### Q3: batch 会保证顺序吗？

`batch` 返回列表顺序与输入顺序一致，即使内部并发执行。

### Q4: stream 的 chunk 为什么有时 content 为空？

部分 chunk 仅含元数据或工具调用片段，拼接时需过滤 `chunk.content` 为空的块。

---

## 9. 动手练习

1. **对比三种初始化**：分别跑 Demo 1～3，观察响应差异（应一致，区别在接入层）
2. **多轮记忆**：修改 Demo 5 的系统提示，测试模型是否遵守「错误计算器」人设
3. **流式 UI**：用 Demo 7 的 `stream`，写 10 行代码实现「打字机效果」终端输出
4. **并发计时**：写两个 `ainvoke`，用 `asyncio.gather` 与顺序 `invoke` 对比耗时
5. **切换模型**：把 `MODEL` 改为 `deepseek-chat`，确认 `init_chat_model` 只需改字符串

---

## 10. 参考

- 示例代码：`examples/llm-apps/11_langchain_advanced.py`
- LangChain Chat Models 文档：[init_chat_model](https://python.langchain.com/docs/how_to/chat_models_universal_init/)
- 前置笔记：`notes/phase4-projects/01_LangChain基础与RAG.md`
- API 接入笔记：`notes/phase4-projects/04_AI_Agent接入真实LLM_API.md`

---

*完成本课后，你已掌握 LangChain Chat Model 的完整调用面：初始化、输入格式、流式 / 批量 / 异步。这是构建 Agent、RAG 与生产级服务的 Model 层基础。*
