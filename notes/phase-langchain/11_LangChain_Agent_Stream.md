# 第42课：LangChain Agent 流式输出

**项目**: `agent.stream` 与多种 `stream_mode` 观察 Agent 执行过程  
**技术栈**: LangChain v1, langchain.agents, LangGraph, langchain-core  
**示例代码**: `examples/langchain/11_langchain_agent_stream.py`  
**前置课程**: 第40课 LangChain Agent  
**环境与运行**：见 [第32课 §1 环境配置](01_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/langchain/11_langchain_agent_stream.py`（`main()` 默认 `stream_mode="values"`，注释切换其它模式）

---

## 课程概述

第40课用 `agent.invoke` **同步等待** Agent 跑完整个 ReAct 循环，一次性拿到最终 `messages`。本课改用 `agent.stream`：**边执行边推送**中间状态，便于在终端或 UI 中展示「正在调工具」「模型刚返回」等进度，也利于调试多步 Agent。

示例绑定三个模拟工具（天气、时间、位置），用户问「当前的位置、时间、天气是什么？」会触发多次 tool 循环。`invoke_agent_test(stream_mode, print_chunk)` 封装 `stream` 调用；`main()` 默认启用 `stream_mode="values"` 并 `print_chunk=True`，其余五种模式在注释中切换。

**学习目标：**
1. 理解 `invoke` 与 `stream` 的输入/输出差异
2. 掌握 LangGraph 六种 `stream_mode`：`values`、`updates`、`messages`、`tasks`、`debug`、`checkpoints`
3. 会在循环中解析 chunk，按需打印完整状态或增量更新
4. 了解多工具 Agent 在流式下的典型 chunk 序列
5. 能将流式输出接入简单 CLI 或 Web SSE 前端

---

## 1. 为什么需要 stream？

### 1.1 同步 invoke 的局限

| 维度 | `agent.invoke` | `agent.stream` |
|------|----------------|--------------|
| 返回时机 | 图执行**结束后**一次返回 | **每个步骤** yield 一个 chunk |
| 用户体验 | 长时间无输出，像「卡住」 | 可实时展示进度、工具名、中间消息 |
| 调试 | 需事后翻 `messages` | 逐步看到 model / tools 节点输出 |
| 前端集成 | 需自己轮询或等整包 | 天然适合 SSE / WebSocket 推送 |

Agent 调 3 个工具时，`invoke` 可能数秒无反馈；`stream` 可在每次 tool 返回后立即更新界面。

### 1.2 流式下的 Agent 循环

```
HumanMessage
    │
    ▼ stream chunk ①  model 节点完成 → AIMessage(tool_calls=[weather, time, location])
    │
    ▼ stream chunk ②  tools 节点完成 → ToolMessage × 3
    │
    ▼ stream chunk ③  model 节点完成 → AIMessage（最终自然语言总结）
    │
    … 具体 chunk 形状取决于 stream_mode …
```

---

## 2. 基础用法

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain_core.tools import tool

@tool()
def get_current_weather(city: str) -> str:
    '''获取当前天气'''
    return f"当前天气为：{city} 晴朗"

agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    tools=[get_current_weather, get_current_time, get_current_location],
)

messages = [HumanMessage(content="当前的位置、时间、天气是什么？")]

for chunk in agent.stream({"messages": messages}, stream_mode="values"):
    print(chunk)
    print("-" * 100)
```

### 2.1 输入格式

与 `invoke` 相同：`{"messages": [...]}`。LangChain v1 也支持 role dict 列表。

### 2.2 第二个参数 `stream_mode`

指定**每个 chunk 携带什么信息**。LangGraph 支持：

```python
StreamMode = Literal[
    "values", "updates", "checkpoints", "tasks", "debug", "messages", "custom"
]
```

示例中通过函数参数切换：

```python
def invoke_agent_test(stream_mode: str, print_chunk: bool = False):
    ...
    for chunk in agent.stream({"messages": messages}, stream_mode=stream_mode):
        if print_chunk:
            rprint(chunk)
        print("-" * 100)
```

`main()` 默认：

```python
invoke_agent_test("values", print_chunk=True)
# invoke_agent_test("updates", print_chunk=True)
# invoke_agent_test("messages", print_chunk=True)
# ...
```

---

## 3. stream_mode 详解

### 3.1 `values` — 每步完整状态（本课默认）

| 项 | 说明 |
|----|------|
| chunk 形态 | **状态字典**，与 `invoke` 返回值结构类似，含当前全部 channel |
| 典型字段 | `messages`：截至该步的完整消息列表 |
| 适用 | 快速对比每一步 state 全貌；调试「现在 messages 有几条」 |
| 缺点 |  chunk 较大，重复传输已有 messages |

多工具场景下，你会看到 `messages` 逐步变长：Human → AI(tool_calls) → Tool × n → AI(最终回答)。

### 3.2 `updates` — 每步节点增量

| 项 | 说明 |
|----|------|
| chunk 形态 | `{节点名: 该节点输出}` 的字典 |
| 示例键 | `"model"`、`"tools"` 等 LangGraph 节点名 |
| 适用 | **生产推荐**：只推送本步变化，体积小 |
| 特殊键 | 可能含 `__interrupt__`、`__metadata__` |

LangChain 官方 `create_agent` 文档示例即用 `stream_mode="updates"` 打印每步节点更新。

### 3.3 `messages` — LLM token / 消息流

| 项 | 说明 |
|----|------|
| chunk 形态 | `(message, metadata)` 元组或 LangGraph v2 的 `MessagesStreamPart` |
| message | 常为 `AIMessageChunk` 等流式片段 |
| metadata | 含 `langgraph_step`、`langgraph_node` 等 |
| 适用 | **打字机效果**、逐 token 展示最终回答 |

与 `values` 不同：关注**模型生成内容**的细粒度流，而非整图状态。

### 3.4 `tasks` — 任务开始与结束

| 项 | 说明 |
|----|------|
| chunk 形态 | task start / task result 事件 |
| 数据 | `TaskPayload`（id、name、input）或 `TaskResultPayload`（result、error） |
| 适用 | 观察并行子任务、子图（multi-agent 进阶） |

### 3.5 `debug` — 调试详情

| 项 | 说明 |
|----|------|
| chunk 形态 | `DebugStreamPart`，含 `DebugPayload` |
| 适用 | 排查 middleware、节点转换、状态合并问题 |
| 注意 | 输出冗长，仅开发环境使用 |

### 3.6 `checkpoints` — 检查点快照

| 项 | 说明 |
|----|------|
| chunk 形态 | `CheckpointStreamPart`，含 checkpoint payload |
| 适用 | 配合 `checkpointer` 做持久化对话、时间旅行调试 |
| 本课 | 未配置 checkpointer，可感知 API 形态，深度用法见 LangGraph 文档 |

### 3.7 模式选择速查

| 需求 | 推荐 mode |
|------|-----------|
| 终端快速看每步 messages 全长 | `values` |
| 后端 SSE 推送增量 | `updates` |
| 聊天 UI 打字机 | `messages` |
| 查 Agent 图执行细节 | `debug` |
| 多 Agent / 子任务 | `tasks` |
| 持久化会话调试 | `checkpoints` |

可同时传 **mode 列表**（LangGraph 支持多 mode 复用），例如 `stream_mode=["updates", "messages"]`，具体以当前 LangGraph 版本文档为准。

---

## 4. 示例中的三个模拟工具

```python
@tool()
def get_current_weather(city: str) -> str:
    '''获取当前天气'''
    return f"当前天气为：{city} 晴朗"

@tool()
def get_current_time() -> str:
    '''获取当前时间'''
    return f"当前时间：2026-07-06 10:00:00"

@tool()
def get_current_location() -> str:
    '''获取当前位置'''
    return f"当前位置：武汉"
```

| 工具 | 作用 |
|------|------|
| `get_current_weather` | 需 `city` 参数；模型可能先调 location 再调 weather |
| `get_current_time` | 无参；返回固定时间字符串 |
| `get_current_location` | 无参；返回固定位置 |

用户问题一次涉及三个维度，模型往往**并行或顺序**发起多个 `tool_calls`，`stream` 下可清楚看到 tools 节点批量执行后 messages 增长。

---

## 5. 解析 chunk — 示例中的注释代码

源码中预留了按 mode 解析的模板（默认注释）：

```python
# if ("messages" in chunk and chunk["messages"] != None):
#     print(f"message count streamed: {len(chunk['messages'])}")
#     rprint(chunk["messages"][-1].pretty_print())
# elif ("content" in chunk and chunk["content"] != None):
#     print(f"content streamed: {chunk['content']}")
# else:
#     print("No messages streamed")
```

| 场景 | 建议解析方式 |
|------|-------------|
| `values` | `chunk["messages"][-1]` 看最新一条；`pretty_print()` 格式化 |
| `updates` | `for node, update in chunk.items(): ...` |
| `messages` | 解包 `(msg, meta) = chunk` 或读 `chunk["data"]`（视 v1/v2 API） |

生产环境不要无脑 `print` 整个 chunk；按 mode 提取 UI 需要的字段即可。

---

## 6. stream 与 invoke 结果等价性

同一输入下，`stream(..., stream_mode="values")` **最后一个 chunk** 的状态应与 `invoke` 返回值一致（在无 interrupt 的情况下）。验证方式：

```python
final = None
for chunk in agent.stream(inputs, stream_mode="values"):
    final = chunk
sync = agent.invoke(inputs)
assert final["messages"][-1].content == sync["messages"][-1].content
```

若使用 `updates`，需自行合并增量或只比较最终 invoke。

---

## 7. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `agent.stream(inputs, stream_mode=...)` | 流式执行 Agent 图 | `invoke_agent_test` |
| `stream_mode="values"` | 每步完整 state | `main()` 默认 |
| `stream_mode="updates"` | 每步节点增量 | 注释切换 |
| `stream_mode="messages"` | 消息/token 流 | 注释切换 |
| `stream_mode="tasks"` | 任务事件 | 注释切换 |
| `stream_mode="debug"` | 调试 payload | 注释切换 |
| `stream_mode="checkpoints"` | 检查点 | 注释切换 |
| `create_agent(model, tools=[...])` | 创建 Agent | 全部演示 |
| `@tool()` | 定义工具 | 三个 get_current_* |
| `HumanMessage` | 用户输入 | 提问 |

---

## 8. 与前面课程的关系

```
第40课 create_agent + invoke  →  同步跑完 ReAct 循环 ✅ 前置
第41课 response_format        →  结构化输出（可 stream 观察校验重试）
第42课 agent.stream           →  流式观察执行过程 ✅ 你在这里
第33课 LangSmith              →  trace 与 stream 互补（离线全链路 vs 在线进度）
```

典型前端架构：**`stream_mode="updates"`** 推工具进度 + **`stream_mode="messages"`** 推最终回答 token；或仅用 `messages` 同时承载 tool 通知（需自行解析 metadata）。

---

## 9. 常见问题

### Q1: `stream` 和模型 `stream()` 有什么区别？

`agent.stream` 流的是 **LangGraph 图状态**（节点级 / 状态级）；Chat Model 的 `stream` 流的是 **单条 AIMessage 的 token**。Agent 场景二者可嵌套：`stream_mode="messages"` 接近后者。

### Q2: 为什么默认用 `values` 而不是 `updates`？

`values` chunk 结构直观，适合学习时对照 `invoke` 返回的 `messages`。生产 SSE 更常用 `updates` 减小 payload。

### Q3: 终端打印 chunk 太多看不清？

设 `print_chunk=False`，仅解析关键字段；或只启用 `updates` 并过滤 `node == "tools"`。

### Q4: 流式中途报错怎么办？

for 循环外加 try/except；LangGraph 可能在 chunk 流中断前已 yield 部分 state，勿假设最后一条即成功。

### Q5: 能否对 `stream` 也传 `config`？

可以，与 `invoke` 相同，例如 `config={"recursion_limit": 10}`（第41课）。

### Q6: 示例 docstring 路径不对？

请以仓库实际文件为准：`examples/langchain/11_langchain_agent_stream.py`，笔记 `notes/phase-langchain/11_LangChain_Agent_Stream.md`。

### Q7: 无 API Key 时如何学 stream？

需配置 `DEEPSEEK_API_KEY` 才会真实调模型；也可先阅读 chunk 结构文档，或在有 Key 时把 `print_chunk=True` 跑一遍对照笔记。

---

## 10. 动手练习

1. **六种 mode**：依次取消 `main()` 中注释，对比同一问题的 chunk 形状与条数
2. **只关心 tools 步**：`updates` 模式下只打印含 `"tools"` 键的 chunk
3. **消息计数**：在 `values` 模式下打印每 chunk 的 `len(chunk["messages"])`，画出增长曲线
4. **单工具对比**：去掉两个工具，只看 stream chunk 数量如何减少
5. **对接 Rich Live**：用 `rich.live.Live` 在终端刷新「当前最新 AIMessage content」
6. **与 invoke 对照**：同一 `messages` 分别 `invoke` 与 `stream(values)`，比较最终 content 是否一致
7. **加 system_prompt**：要求「依次调用三个工具再总结」，观察 stream 顺序是否变化

---

## 11. 参考

- 示例代码：`examples/langchain/11_langchain_agent_stream.py`
- 公共客户端：`examples/langchain/deepseek_client.py`
- 前置笔记：`notes/phase-langchain/09_LangChain_Agent.md`（`create_agent` 与 invoke）
- 环境配置：`notes/phase-langchain/01_LangChain进阶与DeepSeek接入.md`
- LangChain Agents：[Agents](https://docs.langchain.com/oss/python/langchain/agents)（含 `stream` 示例）
- LangGraph Streaming：[Stream outputs](https://langchain-ai.github.io/langgraph/how-tos/streaming/)

---

*完成本课后，你已能用 `agent.stream` 实时观察 Agent 的 ReAct 循环：从 `values` 掌握完整状态快照，到 `updates`、`messages` 等模式服务不同 UI 与调试需求。它与第40课的 `invoke` 互补——同步拿最终结果，流式展示过程——是构建可感知进度 LLM 应用的必备技能。*
