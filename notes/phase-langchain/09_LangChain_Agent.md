# 第40课：LangChain Agent

**项目**: `create_agent` 一键构建工具调用循环  
**技术栈**: LangChain v1, langchain.agents, langchain-core, langchain-community, langchain-deepseek, DuckDuckGo  
**示例代码**: `examples/langchain/09_langchain_agnet.py`  
**前置课程**: 第36课 LangChain Tools、第38课 Tool Choices、第39课 Pydantic 结构化输出  
**环境与运行**：见 [第32课 §1 环境配置](01_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/langchain/09_langchain_agnet.py`（`main()` 中注释切换三个演示）

---

## 课程概述

第36课已能手写「`bind_tools` → 解析 `tool_calls` → 执行工具 → 回填 `ToolMessage` → 再 invoke」的最小闭环。LangChain v1 将这一 **ReAct 式循环** 封装为 `create_agent`：传入 `model`、可选 `tools` 与 `system_prompt`，即可得到基于 LangGraph 的 Agent 图，自动在「模型推理」与「工具执行」之间循环，直到模型不再发起 `tool_calls`。

示例分三段：`create_agent_test` 对比三种 `model` 传参并可视化 Agent 图；`invoke_agent_test` 无工具纯对话；`static_tools_agent_test` 先用自定义 `say_yes` / `say_no` 演示工具选择，再用 `DuckDuckGoSearchRun` + `system_prompt` 查询实时天气。

**学习目标：**
1. 理解 `create_agent` 与第36课手写工具循环的关系
2. 掌握三种 `model` 传参方式（模型字符串、`init_chat_model`、`chat_deepseek()`）
3. 会用 `agent.invoke({"messages": [...]})` 驱动 Agent，并读取返回的 `messages` 列表
4. 会用 `@tool(parse_docstring=True)` 为 Agent 注册自定义工具
5. 了解 `system_prompt` 引导工具使用策略，以及社区搜索工具 `DuckDuckGoSearchRun` 的接入方式

---

## 1. 为什么需要 create_agent？

### 1.1 手写循环 vs 一键 Agent

| 维度 | 第36课手写 `bind_tools` 循环 | 本课 `create_agent` |
|------|-------------------------------|---------------------|
| 图结构 | 自己写 while / 多轮 invoke | LangGraph 内置 model ↔ tools 节点 |
| 工具执行 | 手动 `tool.invoke` + `ToolMessage` | 自动执行并追加消息 |
| 停止条件 | 自己判断 `tool_calls` 是否为空 | 内置：无 `tool_calls` 即结束 |
| 可视化 | 无 | `agent.get_graph().draw_mermaid_png()` |
| 扩展 | 灵活但样板多 | 通过 `middleware` 等机制扩展（进阶） |

第24–25课 llm-apps 里手写过 ReAct Agent；第36课在 LangChain 层复现了同样闭环。本课是 LangChain v1 的**标准入口**：`langgraph.prebuilt.create_react_agent` 已迁移为 `langchain.agents.create_agent`。

### 1.2 Agent 核心循环

```
HumanMessage（用户问题）
        │
        ▼
┌───────────────────────────────────────┐
│  model 节点：LLM 推理                  │
│  → AIMessage（可能含 tool_calls）      │
└───────────────┬───────────────────────┘
                │
        ┌───────┴───────┐
        │ 有 tool_calls │ 无 tool_calls
        ▼               ▼
┌───────────────┐   返回完整 messages
│ tools 节点     │   （最后一项多为最终回答）
│ 执行工具       │
│ → ToolMessage  │
└───────┬───────┘
        │
        └──▶ 回到 model 节点（循环）
```

与第36课 §1.2 工具调用闭环一致，只是 LangChain 用 LangGraph 将节点与边固化成可编译、可观测的图。

### 1.3 与 llm-apps Agent 课的区别

| 课程 | 层次 | 重点 |
|------|------|------|
| `03_ai_agent.py` | 纯 Python ReAct | Thought / Action / Observation 字符串解析 |
| `05_langchain_tools.py` | LangChain 工具层 | `@tool` + `bind_tools` + 手动循环 |
| 本课 | LangChain Agent 层 | `create_agent` 自动循环 + 图可视化 |

---

## 2. 创建 Agent — 三种 model 传参

`create_agent_test()` 依次演示三种等价（或近似等价）的模型绑定方式：

### 2.1 模型字符串（推荐入门）

```python
from langchain.agents import create_agent

agent = create_agent(
    model="deepseek:deepseek-v4-pro"
)
```

| 要点 | 说明 |
|------|------|
| 格式 | `"provider:model_name"`，由 `init_chat_model` 在内部解析 |
| DeepSeek | 本仓库示例用 `deepseek:deepseek-v4-pro`，需配置 `DEEPSEEK_API_KEY` |
| 依赖 | 需安装对应 provider 包（如 `langchain-deepseek`） |

### 2.2 `init_chat_model` 显式初始化

```python
from langchain.chat_models import init_chat_model

agent = create_agent(
    model=init_chat_model(model="deepseek:deepseek-v4-pro")
)
```

与字符串形式类似，但**先构造** `BaseChatModel` 实例再传入。适合需要提前设置 `temperature`、`max_tokens`、`api_base` 等参数的场景。

### 2.3 项目工厂函数 `chat_deepseek()`

```python
from deepseek_client import chat_deepseek

agent = create_agent(
    model=chat_deepseek()
)
```

`deepseek_client.py` 统一读取 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`，默认 `MODEL = "deepseek-v4-pro"`。与第32课、第36–39课保持一致，**生产示例推荐**此方式以便集中改配置。

### 2.4 返回类型

三种方式创建的 `agent` 类型均为 LangGraph 编译后的 Runnable（示例中 `print(type(agent))` 可验证）。接口统一：`invoke` / `stream` / `get_graph` 等。

---

## 3. Agent 图可视化

```python
from IPython.display import Image, display

display(Image(agent.get_graph().draw_mermaid_png()))
```

| API | 作用 |
|-----|------|
| `agent.get_graph()` | 获取底层 StateGraph |
| `.draw_mermaid_png()` | 导出 Mermaid 渲染的 PNG 字节 |
| `IPython.display.Image` | 在 **Jupyter / IPython** 中内联显示 |

**无工具**时图通常只有 `model` 单节点；**绑定 tools** 后会出现 `model` ↔ `tools` 双向边，直观看到 ReAct 循环结构。

> **终端运行**：直接 `python examples/langchain/09_langchain_agnet.py` 时，`display()` 在非 IPython 环境可能无效或报错。可在 Notebook 中运行 `create_agent_test()`，或将 PNG 写入文件：
> ```python
> png = agent.get_graph().draw_mermaid_png()
> with open("agent_graph.png", "wb") as f:
>     f.write(png)
> ```

---

## 4. 基础调用 — 无工具 Agent

`invoke_agent_test()`：

```python
from langchain.messages import HumanMessage

agent = create_agent(model="deepseek:deepseek-v4-pro")

messages = [HumanMessage(content="你好")]
response = agent.invoke({"messages": messages})

rprint(response)
```

### 4.1 输入格式

| 键 | 类型 | 说明 |
|----|------|------|
| `messages` | `list[BaseMessage]` 或兼容 dict | 对话历史；本课用 `HumanMessage` |

LangChain v1 也支持 `[{"role": "user", "content": "..."}]` 形式；与第34课 Message 类型互通。

### 4.2 输出格式

`invoke` 返回 **状态字典**，核心字段为 `messages`：**完整对话轨迹**（含本轮新增的 `AIMessage`，若曾调工具则还含 `ToolMessage`）。

```python
response.get("messages")[-1].content   # 通常最后一条为最终回答
```

注意：`invoke` **不会**原地修改你传入的 `messages` 列表（示例中 `rprint(messages)` 仍为原始单条 HumanMessage）；完整历史在 `response["messages"]` 中。

### 4.3 无 tools 时的行为

未传 `tools` 或 `tools=[]` 时，Agent 退化为**单轮模型调用**（无工具循环），等价于带 system 封装的 Chat Model，但仍走同一套 Graph 接口。

---

## 5. 自定义静态工具 — `say_yes` / `say_no`

`static_tools_agent_test()` 第一段：

### 5.1 工具定义

```python
from langchain_core.tools import tool

@tool(parse_docstring=True)
def say_yes() -> str:
    '''回答yes'''
    print("tool invoked yes.")
    return "yes"

@tool(parse_docstring=True)
def say_no() -> str:
    '''回答no'''
    print("tool invoked no.")
    return "no"
```

| 要点 | 说明 |
|------|------|
| `@tool(parse_docstring=True)` | 与第36课相同，docstring 参与 Schema |
| 无参工具 | 模型根据问题语义选择调用哪一个 |
| 副作用 | `print` 便于在终端观察「工具是否真的被调用」 |

### 5.2 绑定并提问

```python
agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    tools=[say_yes, say_no],
)

messages = [HumanMessage(content="北京是中国的首都吗？")]
response = agent.invoke({"messages": messages})

rprint(response.get("messages")[-1].content)
```

对是非题，模型应推理后调用 `say_yes`（或 `say_no`），工具返回 `"yes"` / `"no"` 后，model 节点再生成面向用户的自然语言总结。终端可看到 `tool invoked yes.` 等打印，验证工具链生效。

**与第38课关系**：此处未设 `tool_choice`，默认为 `"auto"`，由模型自行决定是否调工具、调哪一个。

---

## 6. 搜索工具与 system_prompt — 实时信息

`static_tools_agent_test()` 第二段：

```python
from langchain_community.tools import DuckDuckGoSearchRun

agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    tools=[DuckDuckGoSearchRun()],
    system_prompt="你是助手。遇到天气、新闻等实时信息时，请使用搜索工具查询后再回答。",
)

messages = [HumanMessage(content="今天武汉的天气怎么样？")]
response = agent.invoke({"messages": messages})
```

### 6.1 `system_prompt`

| 项 | 说明 |
|----|------|
| 作用 | 写入对话开头的 System 指令，引导模型**何时**该搜索 |
| 类型 | `str` 或 `SystemMessage` |
| 本例 | 强调天气、新闻类问题必须先查再答 |

无 `system_prompt` 时，模型可能对「今天武汉天气」直接幻觉作答；加上后更稳定地触发 `DuckDuckGoSearchRun`。

### 6.2 `DuckDuckGoSearchRun`

| 项 | 说明 |
|----|------|
| 来源 | `langchain_community.tools` |
| 依赖 | 通常需 `duckduckgo-search` 包；需网络访问 |
| 行为 | 将查询发往 DuckDuckGo，返回摘要文本供模型阅读 |

Agent 循环示意：

```
HumanMessage("今天武汉的天气怎么样？")
    → AIMessage(tool_calls=[search...])
    → ToolMessage(搜索结果摘要)
    → AIMessage("武汉今天……")   ← 最终 content
```

### 6.3 读取最终回答

```python
rprint(response.get("messages")[-1].content)
```

生产环境建议遍历 `response["messages"]`，记录每步 `tool_calls` 与 `ToolMessage`，便于审计与 LangSmith 对照（第33课）。

---

## 7. 示例中的三个演示函数

| 函数 | 作用 | `main()` 默认 |
|------|------|---------------|
| `create_agent_test()` | 三种 model 传参 + 打印类型 + 图可视化 | 注释 |
| `invoke_agent_test()` | 无工具，`HumanMessage("你好")` | 注释 |
| `static_tools_agent_test()` | 自定义工具 + DuckDuckGo + system_prompt | **启用** |

取消注释即可切换实验；建议按上表顺序学习。

---

## 8. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `create_agent` | 创建 LangGraph Agent | 全部演示 |
| `model` | 字符串 / `BaseChatModel` | §2 三种传参 |
| `tools` | 工具列表，可选 | `static_tools_agent_test` |
| `system_prompt` | 系统指令 | DuckDuckGo 段 |
| `agent.invoke({"messages": ...})` | 同步运行一轮完整 Agent | 各 invoke 示例 |
| `response["messages"]` | 完整消息轨迹 | 读取最终 `content` |
| `agent.get_graph()` | 获取图结构 | `create_agent_test` |
| `@tool(parse_docstring=True)` | 定义工具 | `say_yes` / `say_no` |
| `DuckDuckGoSearchRun` | 社区搜索工具 | 天气查询段 |
| `HumanMessage` | 用户消息 | 全部 invoke |
| `chat_deepseek()` | DeepSeek 工厂 | `deepseek_client.py` |
| `init_chat_model` | 统一模型初始化 | `create_agent_test` |

---

## 9. 与前面课程的关系

```
第24课 ReAct Agent（llm-apps）     →  字符串 Thought/Action 循环
第28课 Function Calling           →  tool_calls 概念
第36课 @tool + bind_tools         →  手写工具闭环 ✅ 本课自动化
第37课 args_schema                →  工具入参 Schema（Agent 内工具复用）
第38课 tool_choice                →  控制是否/如何调工具（create_agent 进阶可配）
第39课 with_structured_output     →  结构化「输出」；Agent 负责「行动」
第40课 create_agent               →  标准 Agent 入口 ✅ 你在这里
```

典型组合：**Agent 调搜索 / API 工具拿 raw 数据** → **另一步 `with_structured_output` 抽成业务对象** → 写库或展示。

---

## 10. 常见问题

### Q1: `create_agent` 和 `create_react_agent` 有什么区别？

LangChain v1 将 `langgraph.prebuilt.create_react_agent` **迁移**为 `langchain.agents.create_agent`，参数名有调整（如 `prompt` → `system_prompt`）。新项目应使用 `create_agent`；旧代码见 [LangChain v1 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)。

### Q2: 必须传 `tools` 吗？

不必。无 `tools` 时 Agent 仅调用模型，适合纯对话；有实时性或计算需求时再挂载工具列表。

### Q3: 为什么 `invoke` 后我传入的 `messages` 没变？

`invoke` 返回的新状态在 **返回值**里，不会原地修改传入列表。应使用 `response["messages"]` 作为下一轮输入（多轮对话时）。

### Q4: `display(Image(...))` 在终端报错怎么办？

该 API 面向 Jupyter。终端请保存 PNG（见 §3），或只在 Notebook 中运行 `create_agent_test()`。

### Q5: DuckDuckGo 搜索失败？

检查：`pip install duckduckgo-search`、网络代理、以及 DuckDuckGo 限流。可换自建 `@tool` 调用其它搜索 API。

### Q6: 模型不调工具，直接瞎答天气？

加强 `system_prompt`；或查阅 `create_agent` 是否支持 `tool_choice="required"`（视 LangChain 版本与 middleware 而定）；或换更强模型。

### Q7: 如何与 LangSmith 联调？

第33课配置 `LANGCHAIN_TRACING_V2=true` 后，Agent 每次 `invoke` 会在 LangSmith 中留下 Graph  trace，可看到 model / tools 节点交替。

### Q8: 文件名为什么是 `agnet`？

示例文件名为历史笔误（`agnet` → `agent`），笔记与 `main()`  docstring 中路径以仓库实际文件名为准。

---

## 11. 动手练习

1. **三段演示**：依次取消 `main()` 中三个函数注释，观察输出差异
2. **图对比**：分别创建「无 tools」与「有 say_yes/say_no」的 Agent，保存两张 PNG，对比节点与边
3. **换 factory**：将 `static_tools_agent_test` 的 model 改为 `chat_deepseek()`，确认行为一致
4. **加第三个工具**：新增 `@tool def say_maybe(): ...`，问「明天会下雨吗」，看模型是否仍只选 yes/no
5. **多轮对话**：第一次问天气，第二次问「那上海呢？」，将 `response["messages"]` 追加新的 `HumanMessage` 再 `invoke`
6. **对比第36课**：同一问题用手写 `bind_tools` 循环与本课 `create_agent` 各实现一遍，比较代码行数与 message 轨迹
7. **结构化收尾**：搜索得到文本后，另起一条 Chain 用第39课 `with_structured_output` 解析为 `{city, temp, condition}` 对象

---

## 12. 参考

- 示例代码：`examples/langchain/09_langchain_agnet.py`
- 公共客户端：`examples/langchain/deepseek_client.py`
- 前置笔记：`notes/phase-langchain/05_LangChain_Tools工具.md`（工具与手动循环）
- 环境配置：`notes/phase-langchain/01_LangChain进阶与DeepSeek接入.md`
- ReAct 概念：`notes/phase4-projects/03_AI_Agent开发实战.md`
- LangChain Agents：[Agents](https://docs.langchain.com/oss/python/langchain/agents)
- `create_agent` API：[Reference](https://reference.langchain.com/python/langchain/agents/create_agent)
- v1 迁移：[LangChain v1 migration](https://docs.langchain.com/oss/python/migrate/langchain-v1)

---

*完成本课后，你已能用 `create_agent` 将模型与工具组装成可运行的 Agent 图：从三种 model 绑定方式、invoke 状态格式，到自定义工具与搜索工具 + system_prompt 的实战组合。它把第36课的手写循环升级为 LangChain v1 的标准范式；与第39课结构化输出、第33课 LangSmith 联用，即可搭建可观测、可扩展的 LLM 应用。*
