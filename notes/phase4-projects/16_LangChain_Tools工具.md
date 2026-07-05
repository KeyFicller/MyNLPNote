# 第36课：LangChain Tools 工具

**项目**: `@tool` 装饰器与 `bind_tools` 工具调用  
**技术栈**: LangChain, langchain-core, langchain-deepseek, ChatDeepSeek  
**示例代码**: `examples/llm-apps/15_langchain_tools.py`  
**前置课程**: 第28课 Function Calling 与 Tools、第35课 LangChain Prompt 模板

---

## 课程概述

第28课用 OpenAI 原生 API 手写 `tools` JSON Schema 与 `tool_calls` 解析；第34–35课掌握了 `messages` 列表与 `ChatPromptTemplate`。本课进入 LangChain **工具层**：用 `@tool` 把普通 Python 函数变成模型可识别的 Tool，用 `bind_tools` 绑定到 `ChatDeepSeek`，并手动完成「模型发起调用 → 执行函数 → 把结果塞回 messages → 再次 invoke」的闭环。

示例包含三种定义方式：带 `@tool` 的简单函数、带 `parse_docstring` 的参数化函数、以及无装饰器时用 `convert_to_openai_tool` 转换。

**学习目标：**
1. 理解 `@tool` 如何把函数变成 LangChain `StructuredTool`
2. 掌握 `bind_tools` 与 `AIMessage.tool_calls` 的读取方式
3. 能手写最小工具调用循环（执行工具 + 回填 messages）
4. 会用 `convert_to_openai_tool` 为普通函数生成 OpenAI 兼容 Schema
5. 了解 `parse_docstring=True` 与 docstring 格式的要求

---

## 1. 为什么需要 LangChain Tools？

### 1.1 第28课 vs 本课

| 维度 | 第28课（OpenAI API） | 本课（LangChain） |
|------|----------------------|-------------------|
| Schema 定义 | 手写 JSON `tools` 数组 | `@tool` 或 `convert_to_openai_tool` |
| 绑定模型 | `tools=` 请求参数 | `llm.bind_tools([...])` |
| 调用结果 | 解析 `tool_calls` JSON | `response.tool_calls` + `tool.invoke(...)` |
| 返回值类型 | 自定义 dict | `ToolMessage`（自动并入 messages） |

LangChain 把「Schema 生成、参数校验、结果封装」封装在一套接口里，与 Agent、`create_react_agent`、LCEL Chain 共用，减少样板代码。

### 1.2 工具调用闭环

```
用户 HumanMessage
        │
        ▼
llm.bind_tools([...]).invoke(messages)
        │
        ▼
AIMessage（可能含 tool_calls，content 可能为空）
        │
        ├── 无 tool_calls ──▶ 直接返回 content
        │
        └── 有 tool_calls ──▶ tool.invoke(tool_call)
                                    │
                                    ▼
                            ToolMessage（执行结果）
                                    │
                                    ▼
              messages.extend([AIMessage, ToolMessage])
                                    │
                                    ▼
                    llm.invoke(messages) ──▶ 最终自然语言回答
```

这与第28课、第24课 Agent 的 ReAct「Thought → Action → Observation」本质相同，只是 LangChain 用 `tool_calls` / `ToolMessage` 标准化了消息格式。

---

## 2. 环境配置

### 2.1 依赖

```bash
source activate_env.sh
pip install langchain langchain-deepseek langchain-core rich
```

### 2.2 环境变量

| 变量 | 必需 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | ✅ | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | 可选 | 自定义 API 地址 |

### 2.3 运行

```bash
python examples/llm-apps/15_langchain_tools.py
```

`main()` 中通过注释切换三个演示函数，默认运行 `_test_tool_call_with_tool_modiifier()`：

```python
# _test_tool_call_with_tool_specification()
# _test_tool_call_without_tool_specification()
_test_tool_call_with_tool_modiifier()
```

取消注释即可逐项体验。

---

## 3. `@tool` 装饰器 — 最简定义

### 3.1 定义与直接调用

```python
from langchain_core.tools import tool

@tool
def get_current_time(city: str) -> str:
    """获取当前时间"""
    return f"当前{city}时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
```

`@tool` 会把函数包装为 `StructuredTool`：
- **函数名** → tool `name`
- **docstring 首行** → tool `description`
- **类型注解** → JSON Schema 的 `parameters`

不经过模型，也可直接测试：

```python
result = get_current_time.invoke({"city": "北京"})
```

`invoke` 接受 dict 参数，与模型返回的 `tool_calls[0]["args"]` 结构一致。

### 3.2 绑定模型并触发调用 — `_test_tool_call_with_tool_specification`

```python
model_with_tools = _chat_deepseek().bind_tools([get_current_time])
messages = [HumanMessage(content="现在上海是什么时间？")]
response = model_with_tools.invoke(messages)
```

`bind_tools` 在底层把 Tool 列表序列化为 OpenAI `tools` 格式，发给 DeepSeek；模型若判断需要查时间，会在 `AIMessage` 里带上 `tool_calls`，而不是直接回答。

### 3.3 读取 tool_calls 并执行

```python
if response.tool_calls:
    rprint(response.tool_calls[0])
    # 典型结构：{"name": "get_current_time", "args": {"city": "上海"}, "id": "..."}

    if response.tool_calls[0]["name"] == "get_current_time":
        tool_message = get_current_time.invoke(response.tool_calls[0])
        messages.extend([response, tool_message])
        final_response = model_with_tools.invoke(messages)
        print(final_response.content)
```

要点：
1. **`response`** 是带 `tool_calls` 的 `AIMessage`，必须保留并 append 到 messages
2. **`tool_message`** 是 `ToolMessage`，`invoke(tool_call)` 会把 `id` 与结果关联
3. **第二次 `invoke`** 时模型根据 Observation 生成面向用户的自然语言

若模型认为不需要工具，`response.tool_calls` 为空，直接读 `response.content` 即可。

---

## 4. 无 `@tool`：普通函数 + `convert_to_openai_tool`

### 4.1 手写 docstring 的函数

```python
def get_weather(city: str) -> str:
    """
    获取某城市天气

    Args:
        city: 城市名称

    Returns:
        返回城市的天气
    """
    return f"当前{city}天气：晴天"
```

未加 `@tool` 时，函数仍是普通 callable；需要显式转成 OpenAI Tool Schema：

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

converted_tool = convert_to_openai_tool(get_weather)
rprint(converted_tool)
```

输出类似：

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取某城市天气",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string", "description": "城市名称"}
      },
      "required": ["city"]
    }
  }
}
```

### 4.2 bind_tools 的两种写法 — `_test_tool_call_without_tool_specification`

```python
# 写法 A：传入 convert 后的 dict（本示例打印 schema 用）
converted_tool = convert_to_openai_tool(get_weather)

# 写法 B：直接传函数，LangChain 内部自动转换（推荐）
model_with_tools = _chat_deepseek().bind_tools([get_weather])
response = model_with_tools.invoke(messages)
```

执行工具时，无 `@tool` 的函数不能 `.invoke(tool_call)`，需自行根据 `name` 分发，或先用 `@tool` 包装。本示例主要演示 **Schema 生成** 与 **bind_tools 触发**；完整闭环建议统一用 `@tool`。

---

## 5. `parse_docstring` 与参数描述 — `_test_tool_call_with_tool_modiifier`

### 5.1 带 Args 的 docstring

```python
@tool(description="加法函数", parse_docstring=True)
def add(a: int, b: int) -> int:
    """
    计算 a + b 的和

    Args:
        a: 加数
        b: 加数

    Returns:
        int: 和
    """
    return a + b
```

| 参数 | 作用 |
|------|------|
| `description=` | 覆盖默认 description（否则用 docstring 摘要） |
| `parse_docstring=True` | 从 `Args:` 段解析每个参数的中文说明，写入 Schema |

查看生成的 Schema：

```python
rprint(convert_to_openai_tool(add))
```

模型会看到 `a`、`b` 各有清晰描述，有利于在「计算 1 + 2」这类问题上正确选工具。

### 5.2 docstring 格式注意

示例代码中的注释强调：

```
注意: 空行分隔（即使包含空白字符也不可以）
```

`parse_docstring` 依赖 [Google 风格](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) 结构：`Args:` / `Returns:` 前需要**真正的空行**分隔摘要段与参数段，否则解析可能失败或丢失参数说明。

### 5.3 触发计算类 tool call

```python
messages = [HumanMessage(content="计算 1 + 2 的和")]
response = _chat_deepseek().bind_tools([add]).invoke(messages)
rprint(response)
```

观察 `tool_calls[0]["args"]` 是否为 `{"a": 1, "b": 2}`；再按 §3.3 补全执行与二次 invoke，即可得到最终答案。

---

## 6. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `@tool` | 函数 → `StructuredTool` | `get_current_time`, `add` |
| `tool.invoke(input)` | 同步执行工具 | `get_current_time.invoke(...)` |
| `llm.bind_tools([...])` | 给模型挂载 tools | 三个 `_test_*` |
| `AIMessage.tool_calls` | 模型请求的工具列表 | `_test_tool_call_with_tool_specification` |
| `convert_to_openai_tool(fn)` | 函数 → OpenAI tool dict | `get_weather`, `add` |
| `messages.extend([ai, tool_msg])` | 回填对话历史 | 工具闭环 |

---

## 7. 数据流总览

```
Python 函数
    │
    ├─ @tool ──────────────────▶ StructuredTool
    │
    └─ convert_to_openai_tool ─▶ OpenAI JSON Schema
                │
                ▼
        ChatDeepSeek.bind_tools([...])
                │
                ▼
        invoke([HumanMessage(...)])
                │
                ▼
        AIMessage.tool_calls?
           │              │
          否             是
           │              │
           ▼              ▼
      .content      tool.invoke(tool_call)
                           │
                           ▼
                    ToolMessage
                           │
                           ▼
              invoke(扩展后的 messages)
                           │
                           ▼
                    最终 .content
```

---

## 8. 与前面课程的关系

```
第28课 Function Calling     →  OpenAI tools JSON、手写解析 ✅ 原理
第24课 ReAct Agent          →  Thought / Action / Observation 循环
第32课 ChatDeepSeek.invoke  →  Message 列表输入
第34课 messages 多轮       →  append 历史
第35课 ChatPromptTemplate   →  参数化 system / user
第36课 LangChain Tools      →  @tool + bind_tools + 工具闭环 ✅ 你在这里
第29课 智能客服              →  多工具 + Memory + 生产级 Agent
```

本课补齐 **Tool 定义与最小调用环**：搞清 `tool_calls` 与 `ToolMessage` 后，再学 `create_react_agent`、`AgentExecutor` 会自动完成 §3.3 的循环。

---

## 9. 常见问题

### Q1: `bind_tools` 和请求里传 `tools=` 一样吗？

语义相同。`bind_tools` 返回一个绑定了 tools 的 Runnable，内部在每次 `invoke` 时把 Schema 附带给 API。

### Q2: 为什么第一次 `response.content` 经常是空的？

模型决定走工具时，会把「意图」放在 `tool_calls` 里，自然语言留到拿到 `ToolMessage` 后再生成。这是正常现象。

### Q3: 可以一次 `bind_tools` 多个函数吗？

可以：`bind_tools([get_current_time, add, get_weather])`。模型根据 description 选择其一或多个（视模型是否支持 parallel tool calls）。

### Q4: 无 `@tool` 的函数怎么执行？

三种方式：① 加 `@tool`；② 用 `StructuredTool.from_function(get_weather)`；③ 自己写 `if name == "get_weather": get_weather(**args)`。生产代码推荐 ①。

### Q5: `invoke(tool_call)` 传入整个 dict 可以吗？

可以。LangChain 的 `StructuredTool.invoke` 接受完整 tool_call（含 `name`、`args`、`id`），自动提取参数并生成带正确 `tool_call_id` 的 `ToolMessage`。

### Q6: 与 MCP、Skills 的关系？

- **MCP**（第30课）：把工具暴露为跨进程/跨应用协议
- **Skills**（第31课）：教 Agent 何时读哪份文档
- **本课 `@tool`**：在单进程 Python 里定义本地函数工具

三者可组合：Agent 的 tools 列表里既有 `@tool`，也可接 MCP Client 提供的远程工具。

---

## 10. 动手练习

1. **跑通闭环**：取消注释 `_test_tool_call_with_tool_specification`，确认终端先打印 `tool_calls` 再打印最终时间
2. **对比 Schema**：分别 `rprint(convert_to_openai_tool(get_current_time))` 与 `convert_to_openai_tool(add)`，观察 `parse_docstring` 对 `parameters` 的影响
3. **多工具**：`bind_tools([get_current_time, add])`，问「北京几点了，顺便算 3+5」，看模型是否连续或择一调用
4. **补全 weather 闭环**：给 `get_weather` 加 `@tool`，实现与 §3.3 相同的二次 invoke
5. **接第35课**：用 `ChatPromptTemplate` 加 system「你是助手，需要查时间或算数时请调用工具」，再 `bind_tools`，对比无 system 时的调用率

---

## 11. 参考

- 示例代码：`examples/llm-apps/15_langchain_tools.py`
- Function Calling 原理：`notes/phase4-projects/07_Function_Calling与Tools使用.md`
- 前置笔记：`notes/phase4-projects/15_LangChain_Prompt模板.md`
- Chat 调用：`notes/phase4-projects/12_LangChain进阶与DeepSeek接入.md`
- LangChain Tools：[Tools 文档](https://python.langchain.com/docs/concepts/tools/)

---

*完成本课后，你已掌握 LangChain 工具链的核心：`@tool` 定义、`bind_tools` 绑定、以及手动完成 tool call 闭环。这是构建 Agent、智能客服与自动化工作流的前置技能。*
