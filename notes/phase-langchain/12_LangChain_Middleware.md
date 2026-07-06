# 第43课：LangChain Agent Middleware

**项目**: `create_agent` + `middleware` 在 Agent 循环各阶段插入自定义逻辑  
**技术栈**: LangChain v1.3+, langchain.agents, langchain.agents.middleware, LangGraph  
**示例代码**: `examples/langchain/12_langchain_middleware.py`  
**参考课件**: 尚硅谷-08-中间件.pdf  
**前置课程**: 第40课 LangChain Agent、第42课 Agent 流式输出  
**环境与运行**：见 [第32课 §1 环境配置](01_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/langchain/12_langchain_middleware.py`（`main()` 默认 §2.1，注释切换其它演示）

---

## 课程概述

第40课用 `create_agent` 得到 **model ↔ tools** 的 ReAct 循环。真实项目中还需要日志、限流、PII 脱敏、人工审批、上下文摘要等**横切逻辑**——若全部写进 tools 或手写图节点，主流程会迅速膨胀。LangChain v1 的 **Middleware（中间件）** 在 Agent 执行的关键节点暴露钩子，在不改主流程的前提下实现「拦截、修改、增强」。

本课示例对齐尚硅谷课件结构，覆盖：
- **§2 常用内置**：Summarization、HumanInTheLoop、PII、TodoList
- **§3 其它内置**：ModelCallLimit（示例中实现）
- **§4 组合顺序**：洋葱模型 1→2→3→模型→3→2→1
- **§5 自定义**：装饰器 / `AgentMiddleware` 子类

**学习目标：**
1. 理解 Middleware 在 Agent 架构中的位置（Model / Tools / System Prompt / **Middleware**）
2. 掌握 `SummarizationMiddleware` 的 trigger / keep / summary_prompt
3. 会用 `HumanInTheLoopMiddleware` + `InMemorySaver` + `Command(resume=...)`
4. 掌握 `PIIMiddleware` 四种 strategy 与自定义 detector
5. 了解 `TodoListMiddleware` 多步任务规划
6. 理解多 Middleware 的执行顺序（洋葱模型）
7. 会用装饰器或 `AgentMiddleware` 子类编写自定义 hook

---

## 1. 中间件概述（课件 §1）

### 1.1 什么是中间件

Middleware 是 Agent 执行过程中的**钩子函数**，在「模型调用前/后」「工具调用前/后」等固定时机被框架自动调用。主流程预留插槽，开发者挂上自己的函数，无需改 LangGraph 源码。

```
用户输入 → before_agent → before_model → [模型] → after_model
                ↓ 有 tool_calls
           wrap_tool_call → [工具] → 回到 before_model
                ↓ 无 tool_calls
           after_agent → 返回
```

### 1.2 为什么需要中间件

| 需求 | 无 Middleware | 有 Middleware |
|------|--------------|---------------|
| 动态切换模型 | 改主流程 | `ModelFallbackMiddleware` |
| 限制调用次数 | 手写计数 | `ModelCallLimitMiddleware` |
| PII 脱敏 | 每个 tool 重复 | `PIIMiddleware` |
| 人工审批 | 极难实现 | `HumanInTheLoopMiddleware` |
| 上下文压缩 | 手动 trim | `SummarizationMiddleware` |

### 1.3 内置中间件六类（课件 §1.4）

| 类别 | 代表 Middleware | 目标 |
|------|-----------------|------|
| 成本与资源控制 | Model/Tool call limit, Summarization, Context editing | 控费、控配额 |
| 稳定性与容错 | Model fallback/retry, Tool retry | 高可用 |
| 安全与合规 | HITL, PII detection | 可控、合规 |
| 决策增强 | TodoList, LLM tool selector, Subagent | 规划、筛工具 |
| 执行能力扩展 | Shell tool, File search, Filesystem | 操作环境 |
| 开发调试 | LLM tool emulator | Mock 测试 |

---

## 2. 常用内置中间件（课件 §2）

### 2.1 SummarizationMiddleware

**作用**：历史消息过长时自动摘要，压缩上下文。  
**原理**：达到 trigger 条件 → 调用摘要模型 → 摘要以 `HumanMessage` 插入列表头部 → keep 保留最新 N 条原消息。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage, AIMessage

custom_profile = {"max_input_tokens": 128_000}  # DeepSeek 需手动配置才能用 fraction
summary_model = init_chat_model(
    model="deepseek:deepseek-v4-pro",
    profile=custom_profile,
    api_key=api_key(),
    base_url=api_base(),
)

messages = [
    SystemMessage("你是个非常友好的AI助手"),
    HumanMessage("你好啊，我是老王，你是谁？"),
    AIMessage("你好老王，我是小王"),
    HumanMessage("好的小王，很高兴认识你"),
    AIMessage("你高兴得太早了"),
    HumanMessage("呵呵，你什么意思"),
]

agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    middleware=[
        SummarizationMiddleware(
            model=summary_model,
            trigger=[("tokens", 100), ("messages", 6), ("fraction", 0.001)],
            keep=("messages", 2),
        )
    ],
)
response = agent.invoke({"messages": messages})
```

| 参数 | 说明 |
|------|------|
| `model` | 摘要用的模型（字符串或实例） |
| `trigger` | 触发条件列表，任一满足即摘要：`tokens` / `messages` / `fraction` |
| `keep` | 摘要后保留的原始消息（三种度量选一） |
| `summary_prompt` | 自定义提示词，需含 `{messages}` 占位符 |
| `trim_tokens_to_summarize` | 参与摘要的历史最大 token，默认 4000 |

**示例函数**：`summarization_middleware_test()` / `summarization_custom_prompt_test()`

**观察要点**：
1. 需 `profile.max_input_tokens` 才能用 `fraction` trigger
2. 三个 trigger 条件**任一**满足即触发
3. 摘要 HumanMessage 出现在列表头部
4. `keep=("messages", 2)` 保留最新 2 条原消息

---

### 2.2 HumanInTheLoopMiddleware

**作用**：工具调用前**中断** Agent，等待人工 approve / edit / reject。  
**依赖**：`checkpointer=InMemorySaver()` + 固定 `thread_id`。

```python
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    tools=[get_weather, get_news, read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "get_weather": True,           # approve / edit / reject 均可
                "get_news": True,
                "read_email_tool": False,      # 不中断
                "send_email_tool": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "发送邮件中断啦",
                },
            },
            description_prefix="中断啦",
        )
    ],
)

config = {"configurable": {"thread_id": "middleware-hitl-1"}}
response = agent.invoke({"messages": [...]}, config=config)

# 第一次 invoke 返回 __interrupt__
interrupts = response.get("__interrupt__", [])
action_requests = interrupts[0].value["action_requests"]

# 按 action_requests 顺序构造 decisions
decisions = {"decisions": [
    {"type": "edit", "edited_action": {"name": "get_weather", "args": {...}}},
    {"type": "approve"},
    {"type": "approve"},
]}
resumed = agent.invoke(Command(resume=decisions), config=config)
```

**示例函数**：`human_in_the_loop_test()`

**关键点**：
- `interrupt_on` 的 key 是 `@tool` **函数名**
- `read_email_tool: False` 的工具会直接执行，不出现在 interrupt 列表
- `decisions` 顺序必须与 `action_requests` 一致
- resume 时必须使用**相同** `thread_id`

---

### 2.3 PIIMiddleware

**作用**：检测并处理个人身份信息（PII）。

| strategy | 效果 | 场景 |
|----------|------|------|
| `redact` | 替换为 `[REDACTED_EMAIL]` 等 | 日志清洗、合规 |
| `mask` | 部分遮蔽 `****-****-****-5100` | 前端展示 |
| `hash` | 替换为 `<url_hash:...>` | 匿名追踪 |
| `block` | 直接抛异常 | 零容忍 |

```python
agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    tools=[],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware("url", strategy="hash", apply_to_input=True),
        PIIMiddleware("mac_address", strategy="mask", apply_to_input=True),
        PIIMiddleware("ip", strategy="block", apply_to_input=True),
    ],
)
```

**自定义 detector**（课件 2.3.3）：

```python
import re

def detect_phone_number(content: str):
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"[0-9]{11}", content)
    ]

PIIMiddleware("api_key", strategy="hash", detector=r"sk-[a-zA-Z0-9]+")
PIIMiddleware("phone_number", strategy="mask", detector=detect_phone_number)
```

**示例函数**：`pii_middleware_test()` / `pii_custom_detector_test()`

---

### 2.4 TodoListMiddleware

**作用**：通过内置 `write_todos` 工具，让 Agent 先列 CheckList 再逐步执行复杂多步任务。

**课件场景**：修复 `todo_workspace/my_add.py`（故意写成 `a - b`），跑 pytest 验证。

工作区文件（示例运行时自动创建）：
- `examples/langchain/todo_workspace/my_add.py`
- `examples/langchain/todo_workspace/test_my_add.py`

```python
agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    tools=[list_files, read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
    system_prompt="遇到多步骤任务时，先使用 write_todos 制定待办事项...",
)
```

**示例函数**：`todo_list_middleware_test()`（耗时较长，需 pytest）

**何时使用 TodoList**：
- 步骤 ≥ 3 且有依赖关系 → 适合
- 简单问答 / 单次工具调用 → 不必用，浪费 token

---

## 3. 其它内置中间件（课件 §3，示例实现 §3.1）

课件为大部分中间件提供测试代码，本仓库示例实现 **ModelCallLimitMiddleware**：

```python
agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    checkpointer=InMemorySaver(),  # thread_limit 需要
    tools=[],
    middleware=[ModelCallLimitMiddleware(thread_limit=2, exit_behavior="end")],
)
config = {"configurable": {"thread_id": "middleware-mcl-1"}}
# 同一 thread 第 3 次 invoke 会达到上限
```

| 参数 | 说明 |
|------|------|
| `run_limit` | 单次 invoke 内 model 调用上限 |
| `thread_limit` | 跨 invoke 的 thread 级上限（需 checkpointer） |
| `exit_behavior` | `"end"` 优雅退出 / `"error"` 抛 `ModelCallLimitExceededError` |

课件 §3 还涵盖（见官方文档）：`ToolCallLimitMiddleware`、`ModelFallbackMiddleware`、`LLMToolSelectorMiddleware`、`ToolRetryMiddleware`、`ModelRetryMiddleware`、`LLMToolEmulator`、`ContextEditingMiddleware` 等。

**示例函数**：`model_call_limit_test()`

---

## 4. 多个中间件组合及执行顺序（课件 §4）

Middleware 书写顺序**非常重要**，类似洋葱模型：

```python
middleware=[Middleware1(), Middleware2(), Middleware3()]
```

执行顺序：

```
Middleware1.before_model  ↓ 正序
Middleware2.before_model  ↓
Middleware3.before_model  ↓
        [模型调用]
Middleware3.after_model   ↑ 逆序
Middleware2.after_model   ↑
Middleware1.after_model   ↑
```

终端输出：

```
[中间件1] before_model
[中间件2] before_model
[中间件3] before_model
[中间件3] after_model
[中间件2] after_model
[中间件1] after_model
```

**示例函数**：`middleware_order_test()`

推荐组合顺序（课件建议）：

```python
middleware=[
    PIIMiddleware(...),              # 1. 最先检查敏感信息
    ModelCallLimitMiddleware(...),   # 2. 限制调用次数
    SummarizationMiddleware(...),    # 3. 压缩历史
    ToolRetryMiddleware(...),        # 4. 工具重试
]
```

---

## 5. 自定义中间件（课件 §5）

### 5.1 Hook 函数分类

| 类型 | Hook | 风格 | 典型用途 |
|------|------|------|----------|
| Node-style | `before_agent` / `before_model` / `after_model` / `after_agent` | 顺序执行 | 日志、验证、改 state |
| Wrap-style | `wrap_model_call` / `wrap_tool_call` | 包裹调用 | 重试、缓存、mock |

### 5.2 装饰器实现（课件 5.3.1）

```python
from langchain.agents.middleware import before_model, after_model, before_agent, after_agent

@before_model
def before_model_middleware(state, runtime):
    state["messages"][-1].content += " -> before_model <- "
    return None

agent = create_agent(
    model="deepseek:deepseek-v4-pro",
    middleware=[before_model_middleware, after_model_middleware,
                before_agent_middleware, after_agent_middleware],
)
```

输出中 HumanMessage 带 `-> before_agent <- -> before_model <-`，AIMessage 带 `-> after_model <- -> after_agent <-`。

**示例函数**：`custom_decorator_middleware_test()`

### 5.3 类实现（课件 5.3.1）

```python
class MyMiddleware(AgentMiddleware):
    def before_model(self, state, runtime):
        state["messages"][-1].content += " -> before_model <- "
        return None
    # ... after_model / before_agent / after_agent 同理

agent = create_agent(model="...", middleware=[MyMiddleware()])
```

规则：必须继承 `AgentMiddleware`；方法名固定；类名随意。

**示例函数**：`custom_class_middleware_test()`

---

## 6. 示例 main() 切换指南

| 函数 | 课件章节 | 需要 API | 备注 |
|------|----------|----------|------|
| `summarization_middleware_test` | §2.1.2 | ✅ | **默认** |
| `summarization_custom_prompt_test` | §2.1.3 | ✅ | 自定义 summary_prompt |
| `human_in_the_loop_test` | §2.2 | ✅ | 需 checkpointer + resume |
| `pii_middleware_test` | §2.3.2 | ✅ | 含 block 异常演示 |
| `pii_custom_detector_test` | §2.3.3 | ✅ | 正则 + 函数 detector |
| `todo_list_middleware_test` | §2.4 | ✅ | 需 pytest，耗时较长 |
| `model_call_limit_test` | §3.1 | ✅ | 需 checkpointer |
| `middleware_order_test` | §4 | ✅ | 观察打印顺序 |
| `custom_decorator_middleware_test` | §5.3.1 | ✅ | 装饰器 hook |
| `custom_class_middleware_test` | §5.3.1 | ✅ | 类 hook |

---

## 7. 常见问题

### Q1: Middleware 和 LangSmith 有什么区别？

LangSmith 偏**事后 trace**；Middleware 偏**运行时干预**（改 state、限流、脱敏、短路）。

### Q2: Summarization 的 fraction trigger 报错？

DeepSeek 模型 profile 默认无 `max_input_tokens`，需手动传入 `profile={"max_input_tokens": 128_000}`。

### Q3: HITL resume 后工具没执行？

检查 `decisions` 顺序是否与 `action_requests` 一致；`thread_id` 是否相同。

### Q4: PII block 后如何捕获？

用 `try/except` 捕获异常，消息类似 `Detected 1 instance(s) of ip in text content`。

### Q5: TodoList 什么时候不该用？

简单 1–2 步任务不必用 `write_todos`，浪费 token 且增加延迟。

### Q6: 装饰器和类 Middleware 怎么选？

单 hook、无配置 → 装饰器；多 hook、可配置、可测试 → `AgentMiddleware` 子类。

---

## 8. 动手练习

1. **跑通 §2.1**：观察摘要 HumanMessage 与 keep 保留的 2 条原消息
2. **对比 summary_prompt**：分别运行两个 Summarization 演示，看摘要风格差异
3. **HITL edit**：对 `get_weather` 使用 `type: edit` 改 city 参数，验证 ToolMessage
4. **PII 四种 strategy**：对照课件输出，理解 redact / mask / hash / block
5. **洋葱模型**：运行 `middleware_order_test()`，手绘执行顺序图
6. **修复 my_add.py**：运行 `todo_list_middleware_test()`，确认 pytest 通过
7. **自定义 hook**：在 `@before_model` 中统计 messages 条数而非改 content

---

## 9. 参考

- 示例代码：`examples/langchain/12_langchain_middleware.py`
- 工作区：`examples/langchain/todo_workspace/`（TodoList 演示）
- 公共客户端：`examples/langchain/deepseek_client.py`
- 前置笔记：`notes/phase-langchain/09_LangChain_Agent.md`
- LangChain Middleware：[Overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- 自定义 Middleware：[Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- 课件：尚硅谷-08-中间件.pdf

---

*完成本课后，你已掌握 LangChain v1 Middleware 的完整工程化能力：从内置的摘要、审批、PII、Todo 到自定义 hook 与组合顺序——这是构建可治理、可合规、可观测生产 Agent 的核心技能。*
