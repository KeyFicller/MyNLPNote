# 第41课：LangChain Agent 结构化输出

**项目**: `create_agent` + `response_format` 在 Agent 循环内产出 Pydantic 对象  
**技术栈**: LangChain v1, langchain.agents, langchain.agents.structured_output, Pydantic, langchain-deepseek  
**示例代码**: `examples/langchain/10_langchain_agent_output.py`  
**前置课程**: 第39课 Pydantic 结构化输出、第40课 LangChain Agent  
**环境与运行**：见 [第32课 §1 环境配置](01_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/langchain/10_langchain_agent_output.py`（`main()` 依次运行两段演示）

---

## 课程概述

第39课用 `llm.with_structured_output(PydanticModel)` 在**单次模型调用**中约束输出形状。第40课用 `create_agent` 封装 **ReAct 工具循环**。本课将二者合并：在 `create_agent` 上传入 `response_format`，让 Agent 在对话/推理过程中自动产出符合 Schema 的结构化对象，并通过 `response["structured_response"]` 直接读取。

示例定义 `ContractInfo`（姓名、11 位手机号、邮箱），分两段：第一段从含完整信息的文本中抽取；第二段面对信息缺失的文本，配合 `handle_errors` 自定义校验失败提示、`system_prompt` 禁止捏造，以及 `recursion_limit` 限制重试轮数。DeepSeek 不支持原生 `ProviderStrategy`，示例显式使用 `ToolStrategy`。

**学习目标：**
1. 理解 `response_format` 与第39课 `with_structured_output` 的分工与组合方式
2. 掌握三种策略：`ToolStrategy`、`ProviderStrategy`、`AutoStrategy`
3. 会用 Pydantic `BaseModel` 作为 Agent 的结构化输出 Schema
4. 会从 `invoke` 返回值中读取 `structured_response` 与完整 `messages` 轨迹
5. 会用 `handle_errors` 处理校验失败，并用 `recursion_limit` 控制重试上限

---

## 1. 为什么需要 Agent 结构化输出？

### 1.1 单次抽取 vs Agent 内抽取

| 维度 | 第39课 `with_structured_output` | 本课 `create_agent(response_format=...)` |
|------|--------------------------------|------------------------------------------|
| 调用形态 | 一次 `llm.invoke` | Agent 图内多轮 model ↔ tools 循环 |
| 工具 | 无（或需另建 Agent） | 可同时绑定业务 `tools` + 结构化输出「虚拟工具」 |
| 校验失败 | 抛 `ValidationError`，需手写重试 | `ToolStrategy.handle_errors` 可将错误回填为 `ToolMessage` 触发重试 |
| 结果读取 | 直接得 Pydantic 实例 | `response["structured_response"]` + `messages` 轨迹 |

第39课适合「纯抽取、无工具」；本课适合「抽取 + 可能调其它工具 + 校验失败自动纠正」的一体化 Agent。

### 1.2 数据流（ToolStrategy）

```
HumanMessage（待抽取文本）
        │
        ▼
┌───────────────────────────────────────┐
│  model 节点：LLM 调用「结构化输出工具」  │
│  → AIMessage(tool_calls=[ContractInfo])│
└───────────────┬───────────────────────┘
                ▼
┌───────────────────────────────────────┐
│  解析 tool_args → Pydantic 校验         │
│  成功 → structured_response + 结束     │
│  失败 → handle_errors 生成 ToolMessage  │
│         → 回到 model 节点重试           │
└───────────────────────────────────────┘
```

`ToolStrategy` 将 Pydantic Schema **包装成虚拟 StructuredTool**，模型通过 `tool_calls` 填参；LangChain 解析参数并校验，成功则写入 `structured_response`。

### 1.3 与第40课的关系

第40课 `create_agent` 的 `tools` 是**用户定义的可执行工具**（天气、搜索等）。本课的 `response_format` 是**输出契约**，不替代业务工具——二者可同时存在：Agent 先调 API 拿 raw 数据，再以结构化 Schema 输出最终结果。

---

## 2. 三种 response_format 策略

```python
from langchain.agents.structured_output import AutoStrategy, ProviderStrategy, ToolStrategy
```

| 策略 | 机制 | 适用模型 |
|------|------|----------|
| `ToolStrategy(schema)` | 将 Schema 注册为虚拟 tool，模型 `tool_calls` 填参 | **通用**，DeepSeek 等无原生 structured output 的模型 |
| `ProviderStrategy(schema)` | 调用模型厂商原生 JSON Schema / structured output API | OpenAI、Anthropic 等支持 native mode 的模型 |
| `AutoStrategy(schema)` | LangChain 按模型能力自动选 Tool 或 Provider | 跨模型部署、不确定 provider 能力时 |

示例中的注释说明：

```python
# Deepseek doesn't support ProviderStrategy, so we use ToolStrategy instead.
response_format = ToolStrategy(ContractInfo, tool_message_content="成功提取内容")
# response_format = ProviderStrategy(ContractInfo)
# response_format = AutoStrategy(ContractInfo)
```

也可直接传 Pydantic 类（等价于 `AutoStrategy`）：

```python
create_agent(model=..., response_format=ContractInfo)  # 内部包装为 AutoStrategy
```

---

## 3. 输出 Schema — `ContractInfo`

```python
from pydantic import BaseModel, Field

class ContractInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户的姓名", min_length=1)
    phone: str = Field(description="用户的11位手机号码", min_length=11, max_length=11)
    email: str = Field(description="用户的电子邮箱")
```

| 要点 | 说明 |
|------|------|
| `Field(description=...)` | 写入 JSON Schema，引导模型理解字段语义（与第37、39课相同） |
| `min_length` / `max_length` | Pydantic 校验；模型填错会在 `ToolStrategy` 路径触发校验失败 |
| 中文 description | 中文抽取任务建议使用中文描述，利于模型对齐字段 |

示例中还预留了 `@field_validator`（当前用三引号注释掉），用于拒绝「不详」「00000000000」「@example.com」等占位值——取消注释后可与 `handle_errors` 联用，演示更严格的业务校验。

---

## 4. 基础抽取 — ToolStrategy

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

agent = create_agent(
    model=init_chat_model(
        "deepseek:deepseek-v4-pro",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    response_format=ToolStrategy(
        ContractInfo,
        tool_message_content="成功提取内容",
    ),
)

messages = [
    HumanMessage(
        content="从这段话中提取信息，小明的邮箱地址为：xiaoming@163.com，手机号：12345678912"
    )
]
response = agent.invoke({"messages": messages})
```

### 4.1 `tool_message_content`

校验**成功**后，虚拟结构化工具返回的 `ToolMessage` 内容。默认类似 `"Returning structured response: ..."`；设为 `"成功提取内容"` 便于在 `messages` 轨迹中识别抽取完成。

### 4.2 打印轨迹

```python
for msg in response.get("messages"):
    msg.pretty_print()

rprint(response.get("messages")[-1].content)
rprint(response.get("structured_response"))
```

典型流程：`HumanMessage` → `AIMessage`（含对 `ContractInfo` 的 `tool_calls`）→ `ToolMessage`（成功提示）→ 可能还有最终 `AIMessage`。**业务侧应优先读 `structured_response`**，它是已校验的 Pydantic 对象（或 dict，视版本而定）。

---

## 5. 返回值字段

| 键 | 类型 | 说明 |
|----|------|------|
| `messages` | `list[BaseMessage]` | 完整 Agent 轨迹（与第40课相同） |
| `structured_response` | `ContractInfo` 或 `None` | 结构化输出成功时的解析结果 |

```python
response.get("structured_response")
# ContractInfo(name='小明', phone='12345678912', email='xiaoming@163.com')
```

若未配置 `response_format`，则无 `structured_response` 字段或为 `None`。

---

## 6. 校验失败与 handle_errors

第二段演示面对**信息不完整**的输入：

```python
agent_with_error_handling = create_agent(
    model=init_chat_model(
        "deepseek:deepseek-v4-pro",
        extra_body={"thinking": {"type": "disabled"}},
    ),
    response_format=ToolStrategy(
        ContractInfo,
        tool_message_content="成功提取内容",
        handle_errors="校验失败：姓名、11位手机号、邮箱必须严格来自原文，禁止编造或使用占位值。",
    ),
    system_prompt=(
        "你是信息抽取助手。只抽取文本中明确出现的姓名、11位手机号和邮箱，捏造行为将视为严重违法"
    ),
)

messages = [
    HumanMessage(content="从这段话中提取信息：邮箱地址缺失，手机号只有三位：123")
]
response = agent_with_error_handling.invoke({"messages": messages}, config={"recursion_limit": 5})
```

### 6.1 `handle_errors` 取值

| 值 | 行为 |
|----|------|
| `True`（默认） | 捕获校验错误，用默认模板生成 `ToolMessage` 提示模型重试 |
| `str` | 捕获错误，用**自定义字符串**作为 `ToolMessage` 内容（本例） |
| `type[Exception]` / `tuple[...]` | 只捕获指定异常类型 |
| `Callable[[Exception], str]` | 自定义函数，根据异常生成提示 |
| `False` | 不捕获，校验失败直接向上抛出 |

当模型「编造」11 位手机号或占位邮箱时，Pydantic 校验失败 → LangChain 将 `handle_errors` 文案写入 `ToolMessage` → model 节点再次推理，形成**自愈循环**。

### 6.2 `system_prompt` 的配合

`system_prompt` 从指令层禁止捏造；`handle_errors` 从反馈层在失败后纠正。二者叠加比单独依赖 prompt 更稳定。

---

## 7. recursion_limit 与重试上限

```python
config = {"recursion_limit": 5}
response = agent.invoke({"messages": messages}, config=config)
```

| 项 | 说明 |
|----|------|
| 作用 | LangGraph 图执行的**最大步数**（含 model、tools、结构化输出解析等节点） |
| 本课场景 | 校验失败会多轮重试；限制为 5 避免模型反复编造导致无限循环 |
| 与第40课 | 无 `response_format` 时主要限制 tool 循环次数；有结构化输出时同样适用 |

信息确实无法凑齐 Schema 时，Agent 可能在达到 `recursion_limit` 后停止；此时 `structured_response` 可能仍为 `None`，应结合 `messages` 最后几条判断是「如实无法抽取」还是「中途截断」。

---

## 8. DeepSeek 与 thinking 模式

与第39课一致，示例通过 `init_chat_model` 的 `extra_body` 关闭 thinking：

```python
extra_body={"thinking": {"type": "disabled"}}
```

结构化输出依赖稳定的 tool call 参数形态；thinking 模式可能影响格式或与部分 API 特性冲突。

---

## 9. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `create_agent(..., response_format=...)` | 创建带结构化输出的 Agent | 全部演示 |
| `ToolStrategy` | 虚拟 tool 填参策略 | 主路径 |
| `ProviderStrategy` | 厂商原生 structured output | 注释备选 |
| `AutoStrategy` | 自动选择策略 | 注释备选 |
| `tool_message_content` | 成功时的 ToolMessage 文案 | `ToolStrategy(...)` |
| `handle_errors` | 校验失败反馈与重试 | 第二段演示 |
| `response["structured_response"]` | 结构化结果 | 两段 invoke 末尾 |
| `response["messages"]` | 完整轨迹 | `pretty_print` 循环 |
| `config["recursion_limit"]` | 最大图步数 | 第二段 invoke |
| `ContractInfo` | Pydantic 输出 Schema | 模型定义 |
| `init_chat_model` | 模型初始化 + extra_body | 两段 Agent |

---

## 10. 与前面课程的关系

```
第37课 args_schema              →  Tool 入参 Schema
第39课 with_structured_output   →  单次 LLM 出参 Schema ✅ 语法相同
第40课 create_agent             →  Agent 工具循环 ✅ 本课在其上叠加输出契约
第41课 response_format          →  Agent 内结构化输出 + 校验重试 ✅ 你在这里
```

典型组合：**Agent 调搜索/API 工具** → **`response_format` 抽成业务对象** → 写库；或 **纯文本信息抽取** → **`structured_response` 直接入库**。

---

## 11. 常见问题

### Q1: 和 `with_structured_output` 能一起用吗？

不要在同一轮混用两种「出参约束」。选其一：简单抽取用 `with_structured_output`；需要工具循环或校验重试用 `create_agent(response_format=...)`。

### Q2: DeepSeek 为什么必须用 ToolStrategy？

DeepSeek 当前不提供 OpenAI 式 `response_format: json_schema` 原生接口。`ProviderStrategy` 依赖厂商 native API；对 DeepSeek 应显式 `ToolStrategy` 或让 `AutoStrategy` 回退到 Tool 路径。

### Q3: `structured_response` 是 dict 还是 BaseModel？

通常为 **Pydantic 实例**（与第39课一致）。可用 `.model_dump()`、`model_dump_json()` 序列化。

### Q4: 校验一直失败，Agent 不停重试？

检查 Schema 是否过严（原文根本没有 11 位手机号却要求必填）；放宽为 `Optional` 或拆分 Schema；降低 `recursion_limit` 并捕获图执行超时；加强 `system_prompt` 允许「字段缺失时报错说明」而非编造。

### Q5: 能否同时有业务 tools 和 response_format？

可以。`tools=[...]` 与 `response_format=ToolStrategy(...)` 并存：业务工具与结构化虚拟工具分别出现在模型的 tool 列表中。

### Q6: `handle_errors=False` 会怎样？

校验失败直接抛异常（如 `StructuredOutputValidationError`），不会自动重试。适合要严格失败、由上层捕获的场景。

### Q7: 注释里的 `@field_validator` 何时启用？

当业务要拒绝占位符、全零手机号等**规则型校验**时，取消三引号注释即可；校验抛错会走 `handle_errors` 路径（默认 `True` 或自定义 str）。

---

## 12. 动手练习

1. **运行两段演示**：观察第一段 `structured_response` 与第二段 `messages` 中多轮 ToolMessage 差异
2. **切换策略**：将 `ToolStrategy` 改为 `AutoStrategy(ContractInfo)`，确认 DeepSeek 下仍走 tool 路径
3. **放宽 Schema**：将 `phone` 改为 `Optional[str]`，对「手机号只有三位」看是否仍重试或返回部分字段
4. **启用 validator**：取消 `ContractInfo` 内 `@field_validator` 注释，输入含「姓名：不详」的文本，观察 `handle_errors` 提示
5. **调 recursion_limit**：设为 `2`，看信息缺失时是否在 limit 前停止
6. **加业务工具**：增加 `@tool def lookup_carrier(phone: str)`，问「根据下面文本查运营商并结构化输出联系人」
7. **对比第39课**：同一 `ContractInfo` 与同一 HumanMessage，分别用 `with_structured_output` 与本课 Agent 各跑一遍，比较代码行数与失败重试行为

---

## 13. 参考

- 示例代码：`examples/langchain/10_langchain_agent_output.py`
- 公共客户端：`examples/langchain/deepseek_client.py`
- 前置笔记：`notes/phase-langchain/08_LangChain_Pydantic.md`（Pydantic 与 `with_structured_output`）
- Agent 基础：`notes/phase-langchain/09_LangChain_Agent.md`（`create_agent` 与 invoke）
- 环境配置：`notes/phase-langchain/01_LangChain进阶与DeepSeek接入.md`
- LangChain Structured output：[Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
- `create_agent` API：[Reference](https://reference.langchain.com/python/langchain/agents/create_agent)

---

*完成本课后，你已能在 Agent 图内定义结构化输出契约：通过 `ToolStrategy` 在 DeepSeek 上完成信息抽取，用 `structured_response` 读取校验后的 Pydantic 对象，并用 `handle_errors` 与 `recursion_limit` 控制失败重试。它把第39课的「输出形状」与第40课的「Agent 循环」合成一条生产可用的抽取链路。*
