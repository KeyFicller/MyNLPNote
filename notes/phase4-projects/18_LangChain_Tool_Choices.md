# 第38课：LangChain Tool Choices

**项目**: `bind_tools` 的 `tool_choice` 参数与调用策略控制  
**技术栈**: LangChain, langchain-core, langchain-deepseek, ChatDeepSeek  
**示例代码**: `examples/llm-apps/17_langchain_tool_choices.py`  
**前置课程**: 第36课 LangChain Tools 工具、第37课 Tool Schema  
**环境与运行**：见 [第32课 §1 环境配置](12_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/llm-apps/17_langchain_tool_choices.py`（连续四次 invoke 对比 `tool_choice`）

---

## 课程概述

第36–37课已经会用 `@tool` 定义工具、用 `bind_tools` 挂载到 `ChatDeepSeek`，模型默认在「是否调用工具、调用哪一个」上自行决策。本课聚焦 **`tool_choice`**：在 `bind_tools(..., tool_choice=...)` 里显式约束模型的工具调用行为——禁止调用、自动选择、强制至少调用一次、或**指定某个工具名**。

示例用 `add` / `subtract` 两个简单算术工具，对同一问题 `"1+1=?"` 依次演示四种策略，并说明 DeepSeek **思考模式（thinking）** 下部分 `tool_choice` 需关闭 thinking 才能生效。

**学习目标：**
1. 理解 `tool_choice` 在 OpenAI 兼容 API 中的语义
2. 掌握 `none` / `auto` / `required` / 工具名 四种常见取值
3. 会在 LangChain 里用 `bind_tools(..., tool_choice=...)` 控制调用策略
4. 了解 DeepSeek thinking 模式与 `tool_choice` 的兼容限制
5. 能根据业务场景（纯聊天 vs 必调工具 vs 指定工具）选对策略

---

## 1. 为什么需要 tool_choice？

### 1.1 默认行为 vs 显式控制

| 场景 | 期望行为 | 推荐 `tool_choice` |
|------|----------|-------------------|
| 闲聊、知识问答，工具只是「可选增强」 | 能算就算，不能算就直答 | `"auto"`（默认） |
| 已绑定工具但本轮**绝不要**调工具 | 只生成文本 | `"none"` |
| 工作流下一步**必须**走工具拿结构化结果 | 至少产生一次 `tool_calls` | `"required"` |
| 编排已知步骤，只要某个固定工具 | 只调 `subtract`，不调 `add` | `"subtract"`（工具函数名） |

不设 `tool_choice` 时，LangChain 行为等价于 OpenAI 默认：模型看到 tools 列表，自行决定是否调用。生产里常见需求是「这一步必须查库」「这一步禁止调外部 API」——这就需要本课参数。

### 1.2 数据流

```
@tool 定义的 add / subtract
        │
        ▼
bind_tools([add, subtract], tool_choice="???")
        │
        ▼
invoke([HumanMessage("1+1=?")])
        │
        ├── tool_choice="none"     ──▶ AIMessage.content（无 tool_calls）
        ├── tool_choice="auto"     ──▶ 可能 tool_calls 或 content
        ├── tool_choice="required" ──▶ 必有 tool_calls（通常 add 或 subtract）
        └── tool_choice="subtract" ──▶ tool_calls 仅 subtract
```

`tool_choice` 只影响**这一轮**模型输出是否含 `tool_calls`、以及调用哪个工具；工具执行与二次 `invoke` 仍按第36课闭环处理。

---

## 2. 示例工具：add 与 subtract

```python
@tool
def add(a: int, b: int) -> int:
    """计算 a + b 的和"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """计算 a - b 的差"""
    return a - b
```

与第36课相同：`@tool` 自动生成 Schema，函数名 `add` / `subtract` 即 **`tool_choice` 可引用的工具名**。绑定两个工具后，模型在 `auto` 下可在二者间选择；在 `tool_choice="subtract"` 下被限制为只能发起 `subtract` 调用。

---

## 3. tool_choice 四种演示

### 3.1 `"none"` — 禁止工具调用

```python
model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="none")
rprint(model_with_tools.invoke(messages))
```

| 项 | 说明 |
|----|------|
| API 语义 | 等价于 OpenAI `tool_choice: "none"` |
| 典型结果 | `AIMessage.content` 为「2」等文本，`tool_calls` 为空 |
| 适用 | 纯对话轮次、工具已绑定但暂不可用、A/B 对比「有工具定义但不许用」 |

即使 messages 里是明显算术题，模型也**不能**输出 `tool_calls`，只能在 `content` 里心算或说明。

### 3.2 `"auto"` — 模型自行决定（默认策略）

```python
model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="auto")
rprint(model_with_tools.invoke(messages))
```

| 项 | 说明 |
|----|------|
| API 语义 | 模型可选：不调工具 / 调 add / 调 subtract |
| 典型结果 | 多数会 `tool_calls` → `add`，`args` 如 `{"a": 1, "b": 1}`；也可能直接 `content="2"` |
| 适用 | **通用 Agent**、智能客服、不确定是否需查库时的默认选项 |

第28课 OpenAI 示例里的 `tool_choice="auto"` 与本节一致；第29课智能客服 DeepSeek 流程图也采用 `auto`。

### 3.3 `"required"` — 强制至少调用一个工具

```python
model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="required")
rprint(model_with_tools.invoke(messages, extra_body={"thinking": {"type": "disabled"}}))
```

| 项 | 说明 |
|----|------|
| LangChain 语义 | `'required'`、`'any'`、`True` 均表示「必须产生 tool_calls」 |
| 典型结果 | `tool_calls` 非空；对 `1+1=?` 通常调用 `add` |
| 适用 | 结构化流水线下一步**依赖**工具结果（如必须先 `query_order` 再生成回复） |

注意：示例里 **`extra_body={"thinking": {"type": "disabled"}}`**。DeepSeek 部分模型在 **thinking 模式** 下不支持 `required` 等约束，需显式关闭 thinking（见 §6）。

### 3.4 工具名字符串 — 强制指定工具

```python
model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="subtract")
rprint(model_with_tools.invoke(messages, extra_body={"thinking": {"type": "disabled"}}))
```

| 项 | 说明 |
|----|------|
| LangChain 转换 | 若字符串匹配已绑定工具名，转为 `{"type": "function", "function": {"name": "subtract"}}` |
| 典型结果 | 仅 `subtract` 的 `tool_calls`，例如 `args={"a": 2, "b": 1}`（模型可能「绕路」用减法凑答案） |
| 适用 | 固定工作流步骤、只允许单一副作用（如仅允许 `cancel_order` 不允许 `create_order`） |

OpenAI 原生写法等价于：

```python
tool_choice={"type": "function", "function": {"name": "subtract"}}
```

LangChain 允许直接写 **`tool_choice="subtract"`**，更简洁。

---

## 4. LangChain 支持的 tool_choice 完整对照

摘自 `BaseChatOpenAI.bind_tools` 文档，便于查阅：

| 取值 | 含义 |
|------|------|
| `None` / `False` | 不传约束，OpenAI 默认 |
| `"auto"` | 自动选择（含「不调用」） |
| `"none"` | 禁止调用任何工具 |
| `"required"` / `"any"` / `True` | 必须至少调用一个工具 |
| `"add"` 等工具名 | 强制调用名为 `add` 的工具 |
| `{"type": "function", "function": {"name": "xxx"}}` | 与工具名等价，OpenAI 原生 dict 形式 |

绑定多个工具时，`auto` 与 `required` 仍由模型在列表中选具体哪一个；只有传入**工具名**时才锁定单一函数。

---

## 5. DeepSeek thinking 模式注意事项

示例注释：

```python
# some tool_choice is not supported in thinking mode
model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="required")
rprint(model_with_tools.invoke(messages, extra_body={"thinking": {"type": "disabled"}}))
```

| 情况 | 建议 |
|------|------|
| `tool_choice="none"` / `"auto"` | 一般可直接 `invoke`，thinking 影响较小 |
| `tool_choice="required"` 或指定工具名 | 若 API 报错或不生效，在 `invoke` 时加 `extra_body={"thinking": {"type": "disabled"}}` |
| 生产环境 | 需要强制工具时，在模型配置层统一关闭 thinking，或换用非 thinking 模型 |

`extra_body` 会透传给底层 OpenAI 兼容客户端，与第32课 `ChatDeepSeek` 其它扩展参数用法一致。遇到「绑了 `required` 却没有 `tool_calls`」时，先查 thinking 是否开启。

---

## 6. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `@tool` | 定义 `add` / `subtract` | 模块顶部 |
| `bind_tools(tools, tool_choice=...)` | 绑定工具并指定调用策略 | `main()` 四次 |
| `invoke(messages)` | 发起一轮补全 | 每次 `rprint` 前 |
| `invoke(..., extra_body={...})` | 关闭 thinking 等扩展 | `required` / `subtract` |
| `AIMessage.tool_calls` | 查看是否触发、调用了谁 | 观察 `rprint` 输出 |
| `AIMessage.content` | 无工具时的文本回答 | `tool_choice="none"` 时 |

---

## 7. 与前面课程的关系

```
第28课 Function Calling     →  OpenAI tools、tool_choice 概念 ✅ 原理
第36课 @tool + bind_tools    →  工具定义与绑定
第37课 args_schema          →  参数 Schema 精确定义
第38课 tool_choice          →  控制「是否调 / 调哪个」✅ 你在这里
第29课 智能客服              →  生产默认 tool_choice="auto"
Agent / LCEL                →  不同节点可 bind 不同 tool_choice
```

本课补的是 **调用策略层**：Schema 定好「工具长什么样」（第37课），`tool_choice` 定好「这一轮模型能不能用、必须用哪个」（本课）。二者正交，可组合使用。

---

## 8. 常见问题

### Q1: 不传 `tool_choice` 和传 `"auto"` 一样吗？

在 LangChain + OpenAI 兼容实现中，语义上接近：模型自主决定是否调用。显式写 `"auto"` 便于读代码表达意图；不传则依赖 API 默认。

### Q2: `required` 时模型会调 add 还是 subtract？

由模型根据用户问题与 tool `description` 选择。`1+1=?` 通常选 `add`；若 `tool_choice="subtract"` 则只能选 `subtract`，可能出现「2-1=1」类绕路参数。

### Q3: 强制工具后还要自己执行 `tool.invoke` 吗？

要。`tool_choice` 只约束**模型输出**；第36课闭环不变：`tool_calls` → `add.invoke(...)` → `ToolMessage` → 再次 `invoke`。

### Q4: 能否一轮强制调两个工具？

`tool_choice` 约束的是「是否必须出现 tool_calls」，不保证 parallel 多次调用。需并行时另设 `parallel_tool_calls=True`（默认允许），且模型与 API 须支持 parallel tool calls。

### Q5: `tool_choice="none"` 时 tools 还会发给 API 吗？

会。tools 仍在请求里，只是 API 禁止模型在本轮生成 `tool_calls`。若希望减 token，应使用未 `bind_tools` 的纯 `ChatDeepSeek` 实例。

### Q6: 与 MCP / Agent 里工具选择的关系？

- **MCP**：远程工具列表，同样通过 `bind_tools` 挂载，`tool_choice` 规则相同  
- **ReAct Agent**：通常每步 `auto`；某些节点可换 `RunnableBinding` 固定 `required`  
- **Skills**：文档层引导「何时用何能力」；`tool_choice` 是 API 层硬约束，更强制

---

## 9. 动手练习

1. **四段对比**：按第32课配置环境后运行脚本，记录四次输出的 `content` 与 `tool_calls[0]["name"]`（若有）
2. **改问题**：把 `HumanMessage` 改成 `"10减3等于多少？"`，在 `auto` 下观察是否倾向 `subtract`
3. **指定 add**：设 `tool_choice="add"`，对比与 `subtract` 的 `args` 差异
4. **闭环**：对 `auto` 的一次响应，手动 `add.invoke(tool_call)` 并二次 `invoke`，拿到自然语言总结
5. **thinking 对比**：去掉 `extra_body`，看 `required` 是否报错或行为异常（视当前 DeepSeek 模型而定）
6. **接第29课**：在智能客服思路里设计：意图识别用 `none`，订单查询用 `required` + 指定 `query_order` 工具名

---

## 10. 参考

- 示例代码：`examples/llm-apps/17_langchain_tool_choices.py`
- 前置笔记：`notes/phase4-projects/16_LangChain_Tools工具.md`、`notes/phase4-projects/17_LangChain_Tool_Schema.md`
- Function Calling 原理：`notes/phase4-projects/07_Function_Calling与Tools使用.md`
- OpenAI tool_choice：[Chat Completions — Tool choice](https://platform.openai.com/docs/guides/function-calling)
- LangChain `bind_tools`：`langchain_openai.chat_models.base.BaseChatOpenAI.bind_tools`

---

*完成本课后，你已能在 LangChain 里用 `tool_choice` 精确控制模型是否调用工具、以及调用哪一个：从默认的 `auto` 到禁止调用的 `none`、必须调用的 `required`、以及指定工具名。结合第36–37课的工具定义与 Schema，工具链的控制粒度已覆盖「定义—参数—策略」三层。*
