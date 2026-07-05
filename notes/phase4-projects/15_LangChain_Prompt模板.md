# 第35课：LangChain Prompt 模板

**项目**: ChatPromptTemplate 与多轮 Prompt 组装  
**技术栈**: LangChain, langchain-core, langchain-deepseek, ChatDeepSeek  
**示例代码**: `examples/llm-apps/14_langchain_prompt.py`  
**前置课程**: 第34课 消息历史与多轮对话、第19课 LangChain 基础（Prompt Template 概念）  
**环境与运行**：见 [第32课 §1 环境配置](12_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/llm-apps/14_langchain_prompt.py`（`main()` 中注释切换演示函数）

---

## 课程概述

第19课介绍了面向**文本补全**的 `PromptTemplate`；第34课用字典 `messages` 列表手写多轮对话。本课聚焦 **Chat Model 专用** 的 `ChatPromptTemplate`：如何用占位符参数化 system / user 消息、如何预填部分变量、如何把历史对话「插槽」进模板，以及如何用 `+` 拼接多个模板。

示例延续「杠精 / 教授」等人设，便于直观感受 **Prompt 结构变化** 对模型输出的影响。

**学习目标：**
1. 掌握 `ChatPromptTemplate.from_messages` 的元组写法与 Message 类写法
2. 理解 `invoke` / `format` / `format_messages` 三种渲染方式的区别
3. 会用 `partial()` 预填变量，复用同一模板的不同「场景版」
4. 会用 `MessagesPlaceholder` 与 `("placeholder", ...)` 注入多轮历史
5. 了解 `ChatPromptTemplate + ChatPromptTemplate` 组合模板

---

## 1. 为什么需要 ChatPromptTemplate？

### 1.1 手写 messages 的问题

```python
# 每次换人设都要改字符串，容易漏改、难维护
messages = [
    {"role": "system", "content": "你是一个杠精，喜欢和用户抬杠。"},
    {"role": "user", "content": user_input},
]
```

第34课在循环里 `append` 历史，适合**运行时动态累积**；但很多场景需要在**编译期**就固定 Prompt 骨架，只替换少量变量（部门、角色、用户输入等）。`ChatPromptTemplate` 就是 LangChain 提供的**可复用、可组合**的 Chat Prompt 构建器。

### 1.2 PromptTemplate vs ChatPromptTemplate

| 类型 | 面向 | 输出 | 典型场景 |
|------|------|------|----------|
| `PromptTemplate` | Completion LLM | 单个字符串 | 第19课 RAG 问答拼接 |
| `ChatPromptTemplate` | Chat Model | `ChatPromptValue` / Message 列表 | 本课：system + user 结构化对话 |

`ChatDeepSeek.invoke()` 两种都能接：模板渲染后得到 Message 列表，与第34课手写的字典 messages **等价**。

---

## 2. 模板格式化 — `_test_chat_prompt_format`

### 2.1 定义模板

```python
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个{role}，喜欢和用户{behavior}。"),
        ("user", "{input}"),
    ]
)
```

元组格式 `("role名", "内容模板")` 是 LangChain 推荐的简洁写法。`{role}`、`{behavior}`、`{input}` 为**输入变量**，调用时再填入。

### 2.2 三种渲染方式

| 方法 | 返回类型 | 能否直接 `llm.invoke()` | 说明 |
|------|----------|-------------------------|------|
| `invoke({...})` | `ChatPromptValue` | ✅ | 推荐；与 LCEL Chain 一致 |
| `format(...)` | `str` | ❌（需再解析） | 整段 prompt 拼成字符串 |
| `format_messages(...)` | `list[BaseMessage]` | ✅ | 直接得到 Message 对象列表 |

```python
# 方式 1：invoke → ChatPromptValue
prompt = prompt_template.invoke({
    "role": "杠精",
    "behavior": "抬杠",
    "input": "你好，你是谁？",
})
response = _chat_deepseek().invoke(prompt)

# 方式 2：format → 字符串（本示例仍传给 invoke，LangChain 会做转换）
prompt = prompt_template.format(
    role="马屁精",
    behavior="溜须拍马",
    input="你好，你是谁？",
)

# 方式 3：format_messages → [SystemMessage, HumanMessage, ...]
prompt = prompt_template.format_messages(
    role="社恐",
    behavior="社恐",
    input="你好，你是谁？",
)
```

**实践建议**：在 Chain 或需要类型安全的场景用 `invoke`；调试时可 `print(format_messages(...))` 查看最终 Message 结构。

### 2.3 人设即变量

同一模板，换三组变量 → 三种完全不同的回复风格（杠精 / 马屁精 / 社恐）。这说明 **system 内容参数化** 是快速切换 Agent 人设的常用手段，智能客服、多角色助手都基于此模式。

---

## 3. 多种初始化方式 — `_test_chat_prompt_initialization`

LangChain 允许用不同 API 表达**同一件事**，理解等价关系有助于读官方文档和第三方示例。

### 3.1 元组写法（最常用）

```python
ChatPromptTemplate.from_messages([
    ("system", "你是一个杠精，喜欢和用户抬杠。"),
    ("user", "你好，{user_input}"),
])
```

### 3.2 MessagePromptTemplate 写法

```python
ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是一个杠精，喜欢和用户抬杠。"),
    HumanMessagePromptTemplate.from_template("你好，{user_input}"),
])
```

与元组写法**完全等价**，适合需要单独复用某一类 Message 模板的场景。

### 3.3 嵌套 from_messages（固定内容、无变量）

```python
ChatPromptTemplate.from_messages([
    ChatPromptTemplate.from_messages(("system", "你是一个杠精，喜欢和用户抬杠。")),
    ChatPromptTemplate.from_messages(("user", "你好，你是谁？")),
])
prompt = prompt_template.invoke({})  # 无输入变量
```

内层每个 `from_messages` 相当于一条固定消息；外层再包一层。无占位符时 `invoke({})` 即可。这种写法较少见，多见于动态组装 Prompt 的中间步骤。

---

## 4. 部分预填 — `_test_chat_prompt_partial`

当多个场景共享同一骨架、只有部分变量不同时，可用 `partial()` **提前绑定**部分变量，得到「子模板」。

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是{department} 部门的{role}。"),
    ("user", "{user_input}"),
])

IT_department = prompt_template.partial(department="IT", role="工程师")
SALING_department = prompt_template.partial(department="销售", role="销售员")

prompt = IT_department.invoke({"user_input": "我的鼠标为什么坏了？"})
```

```
原始模板变量:  department, role, user_input
                    ↓ partial
IT 子模板变量:  user_input          （department、role 已固定）
销售子模板变量: user_input
```

同一用户问题「我的鼠标为什么坏了？」，IT 工程师与销售员的回答角度会不同——**部门上下文在 system 层注入**，无需每次 invoke 重复传参。

| 场景 | partial 预填示例 |
|------|------------------|
| 多部门客服 | `department`, `role` |
| 多语言 | `language="中文"` |
| 固定安全策略 | `safety_rules="..."` |

---

## 5. 历史消息占位 — `_test_chat_prompt_placeholder`

多轮对话需要把**已有对话**插入 Prompt。第34课在 Python 列表里 `append`；本课用模板占位符，更适合与 Chain、Memory 组件对接。

### 5.1 元组 placeholder

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一名教授。"),
    ("placeholder", "{conversation}"),
])

prompt = prompt_template.invoke({
    "conversation": [
        ("user", "你好，你是谁？"),
        ("assistant", "我是主攻土木工程的教授。"),
        ("user", "我没听清楚，你主攻哪个方向来着？"),
    ]
})
```

`("placeholder", "{conversation}")` 表示：`conversation` 传入的内容会**展开**为多条 Message，而不是一条字符串。

### 5.2 MessagesPlaceholder（推荐）

```python
from langchain_core.prompts import MessagesPlaceholder

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一名教授。"),
    MessagesPlaceholder(variable_name="conversation"),
])

prompt = prompt_template.invoke({
    "conversation": [
        HumanMessage(content="你好，你是谁？"),
        AIMessage(content="我是主攻土木工程的教授。"),
        HumanMessage(content="我没听清楚，你主攻哪个方向来着？"),
    ]
})
```

与元组写法等价；显式类名可读性更好。`conversation` 可传：

- 元组列表：`("user", "...")` / `("assistant", "...")`
- Message 对象：`HumanMessage` / `AIMessage`

### 5.3 模板拼接 `+`

```python
prompt_template1 = ChatPromptTemplate.from_messages([
    ("system", "你是一名教授。"),
])
prompt_template2 = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="conversation"),
])
prompt_template = prompt_template1 + prompt_template2
```

```
prompt_template1          prompt_template2
  [system]        +         [placeholder: conversation]
              ↓
合并后:
  [system] [HumanMessage] [AIMessage] [HumanMessage] ...
```

**适用场景**：system 固定、历史动态——与第34课「system 始终保留 + 截断对话」的设计一致；这里把「历史槽位」写进模板，便于接入 `RunnableWithMessageHistory` 等封装。

### 5.4 与第34课的关系

```
第34课  messages 列表 append + 截断     →  命令行聊天循环，裸数据结构
第35课  MessagesPlaceholder + 模板      →  声明式 Prompt，便于 Chain / Memory
```

两者可结合：循环里维护 `conversation` 列表，每轮 `prompt_template.invoke({"conversation": history})`。

---

## 6. 整体数据流

```
┌──────────────────────────────────────────────────────────┐
│              ChatPromptTemplate.from_messages             │
│   ("system", "...{var}...")  +  ("user", "...")          │
│   +  MessagesPlaceholder("conversation")  （可选）        │
└─────────────────────────┬────────────────────────────────┘
                          │ invoke / format_messages
                          ↓
              ChatPromptValue 或 list[BaseMessage]
                          │
                          ↓
                   ChatDeepSeek.invoke()
                          │
                          ↓
                    AIMessage.content
```

可选分支：

```
partial(department="IT")  →  减少 invoke 时传入的变量
template_a + template_b   →  模块化拼接 system 与 history 段
```

---

## 7. 与前面课程的关系

```
第19课 PromptTemplate        →  文本补全模板、Few-shot 概念
第32课 invoke 输入格式       →  字符串 / 字典 / Message 对象
第34课 messages 多轮历史     →  append + 截断，手写列表 ✅ 基础
第35课 ChatPromptTemplate    →  参数化 system/user + 历史占位 ✅ 你在这里
第29课 智能客服              →  生产级 system + Memory + Tools 组合
第27课 Gradio 部署           →  UI 层消费同一套 Prompt 逻辑
```

本课补齐 **Prompt 层**：搞清模板如何渲染成 Message 后，再学 `RunnableWithMessageHistory`、Agent 的 `prompt | llm` 链会更顺畅。

---

## 8. 常见问题

### Q1: `invoke` 和 `format_messages` 选哪个？

功能上等价于「得到 Message 列表再调 LLM」。在 LCEL 里用 `prompt | llm` 时必须 `invoke`；单独调试时 `format_messages` 更直观。

### Q2: `format` 返回字符串还能 invoke 吗？

可以，LangChain 会尝试解析；但 Chat 场景更推荐 `invoke` 或 `format_messages`，避免丢失 system / user 边界。

### Q3: placeholder 里能放字典 messages 吗？

可以传元组 `("user", "content")` 或 `HumanMessage` / `AIMessage`；与 OpenAI 风格 `{"role": "user", "content": "..."}` 在 Chain 入口通常也可自动转换，但本示例统一用 LangChain Message 类型。

### Q4: partial 和 invoke 时传同一个变量冲突吗？

`partial` 绑定的变量在 `invoke` 时**不可再传**；未绑定的变量仍需传入。若重复传会报错。

### Q5: 模板 `+` 拼接有顺序要求吗？

有。`template_a + template_b` 按顺序合并消息列表，一般 `system` 放前，`MessagesPlaceholder` 放后。

### Q6: 与第19课 `LLMChain(llm, prompt)` 的关系？

旧版 `LLMChain` + `PromptTemplate` 面向 Completion；现代写法是 `ChatPromptTemplate | ChatModel`（LCEL）。本课只演示模板本身，下一课可接 `|` 组成 Chain。

---

## 9. 动手练习

1. **切换演示**：依次取消注释四个 `_test_*` 函数，观察不同 API 的输出差异
2. **新人设**：在 `_test_chat_prompt_format` 增加第四组变量（如「翻译官」），对比回复
3. **partial 扩展**：为「法务部 / 人事部」各建一个 `partial` 子模板，问同一政策问题
4. **接第34课**：写一个小循环，用 `MessagesPlaceholder` 模板 + 用户 `input()`，每轮把完整 `conversation` 传入（可先不做截断）
5. **调试渲染**：对任意模板 `print(prompt_template.invoke({...}).to_messages())`，对照 API 实际发送内容

---

## 10. 参考

- 示例代码：`examples/llm-apps/14_langchain_prompt.py`
- 前置笔记：`notes/phase4-projects/14_消息历史与多轮对话.md`
- Prompt 概念：`notes/phase4-projects/01_LangChain基础与RAG.md`（§3 Prompt Template）
- Chat Model 调用：`notes/phase4-projects/12_LangChain进阶与DeepSeek接入.md`（§4 Invoke 输入格式）
- LangChain Prompts：[Prompt Templates 文档](https://python.langchain.com/docs/concepts/prompt_templates/)

---

*完成本课后，你已掌握 Chat 场景下的 Prompt 工程核心工具：参数化人设、预填场景变量、占位注入多轮历史，以及模板的模块化拼接。这是构建可维护 LLM 应用与 Agent 的基础层。*
