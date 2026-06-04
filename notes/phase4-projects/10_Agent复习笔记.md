# Agent 专题复习笔记

> 覆盖 MyNLPNote 第 24–29 课中与 Agent 相关的内容，用于考前/面试前快速回顾。  
> 对应课程：03 → 04 → 05 → 07 → 08（代码见 `examples/llm-apps/`）

---

## 1. 核心概念

**AI Agent** 是在大语言模型之上，具备**自主规划、工具调用、记忆管理**能力的智能体，公式：

```
AI Agent = LLM + Tools + Memory + Planning
```

与「单次问答」的区别：Agent 不会一次性给出答案，而是进入 **感知 → 思考 → 行动 → 观察** 的循环，直到任务完成。

**ReAct**（Reasoning + Acting）是最常用的 Agent 模式：模型先输出 `Thought`（推理），再输出 `Action`（选工具+参数），拿到 `Observation`（工具结果）后继续循环，最后输出 `Final Answer`。

```
用户输入
    ↓
Thought → Action → Observation → Thought → ... → Final Answer
         ↑__________________________|
              ReAct 循环（最多 N 轮）
```

**两种实现路径（本仓库都涉及）：**

| 路径 | 做法 | 代表课程 |
|------|------|---------|
| **Prompt 驱动 ReAct** | 系统提示词约束输出格式，正则解析 Action | 第 24–26 课 |
| **Function Calling** | API 返回结构化 `tool_calls`，框架执行后回传 | 第 28–29 课 |

本质相同：让 LLM 决定**何时、用什么工具、传什么参数**；差异在协议层（手写解析 vs API 原生支持）。

---

## 2. 关键术语

| 术语 | 含义 | 记忆要点 |
|------|------|---------|
| **ReAct** | Reasoning + Acting，推理与行动交替 | 论文 2022，Thought/Action/Observation 三段式 |
| **Tool / 工具** | Agent 可调用的外部函数 | `name` + `description` + `parameters` + `func` |
| **Thought** | 模型对当前状态的推理 | 决定下一步行动的依据 |
| **Action** | 工具名 + 参数 | 如 `search(query="2024 NLP 进展")` |
| **Observation** | 工具执行返回的结果 | 作为下一轮 Thought 的输入 |
| **Final Answer** | 循环结束时的最终回复 | 整合所有 Observation |
| **Memory** | 对话与任务上下文 | 短期（当前对话）、工作（中间结果）、长期（用户偏好） |
| **System Prompt** | 系统提示词 | 定义角色、工具列表、输出格式、调度规则 |
| **max_iterations** | 最大 ReAct 轮数 | 防止无限循环，通常 3–10 |
| **Function Calling** | LLM API 原生工具调用 | 返回 JSON Schema 结构化请求 |
| **knowledge_base** | RAG 封装成的 Agent 工具 | 查私有文档，而非互联网 |
| **MCP** | Model Context Protocol | Function Calling 的上层标准，工具可跨客户端复用 |

---

## 3. 与前后知识的联系

```
第 22–23 课 RAG          第 24 课 Agent 基础        第 25 课 真实 API
检索增强生成      →      ReAct + 模拟 LLM     →     GPT/DeepSeek 接入
                              ↓
                    第 26 课 RAG + Agent 融合
                    knowledge_base 作为工具
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
        第 27 课 Gradio   第 28 课 FC      第 29 课 智能客服
        Web 部署          API 原生工具      RAG+Agent+FC 综合
                              ↓
                        第 30 课 MCP
                        工具标准化暴露
```

**递进关系：**

1. **RAG → Agent**：RAG 是固定「检索→生成」；Agent 把 RAG **降级为可选工具**，按需调用。
2. **模拟 LLM → 真实 API**：规则匹配无法理解复杂意图；真实 LLM 能优化搜索词、组合多工具。
3. **Prompt ReAct → Function Calling**：手写解析易碎；FC 由 API 保证结构，更适合生产。
4. **Agent → MCP**：单个应用内工具 → 跨 Cursor/Claude 等平台复用的标准 Server。

---

## 4. 架构速记

### 4.1 ReAct Agent 核心组件

```
┌─────────────────────────────────────────┐
│              ReActAgent                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │   LLM   │  │  Tools  │  │ Memory  │ │
│  │ Client  │  │  注册表  │  │ 对话历史 │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       └────────────┼────────────┘       │
│                    ↓                    │
│            run(user_input)              │
│         while iteration < max:          │
│           LLM → parse Action            │
│           execute Tool → Observation    │
│           append to messages            │
└─────────────────────────────────────────┘
```

### 4.2 本仓库常用工具矩阵

| 工具 | 能力 | 典型场景 |
|------|------|---------|
| `search` | 互联网搜索（DDGS） | 新闻、时效信息 |
| `calculator` | 精确计算 | 数学、单位换算 |
| `knowledge_base` | ChromaDB 私有检索 | 课程/文档问答 |
| `python` | 代码执行 | 复杂逻辑、数据处理 |
| `get_weather` 等 | 模拟/真实 API | Function Calling 演示 |

### 4.3 工具选择策略（系统提示词）

```
课程/文档问题     → knowledge_base（优先）
最新新闻/事实     → search
数学计算          → calculator
订单/业务 API     → Function Calling 专用工具（第 29 课）
```

---

## 5. 代码锚点（必看文件）

| 文件 | 内容 |
|------|------|
| `examples/llm-apps/03_ai_agent.py` | ReAct 基础、模拟 LLM、工具注册 |
| `examples/llm-apps/agent_core.py` | **共享核心**：LLMClient、Tool、ReActAgent、KnowledgeBase |
| `examples/llm-apps/04_ai_agent_with_api.py` | 多平台 API 接入、成本估算 |
| `examples/llm-apps/05_rag_agent.py` | RAG 封装为 `knowledge_base` 工具 |
| `examples/llm-apps/06_gradio_chatbot.py` | Agent + RAG 的 Web 部署 |
| `examples/llm-apps/07_function_calling.py` | FC Schema、API 调用链 |
| `examples/llm-apps/08_intelligent_customer_service.py` | 规则系统 + DeepSeek FC 双架构 |
| `toy/01_whos_undercover.py` | 多 Agent 游戏、提示词与状态管理 |
| `toy/02_liars_bar.py` | FC + 多轮博弈 |

---

## 6. 易错点与最佳实践

### 6.1 常见坑

| 问题 | 原因 | 对策 |
|------|------|------|
| Agent 死循环 | 无 Final Answer、max_iterations 过大 | 限制轮数；提示词要求明确终止条件 |
| 选错工具 | `description` 不清晰 | 写清「何时用、何时不用」 |
| 解析 Action 失败 | LLM 输出格式不稳定 | 用 FC API；或加强 few-shot 示例 |
| 幻觉回答 | 未调用 knowledge_base | 系统规则：文档问题必须先检索 |
| API 成本高 | 每轮都带完整历史 | 截断 Memory；压缩 Observation |
| 工具报错未处理 | 异常直接抛给 LLM | 返回 `Observation: 错误信息，请换策略` |

### 6.2 设计原则

1. **工具粒度**：一个工具做一件事，`description` 比 `name` 更重要。
2. **提示词分层**：角色定义 + 工具说明 + 输出格式 + 调度规则（`extra_system_rules`）。
3. **可观测性**：打印每轮 Thought/Action/Observation，便于调试。
4. **渐进复杂度**：先模拟 LLM 跑通循环 → 接 API → 加 RAG → 换 Function Calling。

---

## 7. 自测题

### 选择题

**1. ReAct 模式中，Observation 的作用是？**

- A. 存储用户长期偏好  
- B. 工具执行结果，供下一轮推理使用  
- C. 系统提示词的一部分  
- D. 向量检索的相似度分数  

<details><summary>答案</summary>

**B**。Observation 是 Action 执行后的返回值，会被追加进上下文，驱动下一轮 Thought。

</details>

**2. 纯 RAG 与「RAG 作为 Agent 工具」的主要区别是？**

- A. 向量库不同  
- B. Agent 版每次必检索，纯 RAG 按需检索  
- C. Agent 版由模型决定是否检索，纯 RAG 固定检索  
- D. 纯 RAG 不能用的 Embedding  

<details><summary>答案</summary>

**C**。独立 RAG 每次都走检索→生成；融合版把检索封装成 `knowledge_base`，由 Agent 按需调用。

</details>

**3. Function Calling 相比手写 ReAct 解析的主要优势是？**

- A. 不需要 LLM  
- B. API 返回结构化 tool_calls，减少格式解析错误  
- C. 只能用一个工具  
- D. 不能做多轮对话  

<details><summary>答案</summary>

**B**。FC 由模型 API 输出标准 JSON 结构，比正则解析 `Action:` 行更稳定。

</details>

### 简答题

**4. 简述 Agent 执行「北京今天天气怎样，顺便算 15×23」的 ReAct 过程（至少 2 轮 Action）。**

<details><summary>参考答案</summary>

```
Thought: 需要查北京天气并做乘法，分两步
Action: get_weather(location="北京")
Observation: 北京晴，25°C

Thought: 天气已获取，还需计算 15×23
Action: calculator(expression="15 * 23")
Observation: 345

Final Answer: 北京今天晴，25°C；15×23=345。
```

</details>

**5. 说明 Memory 中短期记忆、工作记忆、长期记忆各存什么？**

<details><summary>参考答案</summary>

- **短期记忆**：当前多轮对话的 user/assistant 消息，新会话可清空。  
- **工作记忆**：当前任务 ReAct 链中的 Thought/Action/Observation 及工具中间结果。  
- **长期记忆**：用户偏好、历史关键信息，需显式持久化（数据库/文件）。

</details>

---

## 8. 面试 / 项目话术

介绍 Agent 项目时可按此结构：

1. **问题**：单次 LLM 无法查实时数据、私有文档，数学不准。  
2. **方案**：ReAct Agent + 工具集（search / calculator / knowledge_base）。  
3. **实现**：`agent_core.py` 统一 LLM 客户端；ChromaDB 做 RAG 工具；DeepSeek FC 做业务 API。  
4. **亮点**：RAG 与 Agent 融合、双系统客服（规则 + LLM）、toy 多 Agent 博弈。  
5. **可改进**：MCP 标准化工具、LangSmith 追踪、工具并行调用。

---

## 9. 复习路线建议

```
Day 1  读 03 笔记 + 跑 03_ai_agent.py          → 理解 ReAct 循环
Day 2  读 04 笔记 + 跑 04_ai_agent_with_api.py  → API 与 Prompt
Day 3  读 05 笔记 + 跑 05_rag_agent.py          → RAG 工具化
Day 4  读 07 笔记 + 跑 07_function_calling.py   → FC 与 ReAct 对比
Day 5  读 08 笔记 + 跑 08_intelligent_customer_service.py → 综合实战
Day 6  玩 toy/02_liars_bar.py                   → FC + 多 Agent 直觉
Day 7  闭卷做本文第 7 节自测题                   → 检验掌握度
```

---

## 10. 速查命令

```bash
# Agent 基础（模拟 LLM）
python examples/llm-apps/03_ai_agent.py

# 接入 DeepSeek / OpenAI（需配置 API Key）
python examples/llm-apps/04_ai_agent_with_api.py

# RAG + Agent 融合
python examples/llm-apps/05_rag_agent.py

# Function Calling
python examples/llm-apps/07_function_calling.py

# 智能客服综合实战
python examples/llm-apps/08_intelligent_customer_service.py
```

环境变量示例：`DEEPSEEK_API_KEY=sk-xxx`

---

## 总结

Agent 学习的核心是 **ReAct 循环 + 工具调度 + 记忆/context 管理**。本仓库从模拟到真实 API，从单工具到 RAG 融合，再到 Function Calling 和 MCP，形成完整链路。复习时抓住一条主线：**LLM 负责想，工具负责做，Memory 负责记住上下文**。

---

**相关笔记：**
- `03_AI_Agent开发实战.md`
- `04_AI_Agent接入真实LLM_API.md`
- `05_RAG与Agent融合实战.md`
- `07_Function_Calling与Tools使用.md`
- `08_智能客服系统综合实战.md`

**生成说明：** 基于 MyNLPNote 仓库 Agent 相关课程笔记整理，可与 MCP `study_review` 提示词配合使用。
