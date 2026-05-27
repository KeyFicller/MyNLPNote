# 第26课：RAG 与 Agent 融合实战

**项目**: 带私有知识库的智能 Agent  
**技术栈**: ChromaDB + Sentence-Transformers + ReAct Agent + DeepSeek/OpenAI API  
**前置**: 第23课 RAG、第25课 Agent + 真实 API

---

## 1. 为什么要融合 RAG 和 Agent？

### 1.1 纯 RAG 的局限

第23课的 RAG 系统是**固定流程**：

```
用户问题 → 检索 → 拼 Prompt → LLM 回答
```

每次都会检索，无法：
- 判断是否需要查知识库（简单计算也要走 RAG）
- 组合多种能力（查资料 + 算数 + 搜新闻）
- 多步推理（先检索，再计算，再汇总）

### 1.2 纯 Agent 的局限

第25课的 Agent 有工具，但缺少**私有知识**：
- `search` 查的是互联网，不是你的文档
- 模型可能凭记忆回答课程问题，产生幻觉

### 1.3 融合方案

```
RAG 检索能力  →  封装为 knowledge_base 工具
Agent 规划能力 →  自主决定何时查库、何时计算、何时搜索
```

**本质**：RAG 提供「知识」，Agent 提供「调度」。

---

## 2. 架构设计

```
用户: "RAG有什么优势？再帮我算 2**10"
         ↓
    ReAct Agent (LLM)
         ↓
  第1轮 Action: knowledge_base(query="RAG优势")
  第1轮 Observation: [检索到的文档片段]
         ↓
  第2轮 Action: calculator(expression="2 ** 10")
  第2轮 Observation: 2 ** 10 = 1024
         ↓
  Final Answer: 综合工具结果回答
```

### 2.1 与 Skills 的关系

| 组件 | 角色 |
|------|------|
| `knowledge_base` 工具 | RAG 检索能力（查「是什么」） |
| `calculator` / `python` | 执行能力（做「算什么」） |
| `search` | 实时信息（查「最新新闻」） |
| Agent 系统提示词 | 调度规则（何时用哪个工具） |

---

## 3. 核心实现

### 3.1 KnowledgeBase 类

封装 ChromaDB，与第23课共用持久化目录 `.cache/chroma_db`：

```python
class KnowledgeBase:
    def setup(self) -> bool:
        # 加载已有向量库，或首次索引文档
        ...

    def search(self, query: str, top_k: int = 3) -> str:
        docs = self.vectorstore.similarity_search(query, k=top_k)
        # 格式化返回，带来源标题
```

### 3.2 注册 knowledge_base 工具

```python
tools["knowledge_base"] = Tool(
    name="knowledge_base",
    description="私有课程知识库检索...",
    parameters={"query": "检索关键词"},
    func=knowledge_base_tool,
)
```

### 3.3 扩展 Agent 提示词

通过 `extra_system_rules` 告诉 Agent 工具选择策略：

```
- Python/BERT/RAG 等课程问题 → knowledge_base（优先）
- 最新新闻 → search
- 数学计算 → calculator
```

---

## 4. RAG 工具 vs 独立 RAG 问答

| 维度 | 独立 RAG（第23课） | RAG 作为 Agent 工具 |
|------|-------------------|---------------------|
| 触发方式 | 每次必检索 | Agent 按需调用 |
| 灵活性 | 低 | 高，可组合多工具 |
| 复杂任务 | 难处理 | 支持多步 ReAct |
| 实现复杂度 | 较低 | 较高 |
| 适用场景 | 文档问答 | 智能助手、Copilot |

---

## 5. 运行方式

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key（终端或 .vscode/settings.json）
$env:DEEPSEEK_API_KEY = "your-key"

# 3. HuggingFace 镜像（脚本已自动配置 hf-mirror.com）
# 也可在终端手动设置：
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "600"

# 4. 运行（首次会下载 Embedding 模型并索引知识库）
python examples/llm-apps/05_rag_agent.py
```

### 测试用例

1. `Python列表有哪些常用方法？` → 应调用 `knowledge_base`
2. `RAG技术解决了哪些问题？` → 应调用 `knowledge_base`
3. `计算 2 ** 10` → 应调用 `calculator`

---

## 6. 常见问题

### Q1: 知识库为空或检索不到？

- 确认首次运行完成了索引（控制台显示「索引完成」）
- 检查 `.cache/chroma_db` 目录是否存在
- 尝试更具体的关键词

### Q2: Agent 不调用 knowledge_base？

- 检查系统提示词中的工具选择规则
- 在用户问题中明确「请查知识库」
- 第25课已强制「必须先调用工具再回答」

### Q3: 与第23课向量库冲突吗？

- 共用 `collection_name="knowledge_base"` 和同一 persist 目录
- 若第23课已索引过，第26课会直接加载，无需重复索引

### Q4: 频繁出现「未识别工具调用」？

**原因**：工具调用成功后，模型第二轮常直接回答，但未写 `Final Answer:`，解析失败后又被要求「输出 Action」，形成死循环。

**处理思路**：

| 层级 | 做法 |
|------|------|
| 提示词 | Observation 后明确「不要再调工具，只输出 Final Answer」 |
| 解析器 | 兼容 `最终答案:`、markdown 代码块中的 Action |
| 回合逻辑 | 已有 Observation 时，解析失败则引导 Final Answer 而非 Action |
| 工具集 | 本课只保留 5 个相关工具，减少干扰 |
| 终极方案 | 改用 API 原生 `tool_calls`（Function Calling），告别文本解析 |

代码已在 `agent_core.py` 中实现前几项优化。

---

## 7. 下一步学习

- **RAG 进阶**：混合检索、Rerank、评估指标
- **LangChain Agent 框架版**：用 `create_react_agent` 重构
- **Gradio 部署**：Web 界面聊天机器人
- **Function Calling**：用 API 原生 tool_calls 替代 Action 文本解析

---

**实践项目文件**: `examples/llm-apps/05_rag_agent.py`  
**共享模块**: `examples/llm-apps/agent_core.py`
