# MyNLPNote — 生成式 AI 学习仓库

个人学习记录与练习代码，从 Python 基础到深度学习、大语言模型应用，按阶段整理示例与笔记。

## 快速开始

```bash
# 激活虚拟环境
source activate_env.sh

# 运行第一个 Python 示例
python examples/python-basics/01_hello_python.py

# 退出虚拟环境
deactivate
```

环境变量（大模型 API 等）可在 `.vscode/settings.json` 或本机环境中配置。

---

## 仓库结构

| 目录 | 说明 |
|------|------|
| `examples/python-basics/` | Python 基础练习 |
| `examples/numpy-pandas/` | NumPy、Pandas 数据处理 |
| `examples/pytorch/` | PyTorch 入门与实战 |
| `examples/nlp/` | NLP 与预训练模型 |
| `examples/llm-apps/` | LangChain、RAG、Agent、Function Calling 等 |
| `toy/` | 趣味小项目（AI 对战游戏等） |
| `notes/` | 分阶段学习笔记 |

---

## 学习路线

### 阶段一：编程基础

#### Python 基础

| 课程 | 内容 | 文件 |
|------|------|------|
| 第1课 | 变量、print、字符串 | `01_hello_python.py` |
| 第2课 | 数据类型与运算符 | `02_data_types_and_operators.py` |
| 第3课 | 列表和字典 | `03_list_and_dict.py` |
| 第4课 | 条件判断与循环 | `04_control_flow.py` |
| 第5课 | 函数定义 | `05_functions.py` |
| 第6课 | 模块和包 | `06_modules.py` |

#### 数据科学基础

| 课程 | 内容 | 文件 |
|------|------|------|
| 第7课 | NumPy 基础 | `01_numpy_basics.py` |
| 第8课 | NumPy 进阶 | `02_numpy_advanced.py` |
| 第9课 | Pandas 基础 | `03_pandas_basics.py` |
| 第10课 | Pandas 进阶 | `04_pandas_advanced.py` |

---

### 阶段二：深度学习

#### 理论与 PyTorch

| 课程 | 内容 | 文件 |
|------|------|------|
| 第11课 | PyTorch 基础 | `01_pytorch_basics.py` |
| 第12课 | 神经网络构建 | `02_neural_network.py` |
| 第13课 | 卷积神经网络 | `03_cnn.py` |
| 第14课 | DataLoader | `04_dataloader.py` |

---

### 阶段三：NLP 与大模型

| 课程 | 内容 | 文件 |
|------|------|------|
| 第15课 | 文本预处理与词嵌入 | `01_text_preprocessing.py` |
| 第16课 | Transformer | `02_transformer.py` |
| 第17课 | BERT 与 GPT | `03_bert_gpt.py` |
| 第18课 | HuggingFace 实战 | `04_huggingface_transformers.py` |
| 第19课 | 中文文本分类 | `06_chinese_text_classification.py` |
| 第20课 | 中文星级评价分类 | `07_chinese_text_rank.py` |
| 第21课 | 中文命名实体识别 | `08_chinese_ner.py` |
| 第22课 | LangChain 与 RAG 入门 | `01_langchain_basics.py` |
| 第23课 | 完整 RAG 项目 | `02_complete_rag_project.py` |
| 第24课 | AI Agent（ReAct） | `03_ai_agent.py` |
| 第25课 | Agent 接入真实 API | `04_ai_agent_with_api.py` |
| 第26课 | RAG 与 Agent 融合 | `05_rag_agent.py` |
| 第27课 | Gradio 聊天机器人 | `06_gradio_chatbot.py` |
| 第28课 | Function Calling | `07_function_calling.py` |
| 第29课 | 智能客服综合实战 | `08_intelligent_customer_service.py` |
| 第30课 | MCP 自定义 Server | `09_mcp_server.py` |
| 第31课 | Agent Skills 编写 | `10_agent_skills.py` |
| 第32课 | LangChain 进阶（DeepSeek 接入） | `11_langchain_advanced.py` |
| 第33课 | LangSmith 追踪与可观测性 | `12_langsmith.py` |
| 第34课 | 消息历史与多轮对话 | `13_messages_history.py` |
| 第35课 | LangChain Prompt 模板 | `14_langchain_prompt.py` |

---

### 趣味项目（toy）

使用 DeepSeek API 的多智能体小游戏，用于练习提示词、Function Calling 与游戏状态管理。

| 项目 | 说明 | 运行 |
|------|------|------|
| `01_whos_undercover.py` | 谁是卧底（可配置人数与卧底数） | `python toy/01_whos_undercover.py` |
| `02_liars_bar.py` | 骗子酒馆 | `python toy/02_liars_bar.py` |

可选参数示例（谁是卧底）：

```bash
python toy/01_whos_undercover.py --players 8 --undercovers 3 --max-rounds 20
```

运行日志默认写入 `toy/logs/`。

---

## 学习笔记

- `notes/phase1-python/` — 编程基础
- `notes/phase2-dl/` — 深度学习
- `notes/phase3-nlp/` — NLP 与大模型
- `notes/phase4-projects/` — 项目实战笔记

---

## 学习建议

1. **动手写代码**：示例尽量自己跑一遍、改一改
2. **记笔记**：用 `notes/` 按自己的理解总结
3. **循序渐进**：基础部分不必一次学完，按需跳转
4. **小项目练手**：`toy/` 适合练 LLM 调用与 Agent 逻辑

---

## 进度概览

| 阶段 | 状态 |
|------|------|
| 编程基础（Python + NumPy/Pandas） | 已完成 |
| 深度学习（PyTorch） | 已完成 |
| NLP 与预训练模型 | 已完成 |
| LLM 应用（RAG / Agent / Function Calling / MCP / Skills） | 已完成 |
| LangChain 进阶（DeepSeek / LangSmith / 多轮对话 / Prompt） | 进行中 |
| 趣味 AI 游戏（toy） | 进行中 |

---

*最后更新：2026-07-04（第35课 LangChain Prompt 模板）*
