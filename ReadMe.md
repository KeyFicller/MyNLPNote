# 零基础转行生成式AI - 学习记录

## 项目概要
这是一个零基础小白转行生成式AI的学习记录，涵盖从Python基础到深度学习、大语言模型的完整学习路径。

## 快速开始

```bash
# 1. 激活虚拟环境
source activate_env.sh

# 2. 运行第一个Python程序
python examples/python-basics/01_hello_python.py

# 3. 退出虚拟环境
deactivate
```

---

## 学习计划和路线

### 阶段一：编程基础（4-6周）

#### 1.1 Python基础（2周） ✅ 已完成

| 课程 | 内容 | 文件 |
|------|------|------|
| 第1课 | Python初体验 - 变量、print、字符串 | `01_hello_python.py` |
| 第2课 | 数据类型与运算符 - int/float/str/bool, 各类运算符 | `02_data_types_and_operators.py` |
| 第3课 | 列表和字典 - List/Dict/Tuple/Set, 遍历技巧 | `03_list_and_dict.py` |
| 第4课 | 条件判断与循环 - if/for/while, break/continue, 推导式 | `04_control_flow.py` |
| 第5课 | 函数定义 - def, 参数类型, lambda, 作用域 | `05_functions.py` |
| 第6课 | 模块和包 - import, 标准库, 自定义模块 | `06_modules.py` |

- [x] Python环境搭建与基础语法
- [x] 数据类型：字符串、列表、字典、元组
- [x] 流程控制：if/else、for、while
- [x] 函数定义与调用
- [x] 模块和包的使用
- [ ] 文件读写操作（数据科学阶段学习）
- [ ] 面向对象编程基础（进阶阶段学习）

**推荐资源：**
- 《Python编程：从入门到实践》
- [Python官方教程](https://docs.python.org/zh-cn/3/tutorial/)

#### 1.2 数据科学基础（2周） ✅ 已完成

| 课程 | 内容 | 文件 |
|------|------|------|
| 第7课 | NumPy基础 - 数组创建、索引、广播 | `01_numpy_basics.py` |
| 第8课 | NumPy进阶 - 线性代数、随机数、文件IO | `02_numpy_advanced.py` |
| 第9课 | Pandas基础 - DataFrame、Series、数据选择 | `03_pandas_basics.py` |
| 第10课 | Pandas进阶 - 数据清洗、合并、分组 | `04_pandas_advanced.py` |

- [x] NumPy数组操作
- [x] Pandas数据处理
- [ ] ~~Matplotlib/Seaborn数据可视化~~（跳过，专注生成式AI）
- [ ] ~~基础统计学概念~~（后续遇到再学）

**推荐资源：**
- [Kaggle Learn](https://www.kaggle.com/learn)

---

### 阶段二：深度学习基础（6-8周）

#### 2.1 深度学习理论（3周）
- [ ] 神经网络基础
- [ ] 反向传播算法
- [ ] 激活函数与损失函数
- [ ] 优化器：SGD、Adam
- [ ] 正则化与Dropout
- [ ] 卷积神经网络（CNN）
- [ ] 循环神经网络（RNN/LSTM）

**推荐资源：**
- 吴恩达《深度学习》专项课程
- 《深度学习》（花书）

#### 2.2 PyTorch实战（3-5周）
- [x] PyTorch张量操作
- [x] 自动求导（Autograd）
- [x] 构建神经网络模型
- [ ] 数据加载与预处理
- [ ] 模型训练与评估
- [ ] GPU加速训练

**实践项目：**
- [ ] 手写数字识别（MNIST）
- [ ] 图像分类（CIFAR-10）
- [ ] 情感分析

**推荐资源：**
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- 《动手学深度学习》

---

### 阶段三：自然语言处理与大模型（8-10周）

#### 3.1 NLP基础（3周）
- [ ] 文本预处理：分词、清洗
- [ ] 词嵌入：Word2Vec、GloVe
- [ ] Transformer架构详解
  - [ ] 注意力机制
  - [ ] Self-Attention
  - [ ] 多头注意力
  - [ ] 位置编码
  - [ ] 编码器-解码器结构

**推荐资源：**
- 《Attention Is All You Need》论文
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)

#### 3.2 预训练语言模型（3周）
- [ ] BERT及其变体
- [ ] GPT系列模型演进
- [ ] Hugging Face Transformers库使用
- [ ] 模型微调（Fine-tuning）
- [ ] 文本分类、命名实体识别实践

**实践项目：**
- [ ] 中文文本分类
- [ ] 命名实体识别（NER）
- [ ] 文本摘要

#### 3.3 大语言模型（LLM）进阶（2-4周）
- [ ] 大模型原理与架构
- [ ] 提示工程（Prompt Engineering）
- [ ] 检索增强生成（RAG）
- [ ] 模型量化与推理优化
- [ ] LangChain/LlamaIndex框架
- [ ] 本地部署开源大模型

**实践项目：**
- [ ] 构建个人知识库助手
- [ ] 搭建聊天机器人
- [ ] 开发AI写作助手

**推荐资源：**
- [LangChain文档](https://python.langchain.com/)
- [LlamaIndex文档](https://docs.llamaindex.ai/)

---

### 阶段四：实战与作品集（持续进行）

#### 4.1 综合项目
- [ ] 完整RAG应用开发
- [ ] AI Agent开发
- [ ] 多模态应用探索

#### 4.2 求职准备
- [ ] 整理GitHub项目
- [ ] 撰写技术博客
- [ ] 刷题：LeetCode算法
- [ ] 面试准备：八股文、系统设计

---

## 示例代码和框架

| 项目 | 描述 | 技术栈 |
|------|------|--------|
| `examples/python-basics/` | Python基础练习 | Python |
| `examples/numpy-pandas/` | 数据处理示例 | NumPy, Pandas |
| `examples/pytorch/` | PyTorch入门到进阶 | PyTorch |
| `examples/nlp/` | NLP基础任务 | Transformers |
| `examples/llm-apps/` | 大模型应用开发 | LangChain, LlamaIndex |

---

## 学习笔记

### 笔记分类
- `notes/phase1-python/` - 编程基础笔记
- `notes/phase2-dl/` - 深度学习笔记
- `notes/phase3-nlp/` - NLP与大模型笔记
- `notes/phase4-projects/` - 项目实战经验

---

## 学习建议

1. **每天坚持**：每天至少1-2小时学习时间
2. **动手实践**：看十遍不如写一遍
3. **记录笔记**：用自己的话总结知识点
4. **加入社区**：参与技术讨论，解决实际问题
5. **循序渐进**：不要跳过基础，稳扎稳打

---

## 进度追踪

#### 2.2 PyTorch实战

| 课程 | 内容 | 文件 |
|------|------|------|
| 第11课 | PyTorch基础 - Tensor、Autograd、线性回归 | `01_pytorch_basics.py` |
| 第12课 | 神经网络构建 - nn.Module、MLP、训练流程 | `02_neural_network.py` |

| 阶段 | 开始日期 | 完成日期 | 状态 |
|------|----------|----------|------|
| 阶段一：编程基础 | 2026-05-23 | 2026-05-24 | ✅ Python基础+数据科学完成 |
| 阶段二：深度学习 | 2026-05-24 | - | ⏳ PyTorch基础进行中 |
| 阶段三：NLP与大模型 | - | - | ⏳ 未开始 |
| 阶段四：实战项目 | - | - | ⏳ 未开始 |

---

*最后更新：2026-05-24 - Python基础+数据科学完成！精简计划：跳过传统ML和可视化，直接进入深度学习 → 大模型应用！*
