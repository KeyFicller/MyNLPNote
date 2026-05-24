# 第19课：LangChain基础与RAG入门

## 学习目标
- [x] 理解LangChain的核心价值和组件
- [x] 掌握Prompt Template的使用
- [x] 理解RAG（检索增强生成）的原理
- [x] 了解向量数据库的作用
- [ ] 能够构建简单的RAG系统

---

## 1. 为什么需要LangChain？

### 1.1 直接使用LLM的问题

```python
# 不使用LangChain的代码
import openai

def ask_question(question):
    # 每次都要写完整的prompt工程
    prompt = f"""你是一个专业的客服助手。
    请基于以下知识回答问题：
    
    知识库：
    - 公司年假15天
    - 病假每月2天
    
    用户问题：{question}
    
    请给出专业、友好的回答。"""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 问题：
# 1. Prompt管理混乱
# 2. 没有记忆功能（无法多轮对话）
# 3. 无法接入外部知识
# 4. 输出解析困难
```

### 1.2 LangChain的优势

```python
# 使用LangChain的代码
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferMemory

template = PromptTemplate(
    input_variables=["knowledge", "question"],
    template="""基于以下知识回答问题：
    
知识库：
{knowledge}

问题：{question}

回答："""
)

chain = LLMChain(
    llm=llm,
    prompt=template,
    memory=ConversationBufferMemory()
)

# 优雅、可复用、可扩展！
```

---

## 2. LangChain核心组件

### 2.1 架构图

```
┌─────────────────────────────────────────┐
│           LangChain 架构                │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────┐  ┌─────────┐  ┌────────┐ │
│  │  Model  │  │ Prompt  │  │ Output │ │
│  │ (LLM)   │  │Template │  │ Parser │ │
│  └────┬────┘  └────┬────┘  └────┬───┘ │
│       └─────────────┴─────────────┘    │
│                  ↓                      │
│            ┌──────────┐                 │
│            │  Chain   │                 │
│            │(组合组件)│                 │
│            └────┬─────┘                 │
│                 ↓                       │
│       ┌─────────┴─────────┐            │
│       ↓         ↓         ↓            │
│  ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Memory  │ │Tool   │ │Vector │       │
│  │(记忆)  │ │(工具) │ │Store  │       │
│  └────────┘ └────────┘ └────────┘       │
│                                         │
└─────────────────────────────────────────┘
```

### 2.2 核心组件说明

| 组件 | 作用 | 示例 |
|------|------|------|
| **Model** | LLM调用 | GPT-4, GPT-3.5, 本地模型 |
| **Prompt Template** | 可复用的提示模板 | 参数化输入 |
| **Chain** | 组件串联 | LLMChain, RetrievalChain |
| **Memory** | 对话记忆 | ConversationBufferMemory |
| **Tools** | 外部工具 | 搜索、计算、API调用 |
| **Vector Store** | 向量数据库 | Chroma, FAISS |

---

## 3. Prompt Template

### 3.1 基本用法

```python
from langchain import PromptTemplate

# 定义模板
template = PromptTemplate(
    input_variables=["topic", "tone"],
    template="""请以{tone}的语气，写一篇关于{topic}的短文。

要求：
1. 内容有趣易懂
2. 适合普通读者
3. 200字左右

文章："""
)

# 使用模板
prompt = template.format(
    topic="人工智能",
    tone="轻松幽默"
)

print(prompt)
```

### 3.2 少样本提示（Few-shot）

```python
from langchain import FewShotPromptTemplate

# 示例
examples = [
    {"input": "苹果", "output": "这是一种水果，通常是红色或绿色的"},
    {"input": "Python", "output": "这是一种编程语言，语法简洁优雅"},
]

example_template = """
词语：{input}
解释：{output}
"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(
        input_variables=["input", "output"],
        template=example_template
    ),
    suffix="词语：{input}\n解释：",
    input_variables=["input"]
)

# 使用
print(few_shot_prompt.format(input="LangChain"))
```

---

## 4. Chain

### 4.1 最简单的LLMChain

```python
from langchain import LLMChain, PromptTemplate
from langchain_community.llms import HuggingFacePipeline

# 创建LLM（这里用本地GPT-2，也可用OpenAI API）
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation"
)

# 创建Prompt Template
prompt = PromptTemplate(
    input_variables=["product"],
    template="为{product}写一句吸引人的广告语："
)

# 创建Chain
chain = LLMChain(llm=llm, prompt=prompt)

# 运行
result = chain.run(product="智能手表")
print(result)
```

### 4.2 Sequential Chain（顺序链）

```python
from langchain.chains import SimpleSequentialChain

# 第一个Chain：写标题
first_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template="为主题'{topic}'写一个吸引人的文章标题："
    ),
    output_key="title"
)

# 第二个Chain：基于标题写内容
second_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["title"],
        template="为标题'{title}'写一篇300字的短文："
    ),
    output_key="content"
)

# 组合成顺序链
overall_chain = SimpleSequentialChain(
    chains=[first_chain, second_chain],
    verbose=True
)

# 运行
result = overall_chain.run("人工智能")
```

---

## 5. RAG - 检索增强生成

### 5.1 为什么需要RAG？

**LLM的局限：**
1. 知识截止日期（无法知道最新信息）
2. 幻觉问题（编造不存在的信息）
3. 专业领域知识不足

**RAG的解决思路：**
给LLM配上"外接大脑"——检索相关知识

### 5.2 RAG流程

```
┌─────────────────────────────────────────┐
│              RAG 流程                   │
├─────────────────────────────────────────┤
│                                         │
│   1. 文档准备                            │
│      原始文档 → 切分成Chunks              │
│                    ↓                    │
│   2. 向量化（Indexing）                   │
│      Chunks → Embedding模型 → 向量       │
│                    ↓                    │
│   3. 存储                                │
│      向量 → 向量数据库（Chroma/FAISS）     │
│                    ↓                    │
│   4. 检索（Retrieval）                   │
│      用户问题 → Embedding → 相似度搜索     │
│                    ↓                    │
│   5. 生成（Generation）                  │
│      检索结果 + 用户问题 → LLM → 答案      │
│                                         │
└─────────────────────────────────────────┘
```

### 5.3 简单RAG实现

```python
# 使用Chroma向量数据库
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. 准备文档
documents = [
    "Python是一种解释型编程语言。",
    "Python由Guido van Rossum于1991年创建。",
    "Python广泛应用于数据科学和人工智能。",
]

# 2. 切分文档（如果需要）
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)
texts = text_splitter.create_documents(documents)

# 3. 创建Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 4. 存入向量数据库
db = Chroma.from_documents(texts, embeddings)

# 5. 创建检索器
retriever = db.as_retriever(
    search_kwargs={"k": 2}  # 返回前2个结果
)

# 6. 创建RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # 简单拼接文档
    retriever=retriever
)

# 7. 查询
result = qa_chain.run("Python是谁创建的？")
print(result)
```

---

## 6. 向量数据库

### 6.1 为什么用向量数据库？

**传统搜索的局限：**
- 关键词匹配："年假" ≠ "假期"
- 无法理解语义相似性

**向量检索的优势：**
- 语义相似："Python创建者" ≈ "谁发明了Python"
- 多语言："artificial intelligence" ≈ "人工智能"

### 6.2 Embedding原理

```
文本："Python是一种编程语言"
          ↓
   Embedding模型（如BERT）
          ↓
   向量：[0.1, -0.3, 0.8, ..., 0.2] (768维)
          ↓
   存储到向量数据库

查询："什么是Python？"
          ↓
   向量化：[0.12, -0.28, 0.79, ..., 0.21]
          ↓
   相似度计算（余弦相似度）
          ↓
   找到最相似的文档向量
```

### 6.3 常用向量数据库

| 数据库 | 特点 | 使用场景 |
|--------|------|----------|
| **Chroma** | 简单易用，Python友好 | 快速原型，本地开发 |
| **FAISS** | Meta开源，高效 | 大规模向量检索 |
| **Pinecone** | 云服务，无需维护 | 生产环境 |
| **Weaviate** | 功能丰富，GraphQL | 企业级应用 |
| **Milvus** | 国产，分布式 | 大规模生产 |

---

## 7. Memory - 对话记忆

### 7.1 为什么需要记忆？

```
用户：我叫小明
AI：你好小明！

用户：我叫什么？
AI：（没有记忆的话）我不知道。
```

### 7.2 Memory类型

| Memory类型 | 说明 | 适用场景 |
|------------|------|----------|
| `BufferMemory` | 保存完整对话历史 | 短对话 |
| `BufferWindowMemory` | 只保留最近K轮 | 长对话，控制长度 |
| `SummaryMemory` | 总结对话内容 | 超长对话 |
| `VectorStoreMemory` | 基于向量检索历史 | 需要查找特定信息 |

### 7.3 使用示例

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory()

# 创建带记忆的Chain
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# 第一轮
result1 = conversation.predict(input="你好，我叫小明")
print(result1)

# 第二轮（会自动包含历史）
result2 = conversation.predict(input="我叫什么名字？")
print(result2)  # 应该回答"小明"

# 查看记忆内容
print(memory.buffer)
```

---

## 8. 今日速查表

### 8.1 Prompt Template

```python
from langchain import PromptTemplate

# 基础模板
template = PromptTemplate(
    input_variables=["var1", "var2"],
    template="这是{var1}和{var2}"
)

# 格式化
prompt = template.format(var1="A", var2="B")
```

### 8.2 LLMChain

```python
from langchain import LLMChain

# 创建Chain
chain = LLMChain(llm=llm, prompt=prompt)

# 运行
result = chain.run(input="内容")
```

### 8.3 RAG完整流程

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# 1. 创建Embedding
embeddings = HuggingFaceEmbeddings(model_name="模型名")

# 2. 创建向量库
db = Chroma.from_documents(documents, embeddings)

# 3. 创建检索器
retriever = db.as_retriever(search_kwargs={"k": 3})

# 4. 创建RAG Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# 5. 查询
result = qa.run("问题")
```

---

## 9. 下节课预告

**完整的RAG项目实战**

- 使用Chroma向量数据库
- SBERT Embedding模型
- 文档切分策略
- 构建个人知识库助手
- 接入OpenAI API（可选）

**项目目标：**
构建一个可以回答你个人文档问题的AI助手！

---

*学习日期：2026-05-24*  
*进入LLM应用开发阶段！*
