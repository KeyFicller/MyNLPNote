#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第19课：LangChain基础与RAG入门
===============================
进入LLM应用开发实战！本课程学习：
- LangChain核心概念：Chain、Prompt、Memory
- 使用本地GPT-2和API模型
- 检索增强生成（RAG）原理
- 简单的知识库问答系统

需要安装：pip install langchain langchain-community
"""

import os
import sys

print("=" * 70)
print("第19课：LangChain基础与RAG入门 🚀")
print("=" * 70)

# ============================================================
# 第一部分：LangChain简介
# ============================================================
print("\n" + "=" * 70)
print("第一部分：LangChain是什么？")
print("=" * 70)

print("""
【为什么需要LangChain？】

直接使用GPT模型的问题：
1. 每次都要写重复的Prompt工程代码
2. 多轮对话管理复杂
3. 无法方便地接入外部数据
4. 链式调用多个模型困难
5. 输出解析和错误处理繁琐

LangChain解决了这些问题！

【LangChain核心组件】
┌─────────────────────────────────────────┐
│           LangChain 架构                │
├─────────────────────────────────────────┤
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
└─────────────────────────────────────────┘

核心概念：
- Model: LLM（大语言模型）
- Prompt Template: 可复用的提示模板
- Chain: 将组件串联成工作流
- Memory: 多轮对话记忆
- Tools: 外部工具（搜索、计算等）
- Vector Store: 向量数据库存储
""")

# ============================================================
# 第二部分：安装检查
# ============================================================
print("\n" + "=" * 70)
print("第二部分：环境检查")
print("=" * 70)

try:
    import langchain
    print(f"✅ langchain 已安装 (版本: {langchain.__version__})")
except ImportError:
    print("❌ 需要安装: pip install langchain langchain-community")
    print("""
安装命令：
pip install langchain langchain-community

如需更多功能：
pip install langchain-openai langchain-huggingface
""")
    sys.exit(1)

# 设置Hugging Face镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ============================================================
# 第三部分：简单的LLM调用（本地GPT-2）
# ============================================================
print("\n" + "=" * 70)
print("第三部分：使用LangChain调用本地GPT-2")
print("=" * 70)

print("""
【最简单的Chain】
Prompt Template → LLM → Output

比如：
输入变量：topic = "人工智能"
Prompt Template: "写一篇关于{topic}的短文："
                ↓
Prompt: "写一篇关于人工智能的短文："
                ↓
LLM (GPT-2)
                ↓
输出文本
""")

# 检查本地GPT-2是否可用
hf_cache_path = './hf_cache'
if not os.path.exists(hf_cache_path):
    print(f"⚠️ 未找到本地缓存目录: {hf_cache_path}")
    print("请先运行: python examples/nlp/05_load_gpt2.py 下载模型")
    print("\n下面展示概念代码...")
else:
    print(f"✅ 找到本地缓存: {hf_cache_path}")

try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    
    print("\n【示例1：使用本地GPT-2】")
    
    # 加载本地模型（简化版，实际使用时加载完整模型）
    print("加载本地GPT-2模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        'gpt2',
        cache_dir='./hf_cache'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'gpt2',
        cache_dir='./hf_cache'
    )
    
    # 创建pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8
    )
    
    # 包装成LangChain LLM
    local_llm = HuggingFacePipeline(pipeline=pipe)
    
    print("✅ GPT-2 已包装为LangChain LLM")
    
    # 简单调用
    prompt = "Once upon a time, there was a"
    print(f"\n输入: {prompt}")
    print("生成中...")
    
    result = local_llm.invoke(prompt)
    print(f"\n输出: {result}")
    
except Exception as e:
    print(f"⚠️ 模型加载失败: {e}")
    print("这很正常！继续学习概念，后续可用API或更轻量模型")
    print("\n继续展示LangChain核心概念...")

# ============================================================
# 第四部分：Prompt Template
# ============================================================
print("\n" + "=" * 70)
print("第四部分：Prompt Template - 可复用的提示模板")
print("=" * 70)

print("""
【为什么要用Prompt Template？】

直接写字符串的问题：
- 难以复用
- 参数管理混乱
- 格式容易出错

Prompt Template的好处：
- 参数化输入
- 模板复用
- 自动格式化
""")

# 展示Prompt Template概念（即使模型不可用也能演示）
print("\n【示例2：Prompt Template概念】")

# 模拟Prompt Template
class SimplePromptTemplate:
    """简化版Prompt Template演示"""
    
    def __init__(self, template):
        self.template = template
    
    def format(self, **kwargs):
        return self.template.format(**kwargs)

# 定义模板
translation_template = SimplePromptTemplate(
    """请将以下{text}翻译成{target_language}：

原文：{text}

翻译："""
)

# 使用模板
print("模板定义:")
print(f"{translation_template.template}")

formatted = translation_template.format(
    text="Hello, world!",
    target_language="中文"
)

print(f"\n格式化后:")
print(formatted)

# 模拟LLM输出
print("\n模拟LLM输出:")
print("你好，世界！")

# ============================================================
# 第五部分：RAG - 检索增强生成
# ============================================================
print("\n" + "=" * 70)
print("第五部分：RAG - 检索增强生成（核心！）")
print("=" * 70)

print("""
【为什么需要RAG？】

LLM的问题：
1. 知识截止日期 - 不知道最新信息
2. 幻觉（Hallucination） - 编造不存在的信息
3. 专业领域知识不足

【RAG解决方案】
┌─────────────────────────────────────────┐
│              RAG 流程                   │
├─────────────────────────────────────────┤
│                                         │
│   用户问题："公司的年假政策是什么？"      │
│                    ↓                    │
│         ┌─────────────────┐             │
│         │  1. 检索知识库  │             │
│         │     (Vector DB) │             │
│         └────────┬────────┘             │
│                  ↓                        │
│    相关文档：["员工手册第3章...",         │
│              "HR政策文档..."]            │
│                  ↓                        │
│         ┌─────────────────┐             │
│         │ 2. 构建增强Prompt│             │
│         │                │             │
│         │ 基于以下文档回答：│             │
│         │ [检索到的文档]   │             │
│         │                │             │
│         │ 问题：用户问题   │             │
│         └────────┬────────┘             │
│                  ↓                        │
│         ┌─────────────────┐             │
│         │  3. LLM生成答案  │             │
│         └────────┬────────┘             │
│                  ↓                        │
│         回答："根据员工手册..."           │
│                                         │
└─────────────────────────────────────────┘

【核心步骤】
1. 文档切分 → 文本块（Chunks）
2. Embedding → 向量化
3. 存储 → 向量数据库（如FAISS）
4. 检索 → 相似度搜索
5. 生成 → LLM结合检索内容回答
""")

# 模拟RAG流程
print("\n【示例3：模拟RAG流程】")

# 模拟知识库
documents = [
    "员工年假：每年15天带薪年假，入职满一年后可享受。",
    "病假规定：每月2天带薪病假，需提供医院证明。",
    "加班政策：工作日加班按1.5倍工资，周末2倍，节假日3倍。",
    "远程办公：每周可申请2天远程办公，需提前一天申请。",
]

print("知识库文档:")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc}")

# 模拟用户问题
user_question = "年假有多少天？"
print(f"\n用户问题: {user_question}")

# 模拟检索（简单关键词匹配）
def mock_retrieve(query, docs, top_k=2):
    """模拟检索：根据关键词匹配"""
    # 简化的BM25/相似度模拟
    scores = []
    for doc in docs:
        score = sum(1 for word in query if word in doc)
        scores.append((doc, score))
    
    # 排序返回top_k
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scores[:top_k]]

retrieved_docs = mock_retrieve(user_question, documents)

print("\n检索到的相关文档:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"  {i}. {doc}")

# 构建增强Prompt
augmented_prompt = f"""基于以下信息回答问题：

相关信息：
{chr(10).join(f"- {doc}" for doc in retrieved_docs)}

问题：{user_question}

回答："""

print(f"\n增强后的Prompt:")
print(augmented_prompt)

print("\n模拟LLM回答:")
print("根据公司政策，员工每年享有15天带薪年假，需要入职满一年后才能享受。")

# ============================================================
# 第六部分：简单的本地知识库实现
# ============================================================
print("\n" + "=" * 70)
print("第六部分：简单的知识库问答实现")
print("=" * 70)

print("""
【最简单的RAG实现（无Embedding）】

使用关键词匹配 + LLM生成
""")

class SimpleRAG:
    """简化版RAG系统"""
    
    def __init__(self, documents):
        self.documents = documents
    
    def add_documents(self, docs):
        self.documents.extend(docs)
    
    def retrieve(self, query, top_k=2):
        """基于关键词的简单检索"""
        # 分词
        query_words = set(query.lower().split())
        
        scores = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            # Jaccard相似度简化版
            intersection = query_words & doc_words
            score = len(intersection)
            scores.append((doc, score))
        
        # 排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]
    
    def generate_prompt(self, query, retrieved_docs):
        """构建增强Prompt"""
        context = "\n".join(f"- {doc}" for doc in retrieved_docs)
        return f"""基于以下信息回答问题。如果信息不足，请说明。

相关信息：
{context}

问题：{query}

回答："""
    
    def query(self, query, llm=None):
        """完整的RAG查询流程"""
        # 1. 检索
        docs = self.retrieve(query)
        
        # 2. 构建Prompt
        prompt = self.generate_prompt(query, docs)
        
        # 3. 生成（这里返回Prompt，实际使用时会传给LLM）
        return {
            'retrieved_documents': docs,
            'prompt': prompt,
            'query': query
        }

# 创建知识库
print("\n【示例4：创建简单知识库】")

kb_documents = [
    "Python是一种解释型、高级编程语言，由Guido van Rossum于1991年创建。",
    "Python的设计哲学强调代码的可读性，使用缩进来表示代码块。",
    "Python广泛应用于Web开发、数据分析、人工智能和科学计算。",
    "Django和Flask是Python最流行的Web框架。",
    "TensorFlow和PyTorch是Python中主要的深度学习框架。",
    "Python拥有丰富的标准库和第三方包管理系统pip。",
]

rag = SimpleRAG(kb_documents)

print("知识库已创建，包含以下文档：")
for i, doc in enumerate(kb_documents, 1):
    print(f"  {i}. {doc[:50]}...")

# 测试查询
test_questions = [
    "Python是谁创建的？",
    "Python在人工智能领域有什么应用？",
    "Python的Web框架有哪些？",
]

print("\n测试查询:")
for question in test_questions:
    print(f"\n{'='*50}")
    print(f"问题: {question}")
    
    result = rag.query(question)
    
    print(f"\n检索到的文档:")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        print(f"  {i}. {doc}")
    
    print(f"\n生成的Prompt:")
    print(result['prompt'][:300] + "...")

# ============================================================
# 第七部分：向量数据库概念
# ============================================================
print("\n" + "=" * 70)
print("第七部分：向量数据库（生产级RAG必备）")
print("=" * 70)

print("""
【为什么需要向量数据库？】

我们刚才用的是关键词匹配，问题：
- "年假" 和 "假期" 被认为是不同的词
- "Python创建者" 和 "谁发明了Python" 语义相同但匹配不上

【解决方案：Embedding + 向量检索】

┌─────────────────────────────────────────┐
│         向量数据库流程                  │
├─────────────────────────────────────────┤
│                                         │
│  文档："Python由Guido创建"               │
│          ↓                              │
│  Embedding模型（如BERT）                  │
│          ↓                              │
│  向量：[0.1, -0.3, 0.8, ...] (768维)     │
│          ↓                              │
│  存入向量数据库（FAISS/Chroma）           │
│                                         │
│  ─────────────────────────────────────   │
│                                         │
│  查询："谁发明了Python"                   │
│          ↓                              │
│  Embedding → [0.12, -0.28, 0.79, ...]   │
│          ↓                              │
│  向量相似度搜索（余弦相似度）              │
│          ↓                              │
│  找到最相似的文档向量                      │
│          ↓                              │
│  返回："Python由Guido创建"                │
│                                         │
└─────────────────────────────────────────┘

【常用向量数据库】

| 数据库 | 特点 | 适用场景 |
|-------|------|---------|
| FAISS | Meta开源，高效 | 本地开发，中小型数据 |
| Chroma | 简单易用 | 快速原型，Python友好 |
| Pinecone | 云服务 | 生产环境，大规模数据 |
| Weaviate | 功能丰富 | 企业级应用 |
| Milvus | 国产，分布式 | 大规模生产环境 |
""")

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 70)
print("总结与实践")
print("=" * 70)

print("""
【本课核心知识点】
✅ LangChain：LLM应用开发框架
✅ Prompt Template：参数化、可复用的提示模板
✅ Chain：将LLM、Prompt、Memory等组件串联
✅ RAG：检索增强生成，解决LLM知识局限
✅ 向量数据库：用Embedding实现语义检索

【RAG核心步骤】
1. 文档切分 → Chunks
2. Embedding → 向量化（BERT/GPT）
3. 存储 → 向量数据库
4. 检索 → 相似度搜索
5. 生成 → 增强Prompt + LLM

【课后实践】
1. 安装Chroma/FAISS：pip install chromadb faiss-cpu
2. 使用Embedding模型向量化你的文档
3. 构建完整的RAG系统（文档→向量→检索→生成）
4. 接入GPT-2或API模型做真正的生成
5. 优化检索：尝试不同的Embedding模型和分块策略

【推荐阅读】
- LangChain文档：https://python.langchain.com/
- RAG论文：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- Embedding模型：sentence-transformers (SBERT)

【下节课预告】
完整的RAG项目实战！
- 使用Chroma向量数据库
- SBERT Embedding模型
- 构建个人知识库助手
- 接入OpenAI API（可选）
""")

print("\n" + "=" * 70)
print("第19课完成！进入LLM应用开发阶段！🎉")
print("=" * 70)
