#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第23课：完整的RAG项目实战 - 个人知识库助手
=========================================

项目名称：AI文档问答助手
功能：构建一个基于RAG的个人知识库，可以回答关于上传文档的问题

学习目标：
- 掌握Chroma向量数据库的使用
- 理解SBERT Embedding模型的原理和应用
- 学习文档切分策略（Chunking）
- 实现完整的RAG流程：索引 → 检索 → 生成
- 构建一个可用的问答系统

技术栈：
- LangChain: RAG流程框架
- ChromaDB: 向量数据库
- Sentence-Transformers: Embedding模型
- HuggingFace: 本地LLM（可选OpenAI API）

运行前请确保：
- 虚拟环境已激活
- 已安装: pip install langchain langchain-community chromadb sentence-transformers
"""

import os
import sys

# 配置Hugging Face镜像（中国大陆用户）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

print("=" * 60)
print("🚀 第23课：完整的RAG项目实战 - 个人知识库助手")
print("=" * 60)
print(f"🌐 使用镜像源: {os.environ['HF_ENDPOINT']}")
print(f"📁 缓存目录: {os.environ['HF_HOME']}")
print()

# =============================================================================
# 第一部分：依赖检查与安装提示
# =============================================================================

print("📦 第一步：检查依赖库")
print("-" * 60)

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    print("✅ LangChain 相关库导入成功")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("\n尝试旧版导入方式...")
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        print("✅ 旧版导入方式成功")
    except ImportError as e2:
        print(f"❌ 旧版导入也失败: {e2}")
    print("\n请安装所需依赖库：")
    print("   pip install langchain langchain-community chromadb sentence-transformers")
    print("   pip install langchain-huggingface  # 新版embedding接口")
    sys.exit(1)

try:
    import chromadb
    print("✅ ChromaDB 导入成功")
except ImportError:
    print("❌ 未安装 ChromaDB")
    print("   pip install chromadb")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    print("✅ Sentence-Transformers 导入成功")
except ImportError:
    print("❌ 未安装 sentence-transformers")
    print("   pip install sentence-transformers")
    sys.exit(1)

print("\n✅ 所有依赖检查通过！")
print()

# =============================================================================
# 第二部分：模拟文档数据
# =============================================================================

print("📚 第二步：准备模拟文档数据")
print("-" * 60)

# 模拟知识库文档 - 关于Python编程的知识
DOCUMENTS = [
    {
        "title": "Python简介",
        "content": """Python是一种高级、解释型、通用的编程语言。它由Guido van Rossum于1991年创建。
Python的设计哲学强调代码的可读性和简洁性，使用缩进来表示代码块。
Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
Python有丰富的标准库，覆盖了从文本处理到网络编程的各种功能。"""
    },
    {
        "title": "Python列表",
        "content": """列表（List）是Python中最常用的数据结构之一。列表是有序的可变序列，可以存储任意类型的元素。
创建列表使用方括号：my_list = [1, 2, 3, 'hello', True]
列表支持索引访问：first_element = my_list[0]
列表切片：sub_list = my_list[1:3]  # 获取索引1到2的元素
常用方法包括：
- append(x): 在末尾添加元素
- insert(i, x): 在位置i插入元素x
- remove(x): 删除第一个值为x的元素
- pop(i): 删除并返回位置i的元素
- sort(): 原地排序列表
- reverse(): 原地反转列表"""
    },
    {
        "title": "Python字典",
        "content": """字典（Dictionary）是Python中的键值对数据结构，使用花括号定义。
字典是无序的（Python 3.7+ 保持插入顺序）、可变的映射类型。
创建字典：my_dict = {'name': 'Alice', 'age': 25, 'city': 'Beijing'}
访问值：name = my_dict['name']  # 或使用 my_dict.get('name')
修改值：my_dict['age'] = 26
添加键值对：my_dict['job'] = 'Engineer'
删除键值对：del my_dict['city'] 或 my_dict.pop('city')
遍历字典：
for key in my_dict:  # 遍历键
    print(key, my_dict[key])
for key, value in my_dict.items():  # 遍历键值对
    print(key, value)"""
    },
    {
        "title": "Python函数",
        "content": """函数是组织好的、可重复使用的代码块，用于实现单一或相关联的功能。
定义函数使用def关键字：
def greet(name, greeting='Hello'):
    return f"{greeting}, {name}!"
参数类型：
- 位置参数：调用时必须按顺序传入
- 默认参数：定义时指定默认值
- 关键字参数：调用时指定参数名
- 可变参数：*args接收任意数量的位置参数，**kwargs接收任意数量的关键字参数
lambda表达式创建匿名函数：
square = lambda x: x ** 2
函数可以返回多个值（实际上是元组）：
def get_stats(numbers):
    return min(numbers), max(numbers), sum(numbers)/len(numbers)"""
    },
    {
        "title": "PyTorch简介",
        "content": """PyTorch是一个开源的深度学习框架，由Facebook AI Research (FAIR)开发。
PyTorch的核心是Tensor（张量），类似于NumPy的ndarray，但可以在GPU上运行。
PyTorch的主要特点：
1. 动态计算图：定义即运行，调试方便
2. GPU加速：使用CUDA实现高速并行计算
3. 自动求导：Autograd自动计算梯度
4. 丰富的神经网络模块：torch.nn提供常用层
创建Tensor：
import torch
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.zeros(3, 3)
z = torch.randn(2, 2)  # 标准正态分布
移动到GPU：
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)"""
    },
    {
        "title": "Transformer架构",
        "content": """Transformer是一种深度学习架构，由Vaswani等人在2017年的论文《Attention Is All You Need》中提出。
Transformer的核心创新是自注意力机制（Self-Attention），可以并行处理序列中的所有位置。
Transformer架构包含两个主要部分：
1. 编码器（Encoder）：将输入序列转换为连续的表示
2. 解码器（Decoder）：根据编码器的输出生成目标序列
编码器由多层相同的层组成，每层包含：
- 多头自注意力机制（Multi-Head Self-Attention）
- 前馈神经网络（Feed-Forward Network）
- 层归一化（Layer Normalization）
- 残差连接（Residual Connections）
注意力机制计算查询（Query）、键（Key）、值（Value）之间的相关性。
位置编码（Positional Encoding）为模型提供序列顺序信息。"""
    },
    {
        "title": "BERT模型",
        "content": """BERT（Bidirectional Encoder Representations from Transformers）是Google在2018年发布的预训练语言模型。
BERT的核心创新是使用双向Transformer编码器来预训练深度语言表示。
BERT的两个预训练任务：
1. 掩码语言模型（Masked Language Model, MLM）：随机掩盖输入中的部分词，让模型预测被掩盖的词
2. 下一句预测（Next Sentence Prediction, NSP）：判断两个句子是否连续
BERT的变体包括：
- BERT-Base: 12层，768隐藏维度，12个注意力头，1.1亿参数
- BERT-Large: 24层，1024隐藏维度，16个注意力头，3.4亿参数
- RoBERTa: 优化的BERT训练方法
- ALBERT: 参数共享的轻量级BERT
- DistilBERT: 蒸馏后的BERT，速度快60%，保留97%性能
微调BERT用于下游任务：文本分类、命名实体识别、问答系统等。"""
    },
    {
        "title": "RAG技术",
        "content": """RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合信息检索和文本生成的技术。
RAG的工作流程：
1. 索引阶段（Indexing）：
   - 文档切分（Chunking）
   - 使用Embedding模型将文档转换为向量
   - 存储到向量数据库
2. 检索阶段（Retrieval）：
   - 将用户查询转换为向量
   - 在向量数据库中搜索相似的文档片段
   - 返回最相关的Top-K个结果
3. 生成阶段（Generation）：
   - 将检索到的文档作为上下文
   - 构建增强的提示词（Prompt）
   - 使用LLM生成答案
RAG的优势：
- 解决LLM的知识截止问题
- 减少模型幻觉（Hallucination）
- 可以引用信息来源，提高可信度
- 无需重新训练模型即可更新知识"""
    }
]

print(f"📄 准备了 {len(DOCUMENTS)} 篇文档")
for i, doc in enumerate(DOCUMENTS, 1):
    print(f"   {i}. {doc['title']} ({len(doc['content'])} 字符)")
print()

# =============================================================================
# 第三部分：文档切分（Chunking）策略
# =============================================================================

print("✂️ 第三步：文档切分策略")
print("-" * 60)

def demonstrate_chunking():
    """演示不同的文档切分策略"""
    
    # 示例长文档
    long_text = """Python是一种高级编程语言。它具有简单易学、功能强大的特点。
    Python支持面向对象编程。你可以定义类和对象。
    类是对象的蓝图。对象是根据类创建的实例。
    继承是面向对象的重要特性。子类可以继承父类的属性和方法。
    多态允许子类重写父类的方法。这提高了代码的灵活性。
    Python还支持函数式编程。lambda表达式可以创建匿名函数。
    高阶函数可以接受其他函数作为参数。map、filter都是高阶函数。
    装饰器是Python的高级特性。它可以在不修改原函数的情况下增加功能。"""
    
    print("📐 切分策略对比：\n")
    
    # 策略1：按字符数切分（简单）
    print("策略1：按固定字符数切分")
    print("   特点：简单快速，但可能切断句子")
    chunks_simple = [long_text[i:i+100] for i in range(0, len(long_text), 80)]
    print(f"   切分结果：{len(chunks_simple)} 个片段")
    print(f"   片段1长度：{len(chunks_simple[0])} 字符")
    print()
    
    # 策略2：递归字符切分（推荐）
    print("策略2：递归字符切分（RecursiveCharacterTextSplitter）")
    print("   特点：智能边界检测，优先保持段落和句子完整")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,      # 每个片段目标大小
        chunk_overlap=30,    # 片段间重叠字符数（保持上下文连续性）
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " ", ""]  # 切分优先级
    )
    
    chunks_recursive = splitter.split_text(long_text)
    print(f"   切分结果：{len(chunks_recursive)} 个片段")
    print(f"   参数：chunk_size=150, chunk_overlap=30")
    for i, chunk in enumerate(chunks_recursive[:3], 1):
        preview = chunk[:50].replace('\n', ' ')
        print(f"   片段{i}：{preview}... ({len(chunk)} 字符)")
    if len(chunks_recursive) > 3:
        print(f"   ... 还有 {len(chunks_recursive) - 3} 个片段")
    print()
    
    # 策略3：语义切分（概念性介绍）
    print("策略3：语义切分（Semantic Chunking）")
    print("   特点：基于语义相似度切分，需要Embedding模型")
    print("   实现：计算句子间的语义相似度，在主题变化处切分")
    print("   优势：每个片段主题更集中，检索更准确")
    print()

demonstrate_chunking()

# =============================================================================
# 第四部分：Embedding模型与向量化
# =============================================================================

print("🔢 第四步：Embedding模型与向量化")
print("-" * 60)

def demonstrate_embedding():
    """演示Embedding模型的使用"""
    
    print("📚 SBERT（Sentence-BERT）简介：")
    print("   - 基于BERT的句子Embedding模型")
    print("   - 将句子映射为固定维度的向量")
    print("   - 语义相似的句子在向量空间距离近")
    print()
    
    # 使用HuggingFace的轻量级中文Embedding模型
    # paraphrase-multilingual-MiniLM-L12-v2 支持多语言，包括中文
    print("🔄 加载Embedding模型...")
    print("   模型：sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    print("   维度：384维")
    print("   语言：多语言（含中文）")
    
    try:
        # 创建Embedding模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},  # 使用CPU，如有GPU可改为'cuda'
            encode_kwargs={'normalize_embeddings': True}  # 归一化向量
        )
        print("   ✅ 模型加载成功")
        
        # 测试Embedding
        test_sentences = [
            "Python是一种编程语言",
            "Python is a programming language",
            "BERT是预训练语言模型",
            "机器学习是人工智能的分支"
        ]
        
        print("\n🧪 测试Embedding：")
        vectors = embeddings.embed_documents(test_sentences[:2])
        print(f"   句子1向量维度：{len(vectors[0])}")
        print(f"   向量前5个值：[{', '.join([f'{v:.4f}' for v in vectors[0][:5]])}...]")
        
        # 计算相似度
        import numpy as np
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        sim = cosine_similarity(np.array(vectors[0]), np.array(vectors[1]))
        print(f"   中英文'Python是编程语言'相似度：{sim:.4f}")
        
        return embeddings
        
    except Exception as e:
        print(f"   ❌ 模型加载失败: {e}")
        print("   将使用模拟Embedding进行演示")
        return None

embeddings = demonstrate_embedding()
print()

# =============================================================================
# 第五部分：Chroma向量数据库
# =============================================================================

print("🗄️ 第五步：Chroma向量数据库")
print("-" * 60)

def setup_vector_store(embeddings):
    """设置向量数据库"""
    
    print("📦 ChromaDB特点：")
    print("   - 开源、轻量级向量数据库")
    print("   - 支持持久化存储")
    print("   - 内置相似度搜索")
    print("   - 元数据过滤")
    print()
    
    # 准备文档
    print("📄 准备文档...")
    documents = []
    for doc in DOCUMENTS:
        # 对每个文档进行切分
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        chunks = splitter.split_text(doc['content'])
        
        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "title": doc['title'],
                    "chunk_id": i,
                    "source": "知识库"
                }
            ))
    
    print(f"   原始文档：{len(DOCUMENTS)} 篇")
    print(f"   切分后片段：{len(documents)} 个")
    print()
    
    # 创建向量存储
    print("🔄 创建向量存储...")
    
    # 持久化目录
    persist_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'chroma_db')
    os.makedirs(persist_dir, exist_ok=True)
    
    if embeddings:
        try:
            # 使用真实Embedding模型
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=persist_dir,
                collection_name="knowledge_base"
            )
            print(f"   ✅ 向量存储创建成功")
            print(f"   📁 持久化目录：{persist_dir}")
            return vectorstore
        except Exception as e:
            print(f"   ⚠️ 创建失败: {e}")
            return None
    else:
        print("   ⚠️ 使用模拟模式（无Embedding）")
        return None

vectorstore = setup_vector_store(embeddings)
print()

# =============================================================================
# 第六部分：检索与相似度搜索
# =============================================================================

print("🔍 第六部分：检索与相似度搜索")
print("-" * 60)

def demonstrate_retrieval(vectorstore):
    """演示检索功能"""
    
    if not vectorstore:
        print("⚠️ 向量存储未创建，跳过检索演示")
        return
    
    # 测试查询
    test_queries = [
        "Python列表如何添加元素？",
        "什么是Transformer？",
        "BERT怎么预训练的？",
        "RAG有什么好处？",
        "PyTorch的张量是什么？"
    ]
    
    print("🧪 相似度搜索测试：\n")
    
    for query in test_queries:
        print(f"查询：\"{query}\"")
        
        # 相似度搜索
        results = vectorstore.similarity_search_with_score(query, k=2)
        
        print("   检索结果：")
        for i, (doc, score) in enumerate(results, 1):
            # 归一化分数（越小越相似）
            similarity = 1 - score  # 简单转换
            preview = doc.page_content[:60].replace('\n', ' ')
            print(f"   {i}. [{doc.metadata['title']}] {preview}... (相似度: {similarity:.3f})")
        print()

if vectorstore:
    demonstrate_retrieval(vectorstore)
else:
    print("⚠️ 跳过检索演示（向量存储未创建）")
    print()

# =============================================================================
# 第七部分：完整的RAG问答系统
# =============================================================================

print("🤖 第七部分：完整的RAG问答系统")
print("=" * 60)

class SimpleRAGSystem:
    """
    简化的RAG问答系统
    不依赖外部API，使用模板模拟LLM生成
    """
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.retrieval_count = 3  # 检索文档数量
        
    def retrieve(self, query: str) -> list:
        """检索相关文档"""
        if not self.vectorstore:
            return []
        
        docs = self.vectorstore.similarity_search(query, k=self.retrieval_count)
        return docs
    
    def build_prompt(self, query: str, contexts: list) -> str:
        """构建增强提示词"""
        
        # 拼接上下文
        context_text = "\n\n".join([
            f"[文档{i+1}] {ctx.metadata['title']}:\n{ctx.page_content}"
            for i, ctx in enumerate(contexts)
        ])
        
        # RAG提示词模板
        prompt = f"""基于以下参考文档回答问题：

参考文档：
{context_text}

问题：{query}

请根据参考文档提供准确、简洁的回答。如果文档中没有相关信息，请说明无法回答。

回答："""
        return prompt
    
    def generate(self, query: str, contexts: list) -> dict:
        """
        生成答案（模拟LLM）
        实际应用中，这里应该调用LLM API（如OpenAI、本地模型等）
        """
        
        # 分析查询关键词
        query_lower = query.lower()
        
        # 基于检索内容构建回答
        answers = []
        sources = []
        
        for ctx in contexts:
            content_lower = ctx.page_content.lower()
            title = ctx.metadata['title']
            
            # 简单的关键词匹配（实际使用LLM会更智能）
            if any(kw in query_lower for kw in ['python', '列表', '字典', '函数']):
                if title == 'Python列表' and '列表' in query:
                    answers.append("Python列表使用append()添加元素，insert()插入元素，extend()扩展列表。")
                    sources.append(title)
                elif title == 'Python字典' and '字典' in query:
                    answers.append("Python字典使用键值对存储数据，可以通过键来访问和修改值。")
                    sources.append(title)
                    
            elif any(kw in query_lower for kw in ['transformer', '注意力']):
                if title == 'Transformer架构':
                    answers.append("Transformer是一种基于自注意力机制的深度学习架构，由编码器和解码器组成。")
                    sources.append(title)
                    
            elif any(kw in query_lower for kw in ['bert', '预训练']):
                if title == 'BERT模型':
                    answers.append("BERT使用双向Transformer编码器，通过MLM和NSP任务进行预训练。")
                    sources.append(title)
                    
            elif any(kw in query_lower for kw in ['rag', '检索', '增强']):
                if title == 'RAG技术':
                    answers.append("RAG通过检索相关文档来增强LLM的生成能力，减少幻觉并提供引用来源。")
                    sources.append(title)
                    
            elif any(kw in query_lower for kw in ['pytorch', '张量']):
                if title == 'PyTorch简介':
                    answers.append("PyTorch的核心是张量（Tensor），可以在GPU上加速计算，支持自动求导。")
                    sources.append(title)
        
        # 组合回答
        if answers:
            final_answer = " ".join(set(answers))  # 去重
        else:
            final_answer = "根据知识库中的文档，这个问题涉及" + ", ".join([ctx.metadata['title'] for ctx in contexts[:2]]) + "。请参考相关文档获取详细信息。"
        
        return {
            "answer": final_answer,
            "sources": list(set(sources)) if sources else [ctx.metadata['title'] for ctx in contexts[:2]],
            "prompt": self.build_prompt(query, contexts)
        }
    
    def query(self, question: str) -> dict:
        """完整的RAG查询流程"""
        
        print(f"📝 问题：{question}")
        print()
        
        # Step 1: 检索
        print("🔍 Step 1: 检索相关文档...")
        contexts = self.retrieve(question)
        
        if not contexts:
            return {
                "answer": "未能检索到相关文档。",
                "sources": [],
                "prompt": ""
            }
        
        print(f"   检索到 {len(contexts)} 个相关片段：")
        for i, ctx in enumerate(contexts, 1):
            preview = ctx.page_content[:50].replace('\n', ' ')
            print(f"   {i}. [{ctx.metadata['title']}] {preview}...")
        print()
        
        # Step 2: 构建提示词
        print("📝 Step 2: 构建增强提示词...")
        prompt = self.build_prompt(question, contexts)
        print(f"   提示词长度：{len(prompt)} 字符")
        print("   提示词预览（前200字符）：")
        print(f"   {prompt[:200]}...")
        print()
        
        # Step 3: 生成答案
        print("🤖 Step 3: 生成答案...")
        result = self.generate(question, contexts)
        
        return result


# 创建RAG系统实例
print("🏗️ 初始化RAG系统...")
if vectorstore:
    rag_system = SimpleRAGSystem(vectorstore)
    print("✅ RAG系统初始化完成\n")
    
    # 测试问答
    test_questions = [
        "Python列表如何添加元素？",
        "Transformer是什么？",
        "RAG技术有什么好处？"
    ]
    
    print("🧪 RAG问答测试：\n")
    for q in test_questions:
        print("=" * 50)
        result = rag_system.query(q)
        print()
        print(f"💡 回答：{result['answer']}")
        print(f"📚 来源：{', '.join(result['sources'])}")
        print()
else:
    print("⚠️ 向量存储未创建，无法初始化RAG系统")
    print()

# =============================================================================
# 第八部分：项目总结与进阶方向
# =============================================================================

print("=" * 60)
print("📋 项目总结与进阶方向")
print("=" * 60)

summary = """
✅ 本课完成内容：

1. 文档切分策略
   - 理解了Chunking的重要性和不同策略
   - 使用RecursiveCharacterTextSplitter进行智能切分
   - 设置合适的chunk_size和chunk_overlap

2. Embedding模型
   - 了解了SBERT将句子转换为向量的原理
   - 使用HuggingFaceEmbeddings加载多语言模型
   - 理解了语义相似度的计算方法

3. 向量数据库Chroma
   - 学习了向量数据库的基本概念
   - 实现了文档的索引和持久化存储
   - 掌握了相似度搜索的使用

4. 完整RAG流程
   - 索引：文档 → 切分 → Embedding → 存储
   - 检索：查询 → Embedding → 相似度搜索 → Top-K文档
   - 生成：构建增强提示词 → LLM生成 → 带引用的答案

🔧 实际部署建议：

1. LLM选择：
   - 在线API：OpenAI GPT-4/3.5, Claude, 文心一言
   - 本地模型：ChatGLM, LLaMA, Qwen等开源模型

2. Embedding优化：
   - 中文场景：shibing624/text2vec-base-chinese
   - 通用场景：BAAI/bge-large-zh-v1.5（SOTA中文Embedding）

3. 向量数据库选择：
   - 轻量级：Chroma（本课使用）
   - 生产级：Pinecone, Milvus, Weaviate

4. 性能优化：
   - 使用GPU加速Embedding计算
   - 对向量进行量化减少存储
   - 实施缓存策略避免重复计算

5. 高级功能：
   - 混合搜索：向量搜索 + 关键词搜索
   - 重排序（Reranking）：使用Cross-Encoder精排
   - 对话历史：多轮对话的上下文管理
   - 引用标注：在答案中标注信息来源

📚 下一步学习建议：
- 接入真实的LLM API进行测试
- 尝试加载自己的PDF/Word文档
- 学习LangChain的Agent功能
- 探索LlamaIndex框架
"""

print(summary)

print("=" * 60)
print("🎉 第23课完成！你已经掌握了完整的RAG技术栈！")
print("=" * 60)
print()
print("【下节课预告】")
print("   第24课：AI Agent开发入门")
print("   - Agent的核心概念（规划、工具、记忆）")
print("   - ReAct范式（推理+行动）")
print("   - 构建简单的任务执行Agent")
print()
print("准备好继续吗？还是先休息一下？😊")
