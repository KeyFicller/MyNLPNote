#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第26课：RAG 与 Agent 融合实战
==============================
项目名称：带私有知识库的智能 Agent
任务：将第23课 RAG 知识库封装为 Agent 工具，实现「查资料 + 推理 + 行动」

学习目标：
- 理解 RAG 作为 Agent 工具 vs 独立 RAG 问答的区别
- 将 ChromaDB 向量检索封装为 knowledge_base 工具
- 让 Agent 自主决定何时查知识库、何时用计算器/搜索
- 构建企业级「知识库助手」原型

前置课程：
- 第23课：02_complete_rag_project.py
- 第25课：04_ai_agent_with_api.py / agent_core.py

运行前请确保：
- pip install -r requirements.txt
- 设置 LLM API Key（如 $env:DEEPSEEK_API_KEY）
"""

import os
import sys
from typing import Dict, List, Optional

# HuggingFace 镜像必须在任何 HF 库导入之前配置
_EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
from hf_mirror import setup_hf_mirror

_HF_HOME = setup_hf_mirror()

print("=" * 70)
print("第26课：RAG 与 Agent 融合实战")
print("=" * 70)
print(f"🌐 HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")
print(f"📁 模型缓存目录: {_HF_HOME}")

from agent_core import (
    TOOLS,
    Tool,
    ReActAgent,
    select_model,
    create_llm_client,
)

# =============================================================================
# 第一部分：知识库文档（与第23课一致）
# =============================================================================

KNOWLEDGE_DOCUMENTS = [
    {
        "title": "Python列表",
        "content": """列表（List）是Python中最常用的数据结构之一。列表是有序的可变序列，可以存储任意类型的元素。
常用方法包括：
- append(x): 在末尾添加元素
- insert(i, x): 在位置i插入元素x
- remove(x): 删除第一个值为x的元素
- pop(i): 删除并返回位置i的元素
- sort(): 原地排序列表
- reverse(): 原地反转列表""",
    },
    {
        "title": "PyTorch简介",
        "content": """PyTorch是一个开源的深度学习框架，由Facebook AI Research (FAIR)开发。
PyTorch的核心是Tensor（张量），可以在GPU上运行。
主要特点：动态计算图、GPU加速、Autograd自动求导、torch.nn神经网络模块。""",
    },
    {
        "title": "Transformer架构",
        "content": """Transformer由Vaswani等人在2017年提出，核心创新是自注意力机制（Self-Attention）。
主要组成：编码器（Encoder）、解码器（Decoder）、多头注意力、前馈网络、层归一化、残差连接。
位置编码（Positional Encoding）为模型提供序列顺序信息。""",
    },
    {
        "title": "BERT模型",
        "content": """BERT是Google在2018年发布的预训练语言模型，使用双向Transformer编码器。
两个预训练任务：
1. 掩码语言模型（MLM）：预测被掩盖的词
2. 下一句预测（NSP）：判断两个句子是否连续
可用于文本分类、命名实体识别、问答等下游任务。""",
    },
    {
        "title": "RAG技术",
        "content": """RAG（Retrieval-Augmented Generation）结合检索与生成。
流程：索引（切分→Embedding→向量库）→ 检索（相似度搜索）→ 生成（LLM结合上下文回答）。
优势：解决知识截止、减少幻觉、可引用来源、无需重新训练即可更新知识。""",
    },
]

print("""
【RAG + Agent 融合架构】

用户问题
    ↓
ReAct Agent（LLM 规划）
    ↓
┌──────────────────────────────────────┐
│  knowledge_base  →  查私有知识库(RAG) │
│  search          →  查互联网实时信息   │
│  calculator      →  数学计算          │
│  python          →  代码执行          │
└──────────────────────────────────────┘
    ↓
Observation → 继续推理 → Final Answer

与纯 RAG 的区别：Agent 自己决定调用哪个工具，可组合多步任务。
""")


# =============================================================================
# 第二部分：知识库管理
# =============================================================================

class KnowledgeBase:
    """封装 ChromaDB 向量检索，供 Agent 作为工具调用"""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.vectorstore = None
        self.persist_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "chroma_db")
        )

    def setup(self) -> bool:
        print("\n📚 初始化知识库...")
        print(f"   持久化目录: {self.persist_dir}")

        try:
            from langchain_community.vectorstores import Chroma
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.documents import Document
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma
                from langchain_community.embeddings import HuggingFaceEmbeddings
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain.schema import Document
            except ImportError as e:
                print(f"❌ 缺少依赖: {e}")
                print("   pip install langchain langchain-community chromadb sentence-transformers")
                return False

        os.makedirs(self.persist_dir, exist_ok=True)

        print("   加载 Embedding 模型...")
        print(f"   镜像: {os.environ.get('HF_ENDPOINT', 'default')}")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            print(f"❌ Embedding 模型加载失败: {e}")
            print("   可能原因：网络超时或模型未完整下载")
            print("   建议：重试脚本，或手动执行：")
            print("   $env:HF_ENDPOINT='https://hf-mirror.com'")
            print("   huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            return False

        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=embeddings,
                collection_name=self.collection_name,
            )
            if self.vectorstore._collection.count() > 0:
                count = self.vectorstore._collection.count()
                print(f"   ✅ 加载已有知识库（{count} 条片段）")
                return True
        except Exception:
            self.vectorstore = None

        print("   首次运行，正在索引文档...")
        documents = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "，", " ", ""],
        )
        for doc in KNOWLEDGE_DOCUMENTS:
            for i, chunk in enumerate(splitter.split_text(doc["content"])):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"title": doc["title"], "chunk_id": i, "source": "课程知识库"},
                    )
                )

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name,
        )
        print(f"   ✅ 知识库索引完成（{len(documents)} 条片段）")
        return True

    def search(self, query: str, top_k: int = 3) -> str:
        if not self.vectorstore:
            return "知识库未初始化，请先运行 setup()"

        docs = self.vectorstore.similarity_search(query, k=top_k)
        if not docs:
            return "知识库中未找到相关内容"

        parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "未知来源")
            parts.append(f"[{i}] 来源: {title}\n{doc.page_content}")
        return "\n\n".join(parts)


# =============================================================================
# 第三部分：构建 RAG Agent 工具集
# =============================================================================

def build_rag_tools(knowledge_base: KnowledgeBase) -> Dict[str, Tool]:
    """在第25课工具基础上增加 knowledge_base 工具"""

    def knowledge_base_tool(query: str) -> str:
        return knowledge_base.search(query)

    tools = dict(TOOLS)
    tools["knowledge_base"] = Tool(
        name="knowledge_base",
        description=(
            "私有课程知识库检索。用于 Python/PyTorch/Transformer/BERT/RAG 等"
            "本仓库学习资料相关问题。优先于网络搜索使用。"
        ),
        parameters={"query": "检索关键词或问题，字符串类型"},
        func=knowledge_base_tool,
    )
    return tools


RAG_AGENT_RULES = """
【工具选择规则 — 必须遵守】
- Python/PyTorch/Transformer/BERT/RAG/本课程资料 → knowledge_base（优先）
- 最新新闻、实时事件、未收录信息 → search
- 数学计算 → calculator
- 当前时间 → datetime()
- 运行 Python 代码 → python

【Action 格式 — 必须严格遵守】
Action: knowledge_base(query="Python列表常用方法")
Action: calculator(expression="2 ** 10")

【收到 Observation 后】
- 不要再次调用工具
- 直接输出 Thought + Final Answer
"""

# 本课只保留相关工具，减少模型选错工具的概率
RAG_TOOL_NAMES = ["knowledge_base", "calculator", "search", "datetime", "python"]

RAG_ACTION_HINTS = [
    'Action: knowledge_base(query="RAG技术的优势")',
    'Action: datetime()',
    'Action: calculator(expression="(123 + 456) * 7")',
    'Action: search(query="2024年新闻")',
]


# =============================================================================
# 第四部分：测试运行
# =============================================================================

def main():
    kb = KnowledgeBase()
    if not kb.setup():
        print("\n❌ 知识库初始化失败，退出")
        sys.exit(1)

    tools = build_rag_tools(kb)
    tools = {name: tool for name, tool in tools.items() if name in RAG_TOOL_NAMES}
    print("\n🛠️  Agent 工具（含 knowledge_base）:")
    for name in tools:
        print(f"   • {name}")

    print("\n" + "=" * 70)
    print("第四部分：选择 LLM 并测试")
    print("=" * 70)

    config = select_model()
    llm = create_llm_client(config)
    agent = ReActAgent(
        llm,
        tools,
        max_iterations=5,
        extra_system_rules=RAG_AGENT_RULES,
        action_format_hints=RAG_ACTION_HINTS,
    )

    test_cases = [
        "Python列表有哪些常用方法？请查知识库回答。",
        "RAG技术解决了哪些问题？",
        "Transformer的核心创新是什么？",
        "计算 2 ** 10 等于多少？",
    ]

    print("\n" + "=" * 70)
    print("🧪 开始测试")
    print("=" * 70)

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(test_cases)}")
        print(f"{'='*70}")
        try:
            result = agent.run(test)
            print(f"\n📝 结果: {result}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
        if i < len(test_cases):
            input("\n按回车继续下一个测试...")

    print("\n" + "=" * 70)
    print("💬 交互模式（输入 'exit' 退出）")
    print("=" * 70)
    print("💡 试试：BERT的两个预训练任务是什么？ / 计算 100*25")

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ["exit", "quit", "退出"]:
            break
        if not user_input:
            continue
        try:
            result = agent.run(user_input)
            print(f"\n🤖 Agent: {result}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")

    stats = llm.get_stats()
    print("\n" + "=" * 70)
    print("📊 使用统计")
    print("=" * 70)
    print(f"   总Token数: {stats['total_tokens']}")
    print(f"   预估成本: ${stats['total_cost_usd']:.4f} USD")
    print(f"   预估成本: ¥{stats['total_cost_rmb']:.2f} RMB")
    print("\n✨ 第26课完成！你已掌握 RAG + Agent 融合！")


if __name__ == "__main__":
    main()
