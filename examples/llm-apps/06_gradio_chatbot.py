#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第27课：Gradio 聊天机器人部署
==============================
项目名称：AI Agent Web 聊天界面
任务：为第26课的 RAG Agent 添加 Gradio 可视化界面

学习目标：
- 学习 Gradio 基础：组件、布局、事件绑定
- 将命令行 Agent 封装为 Web 服务
- 实现多轮对话历史展示
- 支持模型切换、成本统计展示
- 本地运行 + 可分享链接

前置课程：
- 第26课：05_rag_agent.py

运行方式：
    python examples/llm-apps/06_gradio_chatbot.py
    
访问：
    http://127.0.0.1:7860

文档：https://www.gradio.app/docs
"""

import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# HuggingFace 镜像配置
_EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
from hf_mirror import setup_hf_mirror
setup_hf_mirror()

import gradio as gr

from agent_core import (
    TOOLS,
    Tool,
    ReActAgent,
    LLMClient,
    ModelConfig,
    PRESET_MODELS,
    LLMProvider,
)

# 第26课的知识库相关（内置简化版）
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

RAG_AGENT_RULES = """
【工具选择规则 — 必须遵守】
- Python/PyTorch/Transformer/BERT/RAG/本课程资料 → knowledge_base（优先）
- 最新新闻、实时事件、未收录信息 → search
- 数学计算 → calculator
- 当前时间 → datetime()
- 运行 Python 代码 → python
"""

RAG_ACTION_HINTS = [
    'Action: knowledge_base(query="RAG技术的优势")',
    'Action: datetime()',
    'Action: calculator(expression="(123 + 456) * 7")',
    'Action: search(query="2024年新闻")',
]

RAG_TOOL_NAMES = ["knowledge_base", "calculator", "search", "datetime", "python"]


class KnowledgeBase:
    """封装 ChromaDB 向量检索，供 Agent 作为工具调用（简化版）"""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.vectorstore = None
        self.persist_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", ".cache", "chroma_db")
        )

    def setup(self) -> bool:
        print("📚 初始化知识库...")
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

# =============================================================================
# 第一部分：配置和初始化
# =============================================================================

print("=" * 70)
print("第27课：Gradio 聊天机器人部署")
print("=" * 70)

# 全局状态
class ChatState:
    """聊天状态管理"""
    def __init__(self):
        self.agent: Optional[ReActAgent] = None
        self.llm: Optional[LLMClient] = None
        self.history: List[Tuple[str, str]] = []  # (user, assistant)
        self.current_model: str = "deepseek"
        self.total_tokens: int = 0
        self.total_cost: float = 0.0
        self.kb_initialized: bool = False
        self.knowledge_base: Optional[KnowledgeBase] = None

state = ChatState()


def init_knowledge_base() -> bool:
    """初始化知识库"""
    if state.kb_initialized:
        return True
    
    print("📚 初始化知识库...")
    kb = KnowledgeBase()
    if not kb.setup():
        print("❌ 知识库初始化失败")
        return False
    
    state.knowledge_base = kb
    state.kb_initialized = True
    print("✅ 知识库就绪")
    return True


def create_agent(model_name: str) -> Tuple[bool, str]:
    """
    创建指定模型的 Agent
    
    Returns:
        (success, message)
    """
    if model_name not in PRESET_MODELS:
        return False, f"未知模型: {model_name}"
    
    config = PRESET_MODELS[model_name]
    
    # 检查 API Key
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        return False, f"未设置环境变量: {config.api_key_env}"
    
    # 初始化 LLM
    try:
        llm = LLMClient(config)
        if not llm.api_key:
            return False, "API Key 无效"
    except Exception as e:
        return False, f"LLM 初始化失败: {e}"
    
    # 准备工具
    if state.knowledge_base:
        tools = build_rag_tools(state.knowledge_base)
        tools = {name: tool for name, tool in tools.items() if name in RAG_TOOL_NAMES}
    else:
        tools = dict(TOOLS)
    
    # 创建 Agent
    agent = ReActAgent(
        llm,
        tools,
        max_iterations=5,
        extra_system_rules=RAG_AGENT_RULES,
        action_format_hints=RAG_ACTION_HINTS,
    )
    
    state.agent = agent
    state.llm = llm
    state.current_model = model_name
    
    return True, f"✅ 已切换到 {model_name} ({config.provider.value})"


# =============================================================================
# 第二部分：Gradio 界面回调函数
# =============================================================================

def respond(message: str, history: List[dict]) -> List[dict]:
    """
    处理用户消息并返回更新后的对话历史（Gradio messages 格式）
    
    Args:
        message: 用户输入的消息
        history: 当前对话历史（Gradio messages 格式）
    
    Returns:
        更新后的对话历史列表 [{"role": "user"/"assistant", "content": ...}, ...]
    """
    if not state.agent:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ 请先选择模型并初始化 Agent（点击「🚀 初始化 Agent」按钮）"})
        return history
    
    if not message.strip():
        return history
    
    # 添加用户消息
    history.append({"role": "user", "content": message})
    
    # 运行 Agent
    try:
        result = state.agent.run(message)
        
        # 更新统计
        if state.llm:
            stats = state.llm.get_stats()
            state.total_tokens = stats["total_tokens"]
            state.total_cost = stats["total_cost_usd"]
        
        # 添加助手回复
        history.append({"role": "assistant", "content": result})
        return history
    except Exception as e:
        history.append({"role": "assistant", "content": f"❌ 错误: {str(e)}"})
        return history


def on_model_change(model_name: str) -> str:
    """切换模型"""
    success, msg = create_agent(model_name)
    return msg


def get_stats_text() -> str:
    """获取统计信息文本"""
    if not state.llm:
        return "未初始化"
    
    stats = state.llm.get_stats()
    return f"""📊 当前会话统计

• 模型: {state.current_model}
• 总 Token: {stats['total_tokens']:,}
• 预估成本: ${stats['total_cost_usd']:.4f} USD
• 预估成本: ¥{stats['total_cost_rmb']:.2f} RMB"""


def clear_chat() -> Tuple[List[dict], str]:
    """清空对话"""
    state.history = []
    # 重置 LLM 统计（可选）
    if state.llm:
        state.llm.total_tokens = 0
        state.llm.total_cost = 0.0
    return [], "✅ 对话已清空"


# =============================================================================
# 第三部分：Gradio 界面构建
# =============================================================================

def build_ui() -> gr.Blocks:
    """构建 Gradio 界面"""
    
    with gr.Blocks(title="AI Agent 聊天机器人") as demo:
        
        # 标题
        gr.Markdown("""
        # 🤖 AI Agent 聊天机器人
        
        基于 **ReAct Agent + RAG 知识库** 的智能助手
        
        **能力**：知识库问答 · 实时搜索 · 数学计算 · Python 执行
        """)
        
        with gr.Row():
            # 左侧：控制面板
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ 设置")
                
                # 模型选择
                model_dropdown = gr.Dropdown(
                    choices=list(PRESET_MODELS.keys()),
                    value="deepseek",
                    label="选择模型",
                    info="需要对应 API Key 环境变量",
                )
                
                init_btn = gr.Button("🚀 初始化 Agent", variant="primary")
                init_status = gr.Textbox(
                    label="状态",
                    value="请选择模型并点击初始化",
                    interactive=False,
                )
                
                gr.Markdown("---")
                
                # 统计信息
                stats_box = gr.Textbox(
                    label="📊 Token 统计",
                    value=get_stats_text(),
                    interactive=False,
                    lines=6,
                )
                
                refresh_stats_btn = gr.Button("🔄 刷新统计")
                
                gr.Markdown("---")
                
                # 操作按钮
                clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                
                gr.Markdown("""
                ---
                
                ### 💡 使用提示
                
                1. 先点击「初始化 Agent」
                2. 在右侧输入框提问
                3. Agent 会自动选择工具回答
                
                **示例问题**：
                - Python列表有哪些常用方法？
                - RAG技术解决了哪些问题？
                - 计算 2**10
                - 2024年诺贝尔文学奖得主是谁？
                """)
            
            # 右侧：聊天界面
            with gr.Column(scale=2):
                gr.Markdown("## 💬 对话")
                
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                )
                
                msg_input = gr.Textbox(
                    label="输入消息",
                    placeholder="请输入问题...",
                    show_label=False,
                )
                
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary")
                    stop_btn = gr.Button("停止")
        
        # 事件绑定
        init_btn.click(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[init_status],
        )
        
        model_dropdown.change(
            fn=on_model_change,
            inputs=[model_dropdown],
            outputs=[init_status],
        )
        
        refresh_stats_btn.click(
            fn=get_stats_text,
            outputs=[stats_box],
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, init_status],
        )
        
        # 聊天交互
        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],  # 清空输入框
        ).then(
            fn=get_stats_text,
            outputs=[stats_box],  # 更新统计
        )
        
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot],
        ).then(
            fn=lambda: "",
            outputs=[msg_input],
        ).then(
            fn=get_stats_text,
            outputs=[stats_box],
        )
        
        return demo


# =============================================================================
# 第四部分：主函数
# =============================================================================

def main():
    """主函数"""
    
    # 初始化知识库
    if not init_knowledge_base():
        print("⚠️ 知识库初始化失败，继续启动（部分功能不可用）")
    
    # 构建界面
    demo = build_ui()
    
    print("\n" + "=" * 70)
    print("🚀 Gradio 服务启动中...")
    print("=" * 70)
    print("本地访问: http://127.0.0.1:7860")
    print("按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    # 启动服务
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,  # 设为 True 可生成公网链接（需 gradio 账号）
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
