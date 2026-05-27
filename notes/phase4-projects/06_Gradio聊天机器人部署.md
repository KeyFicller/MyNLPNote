# 第27课：Gradio 聊天机器人部署

**项目**: AI Agent Web 聊天界面  
**技术栈**: Gradio + ReAct Agent + RAG 知识库  
**前置课程**: 第26课 RAG 与 Agent 融合

---

## 1. 什么是 Gradio？

Gradio 是一个开源 Python 库，让你**用几行代码**就能为机器学习模型创建**可视化 Web 界面**。

### 特点

| 特性 | 说明 |
|------|------|
| **简单易用** | 纯 Python，无需前端知识 |
| **组件丰富** | 输入框、下拉菜单、聊天界面、文件上传等 |
| **实时交互** | 支持流式输出、进度条 |
| **一键分享** | 可生成公网临时链接（share=True） |
| **嵌入支持** | 可嵌入 HuggingFace Spaces、Notebook |

### 与类似工具对比

| 工具 | 学习曲线 | 功能丰富度 | 适用场景 |
|------|----------|------------|----------|
| **Gradio** | ⭐ 低 | 中等 | 快速原型、演示 |
| **Streamlit** | ⭐ 低 | 中高 | 数据应用、仪表盘 |
| **FastAPI + React** | ⭐⭐⭐ 高 | 高 | 生产级应用 |

---

## 2. Gradio 核心概念

### 2.1 组件（Components）

```python
import gradio as gr

# 输入组件
text_input = gr.Textbox(label="输入", placeholder="请输入...")
dropdown = gr.Dropdown(choices=["gpt-4", "deepseek"], label="模型")

# 输出组件
text_output = gr.Textbox(label="输出")
chatbot = gr.Chatbot(label="对话历史")  # 聊天机器人核心组件
```

### 2.2 布局（Layouts）

```python
with gr.Blocks() as demo:
    with gr.Row():           # 横向排列
        with gr.Column(scale=1):   # 左侧栏，占比1
            # 控制面板
        with gr.Column(scale=2):   # 右侧栏，占比2
            # 聊天区域
```

### 2.3 事件绑定（Events）

```python
# 按钮点击
def on_click(msg):
    return f"收到: {msg}"

btn.click(fn=on_click, inputs=[text_input], outputs=[text_output])

# 回车提交
text_input.submit(fn=on_click, inputs=[text_input], outputs=[text_output])

# 下拉框变化
dropdown.change(fn=on_change, inputs=[dropdown], outputs=[status_text])
```

---

## 3. 聊天机器人核心设计

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────┐
│                     Gradio Web UI                        │
│  ┌──────────────┐  ┌─────────────────────────────────┐ │
│  │ 控制面板      │  │           聊天区域               │ │
│  │ - 模型选择    │  │  ┌──────────────────────────┐  │ │
│  │ - 初始化按钮  │  │  │     对话历史 (Chatbot)    │  │ │
│  │ - Token 统计  │  │  └──────────────────────────┘  │ │
│  │ - 清空按钮    │  │  ┌──────────────────────────┐  │ │
│  └──────────────┘  │  │     输入框 + 发送按钮     │  │ │
│                    │  └──────────────────────────┘  │ │
│                    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    ReAct Agent + RAG                   │
│        (第26课的知识库 + 工具调用能力)                    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 关键代码结构

```python
# 状态管理
class ChatState:
    agent: Optional[ReActAgent] = None
    llm: Optional[LLMClient] = None
    history: List[Tuple[str, str]] = []

state = ChatState()

# 处理用户消息
def respond(message: str, history: List[Tuple[str, str]]) -> str:
    if not state.agent:
        return "请先初始化 Agent"
    result = state.agent.run(message)
    return result

# 构建 UI
def build_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):  # 左侧控制面板
                model_dropdown = gr.Dropdown(...)
                init_btn = gr.Button("初始化")
                
            with gr.Column(scale=2):  # 右侧聊天区
                chatbot = gr.Chatbot()
                msg_input = gr.Textbox()
                
        # 事件绑定
        init_btn.click(fn=init_agent, ...)
        msg_input.submit(fn=respond, ...)
    
    return demo
```

---

## 4. 运行方式

### 4.1 安装依赖

```bash
pip install gradio>=4.0.0
```

已在 `requirements.txt` 中添加。

### 4.2 启动服务

```bash
# 确保 API Key 已设置
$env:DEEPSEEK_API_KEY = "your-key"

# 运行
python examples/llm-apps/06_gradio_chatbot.py
```

### 4.3 访问界面

```
本地访问: http://127.0.0.1:7860
```

---

## 5. 核心功能点

### 5.1 模型切换

- 支持下拉框选择不同 LLM（DeepSeek / GPT-4 / Claude 等）
- 切换时自动重新初始化 Agent
- 检查对应 API Key 是否设置

### 5.2 对话历史

- `gr.Chatbot` 组件自动渲染对话气泡
- 支持多轮上下文（Agent 内部维护）
- 一键清空对话

### 5.3 Token 统计

- 实时显示当前会话 Token 消耗
- 预估成本（USD / RMB）
- 按模型分别统计

### 5.4 知识库问答

- 复用第26课的 `KnowledgeBase`
- 向量库持久化，首次加载较慢，后续秒开
- 自动检测已有索引，避免重复处理

---

## 6. 进阶扩展思路

| 功能 | 实现方式 |
|------|----------|
| **流式输出** | `gr.Chatbot` + `yield` 生成器 |
| **文件上传** | `gr.File` + 文档解析入库 |
| **语音输入** | `gr.Audio` + Whisper ASR |
| **图片展示** | `gr.Image` + 多模态 Agent |
| **公网分享** | `demo.launch(share=True)` |
| **用户认证** | Gradio 的 `auth` 参数 |

---

## 7. 常见问题

### Q1: 端口被占用？

```python
demo.launch(server_port=7861)  # 换一个端口
```

### Q2: 如何自定义主题？

```python
demo = gr.Blocks(theme=gr.themes.Soft())  # Soft / Default / Monochrome
```

### Q3: 生产部署？

Gradio 适合**演示和原型**，生产环境建议：
- FastAPI + 前端框架（React/Vue）
- Docker 容器化部署
- 使用官方 `gradio` 的 `queue()` 处理并发

---

## 8. 下一步学习

- **第28课**: Function Calling（解决 Agent 解析不稳定性）
- **第29课**: Agent Memory（长期记忆系统）
- **第30课**: 多模态应用（图像 + 文本）

---

**实践项目文件**: `examples/llm-apps/06_gradio_chatbot.py`  
**文档**: https://www.gradio.app/docs
