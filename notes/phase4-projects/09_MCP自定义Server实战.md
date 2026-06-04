# 第30课：MCP 自定义 Server 实战

## 课程概述

Model Context Protocol（MCP）是连接 AI 应用与外部系统的开放标准。本课在 MyNLPNote 仓库中实现一个自定义 MCP Server，将 `notes/` 学习笔记暴露给 Cursor 等 AI 客户端，并编写完整的单元测试与集成测试。

**学习目标：**
1. 理解 MCP 的 Host / Client / Server 架构
2. 掌握 Tools、Resources、Prompts 三大原语
3. 使用 Python FastMCP 编写 MCP Server
4. 通过 MCP Client 测试 Server 协议交互
5. 在 Cursor 中配置并调用自定义 MCP

---

## 1. 什么是 MCP？

### 1.1 概念定义

MCP（Model Context Protocol）由 Anthropic 于 2024 年开源，提供 AI 应用连接外部数据和工具的统一协议。可以把它类比为 **AI 的 USB-C 接口**——写一次 Server，Cursor、Claude Desktop 等多个 Client 都能使用。

**核心价值：**
- **标准化集成**：替代各应用各自实现的 Function Calling 胶水代码
- **一次编写，多处复用**：同一个 Server 可被多个 AI 客户端连接
- **生态丰富**：社区已有 GitHub、PostgreSQL、Filesystem 等现成 Server

### 1.2 架构角色

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  MCP Host    │────▶│  MCP Client  │────▶│  MCP Server  │
│ (Cursor等)   │     │ (连接管理)   │     │ (暴露能力)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                                            │
       └────────────── LLM 决策调用工具 ──────────────┘
```

| 角色 | 说明 | 例子 |
|------|------|------|
| Host | 协调 MCP 连接的 AI 应用 | Cursor、Claude Desktop |
| Client | Host 内连接单个 Server 的组件 | Cursor 连接 nlp-notes 的那条通道 |
| Server | 暴露 Tools/Resources/Prompts 的程序 | 本课的 `mcp_nlp_notes_server.py` |

### 1.3 与 Function Calling 的区别

| 维度 | Function Calling | MCP |
|------|------------------|-----|
| 范围 | 单次工具调用 | 完整协议（发现、调用、通知、资源） |
| 复用 | 每个应用自己定义 | 写一次 Server，多 Client 共用 |
| 传输 | 无标准 | stdio / Streamable HTTP |
| 生态 | 各家格式不同 | 开放标准 + 社区 Server 目录 |

Function Calling 解决「模型如何调用一个函数」；MCP 解决「工具如何被发现、连接和管理」。

---

## 2. 三大原语

MCP Server 向 AI 暴露三类能力：

| 原语 | 作用 | 控制方 | 本课示例 |
|------|------|--------|---------|
| **Tools** | 可执行函数 | 模型自动决定调用 | `search_notes`, `read_note` |
| **Resources** | 只读数据源 | 应用/用户选择 | `notes://index` |
| **Prompts** | 预置提示词模板 | 用户选择 | `study_review` |

### 2.1 Tools 示例

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("nlp-notes")

@mcp.tool()
def count_words(text: str) -> str:
    """统计文本字数，返回总字符数、中文字符数、英文单词数。"""
    # ... 实现逻辑
    return "总字符: 10, 中文: 2, 英文单词: 3"
```

FastMCP 会根据函数签名和 docstring **自动生成 JSON Schema**，无需手写工具定义。

### 2.2 Resources 示例

```python
@mcp.resource("notes://index")
def notes_index() -> str:
    """返回 MyNLPNote 全部笔记的目录索引。"""
    return build_notes_index()

@mcp.resource("notes://{phase}/{filename}")
def note_resource(phase: str, filename: str) -> str:
    """按 URI 读取单篇笔记（只读资源）。"""
    return read_note_impl(phase, filename)
```

Resources 适合暴露**静态或半静态**的上下文，不需要模型「执行」操作。

### 2.3 Prompts 示例

```python
@mcp.prompt()
def study_review(topic: str) -> str:
    """生成复习某个 NLP 主题的提示词模板。"""
    return f"请帮我复习「{topic}」这个主题..."
```

Prompts 是用户可选的**预置工作流模板**，适合标准化常见任务。

---

## 3. 本项目 MCP Server 设计

### 3.1 文件结构

```
examples/llm-apps/
├── mcp_nlp_notes_server.py   # MCP Server（stdio 模式）
├── 09_mcp_server.py            # 课程演示脚本
└── test_09_mcp_server.py       # 单元测试 + MCP 集成测试

.cursor/
└── mcp.json                    # Cursor MCP 配置
```

### 3.2 暴露的能力

**Tools（4 个）：**

| 工具名 | 功能 |
|--------|------|
| `count_words` | 统计文本总字符、中文、英文单词 |
| `list_note_files` | 列出某阶段的 Markdown 笔记 |
| `search_notes` | 在 notes/ 中搜索关键词 |
| `read_note` | 读取指定笔记内容 |

**Resources（2 个）：**

| URI | 功能 |
|-----|------|
| `notes://index` | 全部笔记目录索引 |
| `notes://{phase}/{filename}` | 单篇笔记内容 |

**Prompts（1 个）：**

| 名称 | 功能 |
|------|------|
| `study_review` | 生成某主题的复习提示词 |

### 3.3 启动 Server

```bash
python examples/llm-apps/mcp_nlp_notes_server.py
```

Server 以 **stdio** 模式运行，通过标准输入/输出与 MCP Client 通信。不要在此模式下向 stdout 打印调试信息，否则会破坏 JSON-RPC 协议。

---

## 4. 测试 MCP Server

### 4.1 单元测试（纯函数）

不依赖 MCP 协议，直接测试业务逻辑：

```python
from mcp_nlp_notes_server import count_words_impl, search_notes_impl

def test_count_words():
    stats = count_words_impl("Hello 世界")
    assert stats["chinese_chars"] == 2
    assert stats["english_words"] == 1
```

### 4.2 集成测试（MCP Client）

模拟 Cursor 的行为，通过 MCP 协议连接 Server：

```python
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server_lists_tools():
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command="python",
        args=["examples/llm-apps/mcp_nlp_notes_server.py"],
    )
    stdio_transport = await exit_stack.enter_async_context(
        stdio_client(server_params)
    )
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(
        ClientSession(stdio, write)
    )
    await session.initialize()
    response = await session.list_tools()
    tool_names = [tool.name for tool in response.tools]
    assert "search_notes" in tool_names
    await exit_stack.aclose()
```

### 4.3 运行测试

```bash
# 安装依赖
pip install mcp pytest-asyncio

# 运行全部测试
python -m pytest examples/llm-apps/test_09_mcp_server.py -v

# 或运行课程脚本（含概念讲解 + 自动跑测试）
python examples/llm-apps/09_mcp_server.py
```

---

## 5. 在 Cursor 中配置

### 5.1 mcp.json

项目根目录 `.cursor/mcp.json`：

```json
{
  "mcpServers": {
    "nlp-notes": {
      "command": "python",
      "args": ["${workspaceFolder}/examples/llm-apps/mcp_nlp_notes_server.py"]
    }
  }
}
```

| 配置位置 | 路径 | 作用域 |
|---------|------|--------|
| 项目级 | `.cursor/mcp.json` | 仅当前项目 |
| 全局 | `~/.cursor/mcp.json` | 所有项目 |

### 5.2 使用步骤

1. `pip install mcp`
2. 重启 Cursor（或 Cmd/Ctrl+Shift+P → Reload Window）
3. 在 Agent 对话中尝试：
   - 「搜索 notes 里关于 Function Calling 的内容」
   - 「列出 phase4-projects 的所有笔记」
   - 「帮我复习 Transformer 主题」

### 5.3 调试

- **View → Output → MCP**：查看连接日志
- 状态栏 MCP 指示器：绿点表示已连接
- 配置修改后需重启 Cursor 才能生效

---

## 6. MCP 消息流（Tools 调用）

```
用户: "搜索 RAG 相关笔记"
         │
         ▼
    Cursor (Host)
         │
         ▼
    LLM 分析 → 决定调用 search_notes(keyword="RAG")
         │
         ▼
    MCP Client ──tools/call──▶ MCP Server
         │                         │
         │◀── 返回匹配笔记列表 ──────┘
         ▼
    LLM 根据结果生成回答
         │
         ▼
    返回给用户
```

底层通信基于 **JSON-RPC 2.0**，主要方法包括：
- `tools/list` — 发现可用工具
- `tools/call` — 调用工具
- `resources/list` / `resources/read` — 读取资源
- `prompts/list` / `prompts/get` — 获取提示词

---

## 7. 最佳实践

### 7.1 Server 设计

| 原则 | 说明 |
|------|------|
| 单一职责 | 一个 Server 专注一类能力（如笔记、数据库） |
| 清晰 docstring | LLM 根据描述决定何时调用工具 |
| 纯函数分离 | 业务逻辑与 MCP 装饰器分离，便于单元测试 |
| 输出截断 | 避免返回过长内容撑爆上下文 |

### 7.2 安全注意

- 验证所有工具输入
- 敏感操作需用户确认（Cursor 默认有确认机制）
- 不要在 `mcp.json` 中硬编码 API Key，使用 `${env:VAR_NAME}`
- stdio Server 禁止向 stdout 打印非协议内容

### 7.3 传输方式选择

| 传输 | 适用场景 |
|------|---------|
| stdio | 本地开发、单机工具（本课使用） |
| Streamable HTTP | 远程 Server、团队共享、需认证 |

---

## 8. 学习路径与进阶

### 8.1 本课完成内容

- MCP 架构与三大原语
- FastMCP 编写自定义 Server
- 单元测试 + MCP Client 集成测试
- Cursor 配置与调用

### 8.2 推荐进阶

| 方向 | 内容 | 资源 |
|------|------|------|
| 官方文档 | 协议规范、SDK 参考 | [modelcontextprotocol.io](https://modelcontextprotocol.io) |
| 现成 Server | GitHub、PostgreSQL、Filesystem | [github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) |
| 调试工具 | MCP Inspector | `npx @modelcontextprotocol/inspector` |
| 远程部署 | HTTP 传输 + OAuth | MCP 官方 Remote Server 文档 |
| Python 教程 | 从零到 Cursor 集成 | [Real Python MCP 教程](https://realpython.com/python-mcp/) |

---

## 总结

MCP 让 AI 应用以标准方式连接外部系统。通过本课，你已经：

1. **理解架构**：Host / Client / Server 与三大原语
2. **实现 Server**：将 MyNLPNote 笔记暴露为 MCP 工具
3. **编写测试**：纯函数单元测试 + MCP 协议集成测试
4. **接入 Cursor**：配置 `mcp.json` 并在 Agent 中使用

MCP 是 Function Calling 的「上层协议」——掌握它，你就有了构建可复用 AI 工具生态的能力。

---

**实践项目文件**: `examples/llm-apps/09_mcp_server.py`  
**MCP Server**: `examples/llm-apps/mcp_nlp_notes_server.py`  
**测试文件**: `examples/llm-apps/test_09_mcp_server.py`  
**Cursor 配置**: `.cursor/mcp.json`

```bash
# 课程演示
python examples/llm-apps/09_mcp_server.py

# 运行测试
python -m pytest examples/llm-apps/test_09_mcp_server.py -v
```
