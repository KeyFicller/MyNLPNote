#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第30课：MCP 自定义 Server 实战
==============================

课程目标：
- 理解 MCP（Model Context Protocol）的架构与三大原语
- 使用 FastMCP 编写自定义 MCP Server
- 通过 MCP Client 测试 Server 的 tools / resources / prompts
- 在 Cursor 中配置并调用自定义 MCP

应用场景：
- 将本仓库 notes/ 笔记暴露给 AI Agent
- 统一封装内部工具，供多个 AI 客户端复用
- 替代各应用各自实现的 Function Calling 集成

运行方式：
    # 课程演示（概念讲解 + 本地函数演示）
    python examples/llm-apps/09_mcp_server.py

    # 运行全部测试
    python -m pytest examples/llm-apps/test_09_mcp_server.py -v

    # 单独启动 MCP Server（供 Cursor 连接）
    python examples/llm-apps/mcp_nlp_notes_server.py
"""

import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_nlp_notes_server import (
    NOTES_ROOT,
    build_notes_index,
    count_words_impl,
    list_note_files_impl,
    search_notes_impl,
)

print("=" * 70)
print("第30课：MCP 自定义 Server 实战")
print("=" * 70)
print()

# =============================================================================
# 第一部分：MCP 核心概念
# =============================================================================

print("第一部分：MCP 核心概念")
print("-" * 70)

print("""
什么是 MCP（Model Context Protocol）？

MCP 是连接 AI 应用与外部系统的开放标准，类似「AI 的 USB-C 接口」。

架构角色：
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  MCP Host    │────▶│  MCP Client  │────▶│  MCP Server  │
│ (Cursor等)   │     │ (连接管理)   │     │ (暴露能力)   │
└──────────────┘     └──────────────┘     └──────────────┘

三大原语：
  Tools     — 可执行函数（模型自动调用）  例：search_notes()
  Resources — 只读数据源（应用/用户选择）  例：notes://index
  Prompts   — 预置提示词模板（用户选择）  例：study_review

与 Function Calling 的关系：
  Function Calling = 单次工具调用能力
  MCP              = 工具发现、连接、管理的完整协议（写一次，多客户端复用）
""")

# =============================================================================
# 第二部分：本项目的 MCP Server 设计
# =============================================================================

print("第二部分：本项目的 MCP Server 设计")
print("-" * 70)

print("""
Server 名称：nlp-notes
Server 文件：examples/llm-apps/mcp_nlp_notes_server.py

Tools（4 个）：
  count_words      — 统计文本字数
  list_note_files  — 列出某阶段的笔记文件
  search_notes     — 在 notes/ 中搜索关键词
  read_note        — 读取指定笔记内容

Resources（2 个）：
  notes://index              — 全部笔记目录索引
  notes://{phase}/{filename} — 单篇笔记（模板 URI）

Prompts（1 个）：
  study_review — 生成某主题的复习提示词
""")

# =============================================================================
# 第三部分：本地函数演示（不经过 MCP 协议）
# =============================================================================

print("第三部分：本地函数演示")
print("-" * 70)

sample_text = "MCP 让 AI 能调用外部工具 Model Context Protocol"
stats = count_words_impl(sample_text)
print(f"count_words: {stats}")

phase4_files = list_note_files_impl("phase4-projects", NOTES_ROOT)
print(f"\nphase4-projects 笔记 ({len(phase4_files)} 篇):")
for name in phase4_files[:5]:
    print(f"  - {name}")
if len(phase4_files) > 5:
    print(f"  ... 共 {len(phase4_files)} 篇")

hits = search_notes_impl("RAG", "phase4-projects", NOTES_ROOT, max_results=3)
print(f"\nsearch_notes('RAG'): 找到 {len(hits)} 条")
for hit in hits:
    print(f"  [{hit['phase']}] {hit['file']}")

index_preview = build_notes_index(NOTES_ROOT)
print(f"\nnotes://index 预览（前 300 字符）:")
print(index_preview[:300] + "...")
print()

# =============================================================================
# 第四部分：Cursor 配置说明
# =============================================================================

print("第四部分：Cursor 配置")
print("-" * 70)

mcp_json = Path(__file__).resolve().parents[2] / ".cursor" / "mcp.json"
print(f"""
项目已包含 Cursor MCP 配置：{mcp_json}

配置内容示例：
{{
  "mcpServers": {{
    "nlp-notes": {{
      "command": "python",
      "args": ["${{workspaceFolder}}/examples/llm-apps/mcp_nlp_notes_server.py"]
    }}
  }}
}}

使用步骤：
  1. pip install mcp pytest-asyncio
  2. 重启 Cursor（或 Reload Window）
  3. 在 Agent 对话中尝试：
     - 「搜索 notes 里关于 Function Calling 的内容」
     - 「列出 phase4-projects 的所有笔记」
     - 「帮我复习 Transformer 主题」
""")

# =============================================================================
# 第五部分：运行 pytest 集成测试
# =============================================================================

print("第五部分：运行 MCP 测试")
print("-" * 70)

test_file = Path(__file__).resolve().parent / "test_09_mcp_server.py"
print(f"执行: python -m pytest {test_file} -v\n")

result = subprocess.run(
    [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
    cwd=str(Path(__file__).resolve().parents[2]),
    capture_output=False,
)

print()
if result.returncode == 0:
    print("全部测试通过！MCP Server 已就绪。")
else:
    print(f"测试失败（exit code {result.returncode}），请检查 mcp 是否已安装。")
    print("  pip install mcp pytest-asyncio")

print()
print("=" * 70)
print("课程完成。下一步：重启 Cursor，在 Agent 中体验 nlp-notes MCP。")
print("=" * 70)
