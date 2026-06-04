#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCP NLP Notes Server 测试 — 单元测试 + MCP 协议集成测试"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 确保同目录模块可导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp_nlp_notes_server import (  # noqa: E402
    NOTES_ROOT,
    count_words_impl,
    list_note_files_impl,
    read_note_impl,
    search_notes_impl,
)

SERVER_PATH = str(Path(__file__).resolve().parent / "mcp_nlp_notes_server.py")
EXPECTED_TOOLS = ["count_words", "list_note_files", "search_notes", "read_note"]


# ---------------------------------------------------------------------------
# 单元测试：纯函数逻辑（不依赖 MCP 协议）
# ---------------------------------------------------------------------------


class TestCountWords:
    def test_mixed_text(self):
        stats = count_words_impl("Hello 世界 NLP")
        assert stats["total_chars"] == 12
        assert stats["chinese_chars"] == 2
        assert stats["english_words"] == 2

    def test_empty_text(self):
        stats = count_words_impl("")
        assert stats["total_chars"] == 0
        assert stats["chinese_chars"] == 0
        assert stats["english_words"] == 0


class TestListNoteFiles:
    def test_phase4_has_notes(self):
        files = list_note_files_impl("phase4-projects", NOTES_ROOT)
        assert len(files) >= 1
        assert all(name.endswith(".md") for name in files)

    def test_invalid_phase_raises(self):
        with pytest.raises(ValueError, match="无效阶段"):
            list_note_files_impl("invalid-phase", NOTES_ROOT)


class TestSearchNotes:
    def test_find_function_calling(self):
        hits = search_notes_impl("Function Calling", "phase4-projects", NOTES_ROOT)
        assert len(hits) >= 1
        assert any("Function" in hit["file"] or "function" in hit["snippet"].lower() for hit in hits)

    def test_empty_keyword_returns_empty(self):
        assert search_notes_impl("", "all", NOTES_ROOT) == []


class TestReadNote:
    def test_read_existing_note(self):
        files = list_note_files_impl("phase4-projects", NOTES_ROOT)
        content = read_note_impl("phase4-projects", files[0], NOTES_ROOT)
        assert len(content) > 0
        assert "#" in content or len(content) > 10

    def test_missing_note_raises(self):
        with pytest.raises(FileNotFoundError):
            read_note_impl("phase4-projects", "nonexistent_note.md", NOTES_ROOT)


# ---------------------------------------------------------------------------
# 集成测试：通过 MCP Client 连接 Server（模拟 Cursor 行为）
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_server_lists_tools():
    """连接 MCP Server，验证 tools/list 返回预期工具。"""
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=None,
    )

    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    await session.initialize()
    response = await session.list_tools()
    tool_names = sorted(tool.name for tool in response.tools)

    assert tool_names == sorted(EXPECTED_TOOLS)
    await exit_stack.aclose()


@pytest.mark.asyncio
async def test_mcp_server_call_count_words():
    """通过 tools/call 调用 count_words 工具。"""
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=None,
    )

    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    await session.initialize()
    result = await session.call_tool("count_words", {"text": "MCP 测试 test"})
    text_blocks = [block.text for block in result.content if hasattr(block, "text")]

    assert len(text_blocks) == 1
    assert "总字符" in text_blocks[0]
    assert "中文" in text_blocks[0]
    await exit_stack.aclose()


@pytest.mark.asyncio
async def test_mcp_server_list_resources():
    """验证 resources/list 包含 notes://index。"""
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=None,
    )

    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    await session.initialize()
    response = await session.list_resources()
    uris = [str(r.uri) for r in response.resources]

    assert any("index" in uri for uri in uris)
    await exit_stack.aclose()


@pytest.mark.asyncio
async def test_mcp_server_list_prompts():
    """验证 prompts/list 包含 study_review。"""
    exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[SERVER_PATH],
        env=None,
    )

    stdio_transport = await exit_stack.enter_async_context(stdio_client(server_params))
    stdio, write = stdio_transport
    session = await exit_stack.enter_async_context(ClientSession(stdio, write))

    await session.initialize()
    response = await session.list_prompts()
    prompt_names = [p.name for p in response.prompts]

    assert "study_review" in prompt_names
    await exit_stack.aclose()
