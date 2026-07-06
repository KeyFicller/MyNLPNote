#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MyNLPNote MCP Server — 暴露本仓库 notes/ 目录的 NLP 学习笔记

启动方式（stdio，供 Cursor / Claude Desktop 等 MCP Host 调用）：
    python examples/llm-apps/mcp_nlp_notes_server.py

Cursor 配置见项目根目录 .cursor/mcp.json
"""

from __future__ import annotations

import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTES_ROOT = PROJECT_ROOT / "notes"

VALID_PHASES = (
    "phase1-python",
    "phase2-dl",
    "phase3-nlp",
    "phase4-projects",
    "phase-langchain",
)

mcp = FastMCP("nlp-notes")


def count_words_impl(text: str) -> dict[str, int]:
    """统计文本字数：总字符、中文字符、英文单词数。"""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_words = len(re.findall(r"[A-Za-z]+", text))
    return {
        "total_chars": len(text),
        "chinese_chars": chinese_chars,
        "english_words": english_words,
    }


def list_note_files_impl(phase: str, notes_root: Path = NOTES_ROOT) -> list[str]:
    """列出指定阶段目录下的 Markdown 笔记文件名。"""
    if phase not in VALID_PHASES:
        raise ValueError(f"无效阶段: {phase}，可选: {', '.join(VALID_PHASES)}")
    phase_dir = notes_root / phase
    if not phase_dir.is_dir():
        return []
    return sorted(p.name for p in phase_dir.glob("*.md"))


def search_notes_impl(
    keyword: str,
    phase: str = "all",
    notes_root: Path = NOTES_ROOT,
    max_results: int = 10,
) -> list[dict[str, str]]:
    """在笔记中搜索关键词，返回匹配的文件与摘要行。"""
    if not keyword.strip():
        return []

    phases = VALID_PHASES if phase == "all" else (phase,)
    results: list[dict[str, str]] = []

    for phase_name in phases:
        phase_dir = notes_root / phase_name
        if not phase_dir.is_dir():
            continue
        for note_path in sorted(phase_dir.glob("*.md")):
            content = note_path.read_text(encoding="utf-8")
            if keyword.lower() not in content.lower():
                continue
            snippet = _extract_snippet(content, keyword)
            results.append(
                {
                    "phase": phase_name,
                    "file": note_path.name,
                    "snippet": snippet,
                }
            )
            if len(results) >= max_results:
                return results
    return results


def read_note_impl(
    phase: str,
    filename: str,
    notes_root: Path = NOTES_ROOT,
    max_chars: int = 4000,
) -> str:
    """读取指定笔记内容（截断以防上下文过长）。"""
    if phase not in VALID_PHASES:
        raise ValueError(f"无效阶段: {phase}")
    note_path = notes_root / phase / filename
    if not note_path.is_file():
        raise FileNotFoundError(f"笔记不存在: {phase}/{filename}")
    content = note_path.read_text(encoding="utf-8")
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + f"\n\n... (已截断，共 {len(content)} 字符)"


def _extract_snippet(content: str, keyword: str, context: int = 40) -> str:
    lower = content.lower()
    idx = lower.find(keyword.lower())
    if idx == -1:
        return content[:80].replace("\n", " ")
    start = max(0, idx - context)
    end = min(len(content), idx + len(keyword) + context)
    snippet = content[start:end].replace("\n", " ")
    if start > 0:
        snippet = "..." + snippet
    if end < len(content):
        snippet = snippet + "..."
    return snippet


def build_notes_index(notes_root: Path = NOTES_ROOT) -> str:
    """生成 notes 目录索引文本。"""
    lines = ["# MyNLPNote 笔记索引\n"]
    for phase in VALID_PHASES:
        files = list_note_files_impl(phase, notes_root)
        lines.append(f"## {phase} ({len(files)} 篇)")
        for name in files:
            lines.append(f"- {name}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def count_words(text: str) -> str:
    """统计文本字数，返回总字符数、中文字符数、英文单词数。

    Args:
        text: 待统计的文本内容
    """
    stats = count_words_impl(text)
    return (
        f"总字符: {stats['total_chars']}, "
        f"中文: {stats['chinese_chars']}, "
        f"英文单词: {stats['english_words']}"
    )


@mcp.tool()
def list_note_files(phase: str = "phase4-projects") -> str:
    """列出指定学习阶段的 Markdown 笔记文件。

    Args:
        phase: 阶段目录名，如 phase1-python、phase4-projects
    """
    files = list_note_files_impl(phase)
    if not files:
        return f"阶段 {phase} 下没有找到笔记。"
    return "\n".join(files)


@mcp.tool()
def search_notes(keyword: str, phase: str = "all") -> str:
    """在 notes/ 目录中搜索包含关键词的笔记。

    Args:
        keyword: 搜索关键词
        phase: 限定阶段，默认 all 表示全部阶段
    """
    hits = search_notes_impl(keyword, phase)
    if not hits:
        return f"未找到包含「{keyword}」的笔记。"
    lines = []
    for hit in hits:
        lines.append(f"[{hit['phase']}] {hit['file']}: {hit['snippet']}")
    return "\n".join(lines)


@mcp.tool()
def read_note(phase: str, filename: str) -> str:
    """读取指定笔记的 Markdown 内容。

    Args:
        phase: 阶段目录名
        filename: 笔记文件名，如 07_Function_Calling与Tools使用.md
    """
    return read_note_impl(phase, filename)


@mcp.resource("notes://index")
def notes_index() -> str:
    """返回 MyNLPNote 全部笔记的目录索引。"""
    return build_notes_index()


@mcp.resource("notes://{phase}/{filename}")
def note_resource(phase: str, filename: str) -> str:
    """按 URI 读取单篇笔记（只读资源）。"""
    return read_note_impl(phase, filename)


@mcp.prompt()
def study_review(topic: str) -> str:
    """生成复习某个 NLP 主题的提示词模板。

    Args:
        topic: 要复习的主题，如 Transformer、RAG、Function Calling
    """
    return f"""请帮我复习「{topic}」这个主题。按以下结构回答：

1. **核心概念**：用 3-5 句话解释是什么、解决什么问题
2. **关键术语**：列出 5 个最重要的术语及简短定义
3. **与前后知识的联系**：它和哪些已学内容相关？
4. **自测题**：出 3 道选择题 + 1 道简答题（附参考答案）
5. **实践建议**：在本仓库 examples/ 或 notes/ 中，我应该看哪些文件？

请结合 MyNLPNote 仓库的学习路线来回答。"""


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
