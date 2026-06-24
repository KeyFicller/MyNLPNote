#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cursor Agent Skill 文件校验工具（供第31课使用）"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

NAME_PATTERN = re.compile(r"^[a-z0-9-]{1,64}$")
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
MAX_SKILL_LINES = 500
MAX_DESCRIPTION_LEN = 1024


@dataclass
class SkillValidationResult:
    skill_dir: Path
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)

    def add_error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


def parse_frontmatter(text: str) -> dict[str, str]:
    """解析 SKILL.md YAML frontmatter（支持单行与 >- 折叠多行）。"""
    match = FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}
    meta: dict[str, str] = {}
    lines = match.group(1).splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value in (">-", ">", "|", "|-", "|-"):
            block: list[str] = []
            i += 1
            while i < len(lines) and lines[i].startswith("  "):
                block.append(lines[i].strip())
                i += 1
            meta[key] = " ".join(block).strip()
            continue
        meta[key] = value.strip('"').strip("'")
        i += 1
    return meta


def validate_skill_dir(skill_dir: Path) -> SkillValidationResult:
    """校验单个 Skill 目录结构。"""
    result = SkillValidationResult(skill_dir=skill_dir)

    skill_md = skill_dir / "SKILL.md"
    if not skill_dir.is_dir():
        result.add_error("路径不是目录")
        return result
    if not skill_md.is_file():
        result.add_error("缺少 SKILL.md")
        return result

    text = skill_md.read_text(encoding="utf-8")
    lines = text.splitlines()
    if len(lines) > MAX_SKILL_LINES:
        result.add_warning(f"SKILL.md 超过 {MAX_SKILL_LINES} 行（当前 {len(lines)} 行）")

    meta = parse_frontmatter(text)
    result.meta = meta

    name = meta.get("name", "")
    description = meta.get("description", "")

    if not name:
        result.add_error("frontmatter 缺少 name")
    elif not NAME_PATTERN.match(name):
        result.add_error("name 只能用小写字母、数字、连字符，且不超过 64 字符")
    elif name != skill_dir.name:
        result.add_warning(f"name '{name}' 与目录名 '{skill_dir.name}' 不一致")

    if not description:
        result.add_error("frontmatter 缺少 description")
    elif len(description) > MAX_DESCRIPTION_LEN:
        result.add_error(f"description 超过 {MAX_DESCRIPTION_LEN} 字符")
    elif len(description) < 20:
        result.add_warning("description 过短，建议写清 WHAT + WHEN")

    body = FRONTMATTER_PATTERN.sub("", text, count=1).strip()
    if len(body) < 80:
        result.add_warning("正文过短，建议补充步骤、示例或检查清单")

    return result


def find_skill_dirs(root: Path) -> list[Path]:
    """查找 root 下所有含 SKILL.md 的子目录。"""
    if not root.is_dir():
        return []
    dirs: list[Path] = []
    for skill_md in sorted(root.rglob("SKILL.md")):
        dirs.append(skill_md.parent)
    return dirs
