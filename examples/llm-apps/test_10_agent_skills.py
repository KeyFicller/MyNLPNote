#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第31课 Skill 校验工具单元测试"""

from pathlib import Path

import pytest

from skill_validator import parse_frontmatter, validate_skill_dir

EXAMPLES_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXAMPLES_DIR.parents[1]
PROJECT_SKILL = PROJECT_ROOT / ".cursor" / "skills" / "nlp-lesson-helper"


def test_parse_frontmatter():
    text = """---
name: demo-skill
description: Does something useful for tests.
---

# Body
"""
    meta = parse_frontmatter(text)
    assert meta["name"] == "demo-skill"
    assert "useful" in meta["description"]


def test_validate_project_skill():
    result = validate_skill_dir(PROJECT_SKILL)
    assert PROJECT_SKILL.is_dir(), "示例 Skill 目录应存在"
    assert result.ok, result.errors
    assert result.meta.get("name") == "nlp-lesson-helper"


def test_validate_missing_skill_md(tmp_path):
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir()
    result = validate_skill_dir(skill_dir)
    assert not result.ok
    assert any("SKILL.md" in e for e in result.errors)


def test_validate_invalid_name(tmp_path):
    skill_dir = tmp_path / "Bad_Name"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Bad_Name\ndescription: x\n---\n\nbody\n",
        encoding="utf-8",
    )
    result = validate_skill_dir(skill_dir)
    assert not result.ok
