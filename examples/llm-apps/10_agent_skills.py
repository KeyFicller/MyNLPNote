#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第31课：Agent Skills 编写实战
==============================

课程目标：
- 理解 Cursor Agent Skills 是什么、与 MCP / Function Calling 的区别
- 掌握 SKILL.md 的结构与 frontmatter 规范
- 编写可被 Agent 自动发现的 description
- 为本仓库创建项目级 Skill，并用校验脚本自检

运行方式：
    python examples/llm-apps/10_agent_skills.py

相关文件：
    examples/llm-apps/skill_validator.py     — Skill 校验工具
    .cursor/skills/nlp-lesson-helper/      — 本仓库示例 Skill
    notes/phase4-projects/11_Agent_Skills编写实战.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skill_validator import find_skill_dirs, validate_skill_dir

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SKILLS = PROJECT_ROOT / ".cursor" / "skills"
DEMO_SKILLS = Path(__file__).resolve().parent / "skills_demo"

print("=" * 70)
print("第31课：Agent Skills 编写实战")
print("=" * 70)
print()

# =============================================================================
# 第一部分：Skills 是什么？
# =============================================================================

print("第一部分：Agent Skills 是什么？")
print("-" * 70)
print("""
Agent Skills = 写给 AI Agent 的「操作手册」（Markdown + YAML 元数据）

┌─────────────────────────────────────────────────────────────────┐
│  用户提问  →  Agent 匹配 description  →  读取 SKILL.md  →  执行  │
└─────────────────────────────────────────────────────────────────┘

与前面学过的能力对比：

  Function Calling   模型调用单个函数（订单查询、投票等）
  MCP                标准化暴露 Tools/Resources/Prompts，多客户端复用
  Agent Skills       教 Agent「怎么做某类任务」的流程、规范、模板

典型用途：
  - 代码审查 checklist、提交信息格式
  - 项目专属约定（目录结构、命名、测试命令）
  - 领域工作流（写笔记、跑示例、创建 PR）

存放位置：
  个人 Skill   ~/.cursor/skills/<skill-name>/SKILL.md
  项目 Skill   .cursor/skills/<skill-name>/SKILL.md   ← 可随仓库共享
""")

# =============================================================================
# 第二部分：SKILL.md 结构
# =============================================================================

print("第二部分：SKILL.md 基本结构")
print("-" * 70)
print("""
目录结构（渐进式披露）：

  my-skill/
  ├── SKILL.md          # 必需：核心指令（建议 < 500 行）
  ├── reference.md      # 可选：详细参考
  ├── examples.md       # 可选：示例
  └── scripts/          # 可选：可执行脚本

SKILL.md 模板：

---
name: my-skill-name
description: 第三人称描述能力 + 触发场景（WHEN 关键词）
---

# Skill 标题

## 快速开始
1. ...
2. ...

## 检查清单
- [ ] ...

## 示例
输入 / 输出各一例
""")

# =============================================================================
# 第三部分：description 写法
# =============================================================================

print("第三部分：description 写法（决定 Agent 何时加载）")
print("-" * 70)
print("""
✅ 好的 description（第三人称 + WHAT + WHEN）：
  "Guides creation of Cursor Agent Skills with SKILL.md frontmatter.
   Use when authoring skills, writing SKILL.md, or asking about skill structure."

❌ 避免：
  - "我可以帮你写 Skill"（第一人称）
  - "帮助文档"（太模糊，无法匹配）

编写原则：
  1. 简洁 — Agent 已经很聪明，只写它不知道的项目/团队知识
  2. 具体 — 给默认方案，少列一长串可选项
  3. 分层 — 细节放 reference.md，SKILL.md 只保留主流程
""")

# =============================================================================
# 第四部分：校验本仓库 Skill
# =============================================================================

print("第四部分：校验本仓库中的 Skill")
print("-" * 70)

targets = []
if PROJECT_SKILLS.is_dir():
    targets.extend(find_skill_dirs(PROJECT_SKILLS))
if DEMO_SKILLS.is_dir():
    targets.extend(find_skill_dirs(DEMO_SKILLS))

if not targets:
    print("未找到 Skill 目录。请创建：")
    print(f"  {PROJECT_SKILLS / 'nlp-lesson-helper' / 'SKILL.md'}")
else:
    all_ok = True
    for skill_dir in targets:
        rel = skill_dir.relative_to(PROJECT_ROOT)
        result = validate_skill_dir(skill_dir)
        status = "✅" if result.ok else "❌"
        print(f"\n{status} {rel}")
        print(f"   name: {result.meta.get('name', '(缺失)')}")
        desc = result.meta.get("description", "")
        if desc:
            preview = desc[:72] + ("…" if len(desc) > 72 else "")
            print(f"   description: {preview}")
        for err in result.errors:
            print(f"   ERROR: {err}")
            all_ok = False
        for warn in result.warnings:
            print(f"   WARN:  {warn}")

    print()
    if all_ok:
        print("所有 Skill 通过基本校验。")
    else:
        print("存在校验失败项，请按提示修改 SKILL.md。")

# =============================================================================
# 第五部分：动手练习
# =============================================================================

print()
print("第五部分：动手练习")
print("-" * 70)
print("""
练习 1 — 阅读示例 Skill
  打开 .cursor/skills/nlp-lesson-helper/SKILL.md
  在 Cursor 中问：「帮我找 Function Calling 相关笔记并建议运行哪个示例」

练习 2 — 修改 description
  给 nlp-lesson-helper 的 description 增加触发词「RAG」「MCP」
  运行：python examples/llm-apps/10_agent_skills.py  确认仍通过校验

练习 3 — 新建个人 Skill
  在 ~/.cursor/skills/ 下创建 commit-helper/SKILL.md
  规定本仓库 git commit 信息格式（参考 notes 中的风格）

练习 4 — 对比 MCP
  MCP Server 暴露「能做什么」（search_notes）
  Skill 教 Agent「遇到某类问题该怎么一步步做」
  两者可配合：MCP 提供工具，Skill 规定调用顺序与输出格式
""")

print("=" * 70)
print("第31课演示结束。详细笔记见 notes/phase4-projects/11_Agent_Skills编写实战.md")
print("=" * 70)
