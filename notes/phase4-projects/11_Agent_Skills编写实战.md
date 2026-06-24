# 第31课：Agent Skills 编写实战

## 课程概述

Cursor **Agent Skills** 是用 Markdown 编写的「Agent 操作手册」，教 AI 在特定场景下按项目约定执行任务。本课在 MCP 之后学习，完成「工具暴露 → 流程规范」的最后一环。

**学习目标：**
1. 理解 Skills 与 Function Calling、MCP 的分工
2. 掌握 `SKILL.md` 的 frontmatter 与正文结构
3. 写出可被 Agent 匹配的 `description`
4. 为本仓库创建项目级 Skill 并用校验脚本自检

---

## 1. 三种「扩展 Agent」方式对比

| 能力 | 解决什么问题 | 本仓库示例 |
|------|--------------|------------|
| **Function Calling** | 模型调用单个函数 | 智能客服查订单、谁是卧底投票 |
| **MCP** | 标准化暴露工具/资源，多客户端复用 | `mcp_nlp_notes_server.py` |
| **Agent Skills** | 教 Agent 某类任务的步骤、格式、约定 | `.cursor/skills/nlp-lesson-helper/` |

```
用户问题
   ↓
Agent 匹配 Skill.description（是否加载 SKILL.md）
   ↓
按 Skill 中的步骤执行（可能再调用 MCP Tools / 本地脚本）
   ↓
按 Skill 要求的格式输出
```

---

## 2. 目录与存放位置

```
skill-name/
├── SKILL.md          # 必需
├── reference.md      # 可选，详细文档
├── examples.md       # 可选
└── scripts/          # 可选，可执行脚本
```

| 类型 | 路径 | 作用域 |
|------|------|--------|
| 个人 Skill | `~/.cursor/skills/<name>/` | 所有项目 |
| 项目 Skill | `.cursor/skills/<name>/` | 当前仓库（可提交 Git） |

**注意：** 不要写入 `~/.cursor/skills-cursor/`（Cursor 内置 Skill 目录）。

---

## 3. SKILL.md 结构

### 3.1 Frontmatter（YAML）

```yaml
---
name: nlp-lesson-helper
description: >-
  Guides work on the MyNLPNote learning repo. Use when the user asks about
  RAG, Agent, MCP, Skills, or llm-apps examples.
---
```

| 字段 | 要求 |
|------|------|
| `name` | 小写字母、数字、连字符，≤64 字符，建议与目录名一致 |
| `description` | 非空，≤1024 字符，**第三人称**，包含 WHAT + WHEN |

### 3.2 正文建议章节

1. **快速开始** — 3～5 步主流程
2. **示例/对照表** — 输入输出或文件映射
3. **检查清单** — 质量关键项
4. **附加资源** — 链接 `reference.md`（一层深度）

主文件建议 **< 500 行**；细节放到附属文件。

---

## 4. description 写法

Agent 靠 `description` 决定是否加载 Skill。

**✅ 推荐：**
- 第三人称：`Guides…` / `Processes…`
- 写清触发词：`RAG`, `MCP`, `SKILL.md`, `commit message`
- 说明场景：`Use when the user asks…`

**❌ 避免：**
- 「我可以帮你…」（第一人称）
- 「帮助文档」（太泛）

---

## 5. 编写原则

1. **简洁** — 只写 Agent 不知道的项目/团队知识
2. **默认方案** — 少给「A 或 B 或 C…」式选项
3. **渐进披露** — SKILL.md 主流程，细节外置
4. **可验证** — 配合脚本或 checklist 做反馈环

常见模式：
- **Template** — 固定输出格式
- **Workflow** — 分步 checklist
- **Conditional** — 按情况分支
- **Scripts** — 复杂操作用 `scripts/` 下的 Python

---

## 6. 本仓库示例 Skill

路径：`.cursor/skills/nlp-lesson-helper/SKILL.md`

作用：在本仓库提问时，引导 Agent：
- 查 `notes/` 哪一篇
- 跑 `examples/llm-apps/` 哪个文件
- 与 MCP 笔记 Server 如何配合

---

## 7. 校验与测试

### 7.1 课程演示

```bash
python examples/llm-apps/10_agent_skills.py
```

### 7.2 单元测试

```bash
python -m pytest examples/llm-apps/test_10_agent_skills.py -v
```

`skill_validator.py` 检查：
- 是否存在 `SKILL.md`
- `name` / `description` 格式
- 行数与正文长度警告

---

## 8. 动手练习

1. **阅读** `.cursor/skills/nlp-lesson-helper/SKILL.md`
2. **修改** description，增加你关心的触发词，再跑校验
3. **新建** 个人 Skill：`~/.cursor/skills/commit-helper/`，规定 commit 格式
4. **对比** MCP：`search_notes` 是能力；Skill 是「先搜笔记再建议示例」的流程

---

## 9. 与前面课程的关系

```
第28课 Function Calling  →  单次工具调用
第30课 MCP               →  工具标准化、可复用
第31课 Agent Skills      →  Agent 行为与项目规范
```

---

## 10. 参考

- Cursor Skill 官方思路：见 Cursor 文档 / Settings → Rules → Agent Skills
- 本课代码：`examples/llm-apps/10_agent_skills.py`
- MCP 笔记 Server：`examples/llm-apps/mcp_nlp_notes_server.py`

---

*完成本课后，你已掌握 LLM 应用开发常见扩展方式：RAG、Agent、Function Calling、MCP、Skills。*
