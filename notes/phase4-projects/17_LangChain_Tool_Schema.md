# 第37课：LangChain Tool Schema

**项目**: `args_schema` 与 Pydantic / JSON Schema 参数定义  
**技术栈**: LangChain, langchain-core, Pydantic, langchain-deepseek, ChatDeepSeek  
**示例代码**: `examples/llm-apps/16_langchain_tool_schema.py`  
**前置课程**: 第36课 LangChain Tools 工具、第28课 Function Calling 与 Tools  
**环境与运行**：见 [第32课 §1 环境配置](12_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/llm-apps/16_langchain_tool_schema.py`

---

## 课程概述

第36课用 `@tool` 依赖函数签名和 docstring 自动生成参数 Schema；当参数变复杂（枚举、默认值、可选字段、嵌套对象）时，需要**显式声明** `args_schema`。本课演示两种写法：

1. **Pydantic `BaseModel`** — 类型安全、IDE 友好，LangChain 推荐方式  
2. **原生 JSON Schema dict** — 与 OpenAI `parameters` 格式一一对应，适合精细控制

两种写法最终都通过 `convert_to_openai_tool` 转成模型能读的 Tool 定义，再 `bind_tools` 触发调用。

**学习目标：**
1. 理解 `args_schema` 在 `@tool` 中的作用与生成流程
2. 会用 Pydantic `BaseModel` + `Field` + `Literal` 定义工具参数
3. 会手写 JSON Schema dict 作为 `args_schema`
4. 掌握 `required`、`default`、`enum` 对模型填参的影响
5. 会用 `convert_to_openai_tool` 调试、对比两种 Schema 输出

---

## 1. 为什么需要 args_schema？

### 1.1 仅靠函数签名的局限

第36课中，`@tool` 默认从**类型注解 + docstring** 推断 Schema：

```python
@tool
def get_current_time(city: str) -> str:
    """获取当前时间"""
```

简单参数够用；但以下场景推断不足或不可控：

| 需求 | 仅靠签名 | `args_schema` |
|------|----------|---------------|
| 参数有默认值 | 部分支持 | `Field(default=...)` 明确写入 Schema |
| 枚举（摄氏/华氏） | `str` 无法表达 `enum` | `Literal["C","F"]` 或 JSON `enum` |
| 必填 vs 可选 | 依赖 Python 默认值推断 | `required` 数组精确控制 |
| 嵌套对象 / 列表 | 推断粗糙 | Pydantic 模型或完整 JSON Schema |
| 与 OpenAI 格式对齐 | 间接转换 | 可直接手写 `parameters` |

`args_schema` 告诉 LangChain：**发给模型的 parameters 长什么样**，与函数体实际接收的参数应对齐。

### 1.2 数据流

```
args_schema（Pydantic 或 dict）
        │
        ▼
@tool(args_schema=...) 包装函数
        │
        ▼
convert_to_openai_tool(tool)
        │
        ▼
OpenAI 格式 { type, function: { name, description, parameters } }
        │
        ▼
ChatDeepSeek.bind_tools([tool]).invoke(messages)
        │
        ▼
AIMessage.tool_calls[0]["args"]  ← 模型按 Schema 填参
```

---

## 2. Pydantic BaseModel — `get_weather`

### 2.1 定义输入模型

```python
from typing import Literal
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(
        description="城市名称",
        default="北京",
    )
    unit: Literal["C", "F"]
    fore_cast: bool = Field(
        description="是否需要天气预报",
        default=False,
    )
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `city` | `str` + `default="北京"` | 有默认值 → Schema 中通常**不在** `required` |
| `unit` | `Literal["C", "F"]` | 生成 `enum: ["C", "F"]`，且**无默认值** → 必填 |
| `fore_cast` | `bool` + `default=False` | 可选布尔，带描述 |

`Field(description=...)` 会进入 JSON Schema 的 `description`，**模型靠它理解参数语义**（与第28课 `parameters.properties.*.description` 相同）。

### 2.2 绑定到 @tool

```python
@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str, fore_cast: bool = False) -> str:
    """获取某城市天气"""
    return f"当前{city}天气：晴天，温度：20{unit}。 - Forecast: {fore_cast}"
```

注意：
- **函数参数名**必须与 `WeatherInput` 字段名一致（`city`、`unit`、`fore_cast`）
- **函数默认值**宜与 Schema 一致，避免校验通过但运行时行为意外
- `args_schema` 优先于从签名自动推断；docstring 仍可作为 tool 的 `description`

### 2.3 查看生成的 OpenAI Tool

```python
rprint(convert_to_openai_tool(get_weather))
```

典型 `parameters` 片段：

```json
{
  "type": "object",
  "properties": {
    "city": { "type": "string", "description": "城市名称", "default": "北京" },
    "unit": { "type": "string", "enum": ["C", "F"] },
    "fore_cast": { "type": "boolean", "description": "是否需要天气预报", "default": false }
  },
  "required": ["unit"]
}
```

只有 `unit` 在 `required` 里——因为 `city` 和 `fore_cast` 有默认值。用户问「明天上海天气」时，模型可能只传 `{"unit": "C", "city": "上海"}` 或补上 `fore_cast`。

### 2.4 触发调用

```python
messages = [HumanMessage(content="明天上海是什么天气？")]
response = _chat_deepseek().bind_tools([get_weather]).invoke(messages)
rprint(response)
```

检查 `response.tool_calls[0]["args"]` 是否符合 Schema（如 `unit` 是否为 `"C"` 或 `"F"`）。

---

## 3. 原生 JSON Schema dict — `get_weather2`

### 3.1 手写 parameters

```python
json_schema = {
    "properties": {
        "city": {
            "default": "北京",
            "description": "城市名称",
            "type": "string",
        },
        "unit": {"enum": ["C", "F"], "type": "string"},
        "fore_cast": {
            "default": False,
            "description": "是否需要天气预报",
            "type": "boolean",
        },
    },
    "required": ["unit"],
    "type": "object",
}
```

与 Pydantic 生成的结构** deliberately 对齐**，便于对比两种写法的结果是否一致。

> 示例里变量名 `json_schema` 与 `from pydantic import json_schema` 同名，后者会被 dict 覆盖；若需使用 Pydantic 的 `json_schema` 工具函数，建议 dict 改用 `weather_json_schema` 等名称。

### 3.2 绑定到 @tool

```python
@tool(args_schema=json_schema)
def get_weather2(city: str, unit: str, fore_cast: bool = False) -> str:
    """获取某城市天气"""
    return f"当前{city}天气：晴天，温度：20{unit}。 - Forecast: {fore_cast}"
```

LangChain 接受 **Pydantic 模型类** 或 **符合 JSON Schema 的 dict** 作为 `args_schema`。

### 3.3 对比与调用

```python
rprint(convert_to_openai_tool(get_weather2))
response = _chat_deepseek().bind_tools([get_weather2]).invoke(messages)
```

`get_weather` 与 `get_weather2` 的 `convert_to_openai_tool` 输出应高度相似；差异通常只在字段顺序或 Pydantic 版本细节。

---

## 4. 两种写法如何选择？

| 维度 | Pydantic `BaseModel` | JSON Schema `dict` |
|------|----------------------|---------------------|
| 类型检查 | ✅ 编译/运行时校验 | ❌ 需自行保证合法 |
| IDE 补全 | ✅ | ❌ |
| 与 OpenAI 格式 | 自动转换 | 已是目标格式 |
| 复杂嵌套 | 模型嵌套更清晰 | 手写易错 |
| 动态 Schema | 需运行时建模型 | dict 更灵活 |
| 推荐场景 | **默认首选** | 迁移旧 Schema、代码生成、极细控制 |

实践建议：**新工具用 Pydantic**；从 OpenAPI / 既有 JSON 粘贴时用 dict，再逐步迁到 Pydantic。

---

## 5. Schema 设计要点

### 5.1 description 是关键

模型**不执行**你的 Python 类型，只读 JSON Schema 文本。`Field(description="...")` 或 `properties.*.description` 要写清：
- 参数含义（「城市名称」而非「city」）
- 格式约束（日期格式、单位含义）
- 何时需要该参数（如 `fore_cast`：是否需要多日预报）

### 5.2 required 与 default

```python
# 有 default → 一般不必列入 required
city: str = Field(default="北京", ...)

# 无 default → 列入 required
unit: Literal["C", "F"]
```

`required: ["unit"]` 表示模型**必须**提供 `unit`；`city` 可省略，运行时 LangChain/Pydantic 用默认值 `"北京"`。

### 5.3 枚举：Literal vs enum

```python
# Pydantic
unit: Literal["C", "F"]

# JSON Schema
"unit": {"type": "string", "enum": ["C", "F"]}
```

避免用裸 `str` 表示枚举，否则模型可能传入 `"celsius"` 等非法值。

### 5.4 函数签名与 Schema 一致

`args_schema` 定义的字段，函数应能按名接收；类型不匹配的调用可能在 `tool.invoke` 时由 Pydantic 校验失败。改 Schema 时**同步改函数参数**。

---

## 6. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `BaseModel` | 声明工具输入结构 | `WeatherInput` |
| `Field(...)` | 默认值、描述、约束 | `city`, `fore_cast` |
| `Literal["C","F"]` | 枚举类型 | `unit` |
| `@tool(args_schema=...)` | 显式绑定 Schema | `get_weather`, `get_weather2` |
| `convert_to_openai_tool` | 查看最终 Tool JSON | `main()` 开头两次 `rprint` |
| `bind_tools` + `invoke` | 触发模型按 Schema 填参 | `main()` |

---

## 7. 与前面课程的关系

```
第28课 Function Calling     →  手写 parameters JSON Schema ✅ 原理
第36课 @tool 基础           →  签名 + docstring 自动推断
第37课 args_schema          →  Pydantic / dict 显式 Schema ✅ 你在这里
第29课 智能客服              →  多工具、复杂业务参数
Agent / LCEL                →  tools 列表统一走同一套 Schema
```

本课补齐 **Tool 参数契约层**：生产里工具一多，靠 Pydantic 集中管理输入比散落 docstring 更可维护。

---

## 8. 常见问题

### Q1: 有 `args_schema` 还要写函数参数吗？

要。`@tool` 仍包装可调用函数；`args_schema` 只管**告诉模型怎么填参**，执行时仍调用 `get_weather(city=..., unit=..., ...)`。

### Q2: Pydantic 校验失败会怎样？

`tool.invoke` 时若 `args` 不符合 `WeatherInput`，会抛校验错误。可在 Agent 层捕获并提示模型重试。

### Q3: `required` 里写了 `city` 又有 `default` 矛盾吗？

JSON Schema 中部分实现允许；LangChain/Pydantic 一般以「有 default 则非必填」为准。示例 intentionally 只 `required: ["unit"]`。

### Q4: 能否只用 `args_schema` 不要函数体？

需要 callable。可写薄包装函数，或改用 `StructuredTool.from_function` / `StructuredTool(name=..., args_schema=..., func=...)`。

### Q5: 嵌套对象怎么写？

Pydantic 嵌套模型即可，例如 `class Address(BaseModel): ...` 再 `location: Address`。JSON Schema 用 `properties` 嵌套 `type: object`。

### Q6: `convert_to_openai_tool` 与 `args_schema` 关系？

`args_schema` 是输入；`convert_to_openai_tool` 把完整 Tool（含 name、description、parameters）序列化为 OpenAI 请求格式，便于调试 diff。

---

## 9. 动手练习

1. **打印对比**：运行脚本，并排比较 `get_weather` 与 `get_weather2` 的 `parameters`，确认 `required` / `enum` / `default` 一致
2. **改必填**：把 `unit` 也加上 `default="C"`，观察 `required` 是否变为空数组，再问同一问题看 `args` 变化
3. **加字段**：在 `WeatherInput` 增加 `days: int = Field(description="预报天数", default=1, ge=1, le=7)`，同步改函数返回值
4. **非法枚举**：临时去掉 `Literal`，用 `unit: str`，看模型是否仍会乱传单位
5. **接第36课闭环**：对 `response.tool_calls[0]` 执行 `get_weather.invoke(...)`，再 `invoke` 第二次拿到自然语言回答
6. **类型判断**：对 `args_schema` 用 `isinstance(WeatherInput, type)` 与 `isinstance(json_schema, dict)` 区分两种定义（呼应 Python 类型判断）

---

## 10. 参考

- 示例代码：`examples/llm-apps/16_langchain_tool_schema.py`
- 前置笔记：`notes/phase4-projects/16_LangChain_Tools工具.md`
- Function Calling Schema：`notes/phase4-projects/07_Function_Calling与Tools使用.md`（§2 Tools Schema 设计）
- Pydantic Field：[Pydantic 文档](https://docs.pydantic.dev/latest/concepts/fields/)
- LangChain Tools：[Tools 文档](https://python.langchain.com/docs/concepts/tools/)

---

*完成本课后，你已掌握用 `args_schema` 精确定义工具参数契约：Pydantic 负责可维护的结构化输入，JSON Schema dict 负责与 OpenAI 格式无缝对齐。这是复杂 Agent 与多工具系统的必备技能。*
