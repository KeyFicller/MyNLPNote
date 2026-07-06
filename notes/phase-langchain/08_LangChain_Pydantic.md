# 第39课：LangChain Pydantic 结构化输出

**项目**: `with_structured_output` 与 Pydantic 模型约束 LLM 返回  
**技术栈**: LangChain, langchain-core, Pydantic, langchain-deepseek, ChatDeepSeek  
**示例代码**: `examples/langchain/08_langchain_pydantic.py`  
**前置课程**: 第37课 Tool Schema、第32课 LangChain 进阶与 DeepSeek 接入  
**环境与运行**：见 [第32课 §1 环境配置](01_LangChain进阶与DeepSeek接入.md#1-环境配置)；本课 `python examples/langchain/08_langchain_pydantic.py`（`main()` 中注释切换六个演示）

---

## 课程概述

第37课用 Pydantic `BaseModel` 定义**工具入参**（`args_schema`），告诉模型「调用工具时该怎么填参」。本课转向**模型输出侧**：用 `llm.with_structured_output(PydanticModel)` 约束 LLM 的回复必须是符合 Schema 的结构化对象，而不是自由文本。

示例从最简单的三字段人物信息，逐步扩展到默认值、`Optional`、枚举、`Literal`、列表、嵌套模型，以及 Pydantic 校验约束（`min_length` / `ge` / `le`）与 `ValidationError` 处理。所有演示均通过 `deepseek_client.chat_deepseek()` 调用 DeepSeek，并在 `invoke` 时关闭 thinking 模式。

**学习目标：**
1. 理解 `with_structured_output` 与 `bind_tools` / `args_schema` 的分工
2. 会用 `BaseModel` + `Field(description=...)` 定义结构化输出 Schema
3. 掌握默认值、`Optional`、枚举、`Literal` 在输出 Schema 中的写法与模型行为
4. 会用嵌套 Pydantic 模型表达复杂对象（公司 + 地址）
5. 了解 Pydantic 字段约束与 LLM 输出校验失败时的处理方式

---

## 1. 为什么需要结构化输出？

### 1.1 自由文本 vs 结构化对象

| 场景 | 自由 `invoke` | `with_structured_output` |
|------|---------------|--------------------------|
| 聊天、写作 | ✅ 自然语言 | 过度约束 |
| 信息抽取（人名、年龄、城市） | 需正则/二次解析 | ✅ 直接得 `BaseModel` 实例 |
| 下游 API / 数据库写入 | 易格式漂移 | ✅ 类型明确、可校验 |
| 多字段表单、订单、简历解析 | 不稳定 | ✅ Schema 即契约 |

第28课 Function Calling 让模型**发起工具调用**；本课让模型**直接产出结构化数据**（内部常借助 JSON mode / function calling 实现，LangChain 封装为 `with_structured_output`）。

### 1.2 与第37课的区别

```
第37课 args_schema          →  Tool 的「输入参数」Schema（模型填参后你执行函数）
第39课 with_structured_output →  LLM 本轮「最终输出」Schema（直接解析为 Pydantic 对象）
```

二者都用 Pydantic，但**数据流方向相反**：前者是工具契约，后者是抽取/生成结果契约。

### 1.3 数据流

```
Pydantic BaseModel（输出 Schema）
        │
        ▼
chat_deepseek().with_structured_output(Model)
        │
        ▼
invoke([HumanMessage("张三是一名30岁的程序员")])
        │
        ▼
PydanticTest(name='张三', age=30, work='程序员')   ← 已是 Python 对象，非 JSON 字符串
```

---

## 2. 基础用法 — `PydanticTest`

### 2.1 定义输出模型

```python
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

class PydanticTest(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    work: str = Field(description="The work of the person")
```

| 要点 | 说明 |
|------|------|
| `Field(description=...)` | 写入 JSON Schema，**引导模型**理解各字段语义（与第37课相同） |
| 无 `default` | 三个字段在 Schema 中均为必填，模型应尽量从文本中抽取 |
| 返回类型 | `invoke` 返回的是 **`PydanticTest` 实例**，可用 `.name`、`.model_dump()` 等 |

### 2.2 绑定并调用

```python
llm = chat_deepseek().with_structured_output(PydanticTest)
messages = [HumanMessage(content="张三是一名30岁的程序员")]
response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
```

典型输出（`rich.print`）：

```
PydanticTest(name='张三', age=30, work='程序员')
```

`with_structured_output` 在底层将 Pydantic 模型转为 JSON Schema，要求模型按 Schema 生成 JSON，再反序列化为 Python 对象。

---

## 3. 默认值与可选字段 — `PydanticTest2`

```python
from typing import Optional

class PydanticTest2(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(default=18, description="The age of the person")
    work: str = Field(description="The work of the person")
    city: Optional[str] = Field(description="The city of the person")
```

### 3.1 信息不全时

输入：`"张三是一名的程序员"`（未提及年龄与城市）

| 字段 | 模型行为 |
|------|----------|
| `name` | 抽取为「张三」 |
| `age` | 无信息 → 使用 `default=18` |
| `work` | 抽取为「程序员」 |
| `city` | `Optional`，无信息 → `None` |

### 3.2 信息完整时

输入：`"李四是一名销售，在杭州工作"`

模型应填 `name='李四'`、`work='销售'`、`city='杭州'`，`age` 仍可能默认 18（未提及年龄时）。

**设计原则**：能从文本推断的字段不要加 default；只有「允许缺失」的字段才用 `default` 或 `Optional`。

---

## 4. 枚举与城市 Literal — `PydanticTest3`

### 4.1 性别枚举

```python
from enum import Enum

class Gender(str, Enum):
    MALE = "男"
    FEMALE = "女"
    UNKNOWN = "未知"

class PydanticTest3(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    gender: Gender = Field(default=Gender.UNKNOWN, description="The gender of the person")
    city: Literal["北京", "上海", ..., "台湾"] = Field(default="北京", description="The city of the person")
```

| 类型 | 作用 |
|------|------|
| `Gender(str, Enum)` | 限制性别为枚举值，生成 JSON Schema `enum` |
| `Literal["北京", "上海", ...]` | 城市白名单，防止模型返回「魔都」等非标准名 |
| `default=Gender.UNKNOWN` / `default="北京"` | 文本未提及时的后备值 |

> **注意**：示例源码中 `MALE = "男",` 若多写逗号会变成元组 `("男",)`，枚举值异常。应写 `MALE = "男"`。

### 4.2 推理型抽取

输入：`"李四是一名30岁的程序员,在沿海地区经济最发达的地区工作"`

模型需**推理**「经济最发达」→ `上海`（或 Schema 白名单中的其它合理城市），并填入 `city`。`Literal` 白名单越大，模型越难「胡说」城市名，但也增加 token 与选择成本。

输入：`"张三是一名30岁的女程序员"` → `gender=Gender.FEMALE`。

---

## 5. 列表字段 — `PydanticTest4`

```python
class PydanticTest4(BaseModel):
    elements: list[str] = Field(description="The elements of the list")
```

输入：`"请列举出地球上最常见的五种化学元素"`

返回示例：

```
PydanticTest4(elements=['氧', '硅', '铝', '铁', '钙'])
```

适用于：标签列表、要点摘要、多实体抽取等。复杂场景可用 `list[SomeModel]` 嵌套对象列表。

---

## 6. 嵌套模型 — `PydanticTest5`

```python
class Address(BaseModel):
    """地址信息"""
    city: str = Field(description="The city of the address")
    district: str = Field(description="The district of the address")

class Company(BaseModel):
    """公司信息"""
    name: str = Field(description="The name of the company")
    address: Address = Field(description="The address of the company")
```

```python
llm = chat_deepseek().with_structured_output(Company)
messages = [HumanMessage(content="小米科技有限公司总部的地址是什么")]
response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
```

典型结果结构：

```
Company(
    name='小米科技有限责任公司',
    address=Address(city='北京', district='海淀区')
)
```

嵌套 `BaseModel` 会生成嵌套 JSON Schema（`address` 为 `type: object`）。与第37课 Q5「工具参数嵌套」写法相同，只是用途从**入参**变为**出参**。

---

## 7. 校验约束与 ValidationError — `PydanticTest6`

### 7.1 本地构造校验

```python
from pydantic import ValidationError

class User(BaseModel):
    name: str = Field(description="The name of the user", min_length=2, max_length=10)
    age: int = Field(description="The age of the user", ge=12, le=27)

try:
    user = User(name="雷军1234", age=30)
except ValidationError as e:
    rprint(e)
```

| 约束 | 含义 | 示例违规 |
|------|------|----------|
| `min_length=2, max_length=10` | 姓名字符数 2–10 | `"雷军1234"` 超长 |
| `ge=12, le=27` | 年龄 ≥12 且 ≤27 | `age=30` 超上限 |

本地 `User(...)` 会立即抛 `ValidationError`；字段错误详情在 `e.errors()` 中。

### 7.2 LLM 输出后的校验

```python
llm = chat_deepseek().with_structured_output(User)
messages = [HumanMessage(content="介绍雷军")]
try:
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)
except ValidationError as e:
    rprint(e)
    print("用户信息验证失败")
```

对「介绍雷军」这类问题，模型可能返回 `name='雷军'`、`age=50+` 等**不符合** `ge=12, le=27` 的值；`with_structured_output` 在解析阶段触发 Pydantic 校验，抛 `ValidationError`。

**生产建议**：
- 约束要与业务一致（名人介绍不必限制 `age≤27` 时，应放宽 Schema）
- 捕获 `ValidationError` 后：重试、降级为宽松 Schema、或提示用户改写问题
- 关键路径可加 `model_validate` 二次确认

---

## 8. DeepSeek 与 thinking 模式

示例中统一使用：

```python
extra_body={"thinking": {"type": "disabled"}}
```

| 项 | 说明 |
|----|------|
| 原因 | 结构化输出依赖稳定的 JSON 形态；thinking 模式可能影响格式或与部分 API 特性冲突（参见第38课） |
| 配置 | 通过 `invoke` 的 `extra_body` 透传，与 `chat_deepseek()` 工厂函数配合 |
| 模型 | `deepseek_client.py` 默认 `MODEL = "deepseek-v4-pro"` |

---

## 9. 核心 API 对照

| API | 作用 | 本课出现位置 |
|-----|------|-------------|
| `BaseModel` | 声明输出结构 | 全部 `PydanticTest*` |
| `Field(...)` | 描述、默认值、约束 | 各模型字段 |
| `Optional[str]` | 可空字段 | `PydanticTest2.city` |
| `Literal[...]` | 字符串枚举白名单 | `PydanticTest3.city` |
| `Enum` | 命名枚举 | `Gender` |
| `list[str]` | 字符串列表 | `PydanticTest4.elements` |
| 嵌套 `BaseModel` | 复杂对象 | `Company` / `Address` |
| `with_structured_output(Model)` | 绑定输出 Schema | 各 `pydantic_test*` |
| `ValidationError` | 校验失败异常 | `pydantic_test6` |
| `chat_deepseek()` | DeepSeek 客户端 | `deepseek_client.py` |

---

## 10. 与前面课程的关系

```
第28课 Function Calling     →  结构化 JSON 概念
第37课 args_schema          →  Pydantic 定义 Tool 入参 ✅ 同语法、不同方向
第38课 tool_choice          →  控制工具调用策略
第39课 with_structured_output →  Pydantic 定义 LLM 出参 ✅ 你在这里
RAG / Agent                 →  抽取实体、填槽、生成 API 请求体
```

典型组合：**RAG 检索文档** → **`with_structured_output` 抽取字段** → **写入数据库或调用下游 API**；与 Agent 工具链互补（工具负责「动作」，结构化输出负责「形状固定的数据」）。

---

## 11. 常见问题

### Q1: `with_structured_output` 和 `bind_tools` 能一起用吗？

可以，但是不同目的：`bind_tools` 让模型发起工具调用；`with_structured_output` 让模型直接返回指定类型。同一轮通常二选一为主；复杂 Agent 可在不同步骤分别使用。

### Q2: 返回的是 dict 还是 BaseModel？

默认是 **Pydantic 模型实例**（本课行为）。部分版本支持 `with_structured_output(Model, method="json_mode")` 等参数，以 LangChain 文档为准。

### Q3: `description` 必须用英文吗？

不必，但需与模型能力匹配。示例用英文 description；生产中文业务可写 `Field(description="用户姓名")`，往往更利于中文抽取。

### Q4: 模型填了 Schema 外的城市怎么办？

`Literal` 白名单 + Pydantic 校验会在解析时失败。可缩小白名单、在 prompt 中强调、或捕获错误后重试。

### Q5: 嵌套很深会失败吗？

过深嵌套增加模型出错率。可拆成多步：先抽公司名，再单独抽地址；或放宽中间步骤 Schema。

### Q6: 与第37课 `WeatherInput` 能复用同一个模型吗？

技术上可以，但**入参 Schema**（工具）与**出参 Schema**（抽取）语义不同，建议分模型定义，避免改工具参数时破坏输出契约。

### Q7: `ValidationError` 后如何自动重试？

```python
for attempt in range(3):
    try:
        return llm.invoke(messages)
    except ValidationError:
        messages.append(HumanMessage(content="请严格按字段约束重新生成"))
```

或使用 LangChain 内置 retry / fallback 链路（视版本而定）。

---

## 12. 动手练习

1. **六段演示**：取消 `main()` 中各函数注释，依次运行 `pydantic_test` ~ `pydantic_test6`，观察 `rich` 输出
2. **改默认值**：将 `PydanticTest2.age` 的 `default` 改为 `0`，对同一句「张三是一名的程序员」看 `age` 变化
3. **缩小 Literal**：将 `PydanticTest3.city` 只保留北上广深，用「在杭州工作」测试是否校验失败或模型改填其它城市
4. **嵌套扩展**：为 `Company` 增加 `founded_year: int`，问「腾讯成立于哪一年，总部在哪」
5. **放宽 User**：去掉 `age` 的 `le=27`，再「介绍雷军」，确认不再 `ValidationError`
6. **对接业务**：设计 `OrderExtract`（订单号、金额、状态），从一段客服对话文本中抽取并 `model_dump_json()` 写入文件
7. **对比第37课**：同一 `Address` 模型，分别用于 `args_schema` 工具入参与本课 `with_structured_output` 出参，打印两种场景下的 JSON Schema（`model_json_schema()`）

---

## 13. 参考

- 示例代码：`examples/langchain/08_langchain_pydantic.py`
- 公共客户端：`examples/langchain/deepseek_client.py`
- 前置笔记：`notes/phase-langchain/06_LangChain_Tool_Schema.md`（Pydantic 字段与 Schema）
- 环境配置：`notes/phase-langchain/01_LangChain进阶与DeepSeek接入.md`
- LangChain Structured Output：[Structured output](https://python.langchain.com/docs/how_to/structured_output/)
- Pydantic 校验：[Validators](https://docs.pydantic.dev/latest/concepts/validators/)

---

*完成本课后，你已能用 Pydantic 为 LLM 定义「输出契约」，并通过 `with_structured_output` 直接得到类型安全的 Python 对象：从简单字段到枚举、列表与嵌套结构，再到校验失败处理。结合第37课的入参 Schema 与第36–38课的工具链，结构化数据在 LangChain 中的「进（工具）—出（抽取）」两条主线均已打通。*
