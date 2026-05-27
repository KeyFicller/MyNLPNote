# 第28课：Function Calling 与 Tools 使用

## 课程概述

Function Calling（函数调用）是大语言模型（LLM）的核心能力之一，它使模型能够调用外部函数或 API 来获取信息、执行计算或与外部系统交互。本课将深入讲解 Function Calling 的原理、实现和最佳实践。

**学习目标：**
1. 理解 Function Calling 的核心概念和工作流程
2. 掌握 Tools 的定义和 Schema 设计
3. 学会使用 OpenAI Function Calling API
4. 实现支持工具调用的智能 Agent
5. 将 Function Calling 与现有 Agent 框架集成

---

## 1. 什么是 Function Calling？

### 1.1 概念定义

Function Calling 是一种让大语言模型在生成回答的过程中，能够识别出何时需要调用外部工具（函数/API），并以结构化格式输出调用请求的能力。

**核心价值：**
- **扩展知识边界**：获取实时信息（天气、股价、新闻）
- **精确计算能力**：解决 LLM 数学计算不准确的问题
- **系统交互能力**：操作数据库、调用 API、控制设备
- **增强实用性**：从"聊天工具"进化为"行动代理"

### 1.2 工作流程

```
用户提问
    │
    ▼
大模型分析意图
    │
    ▼
判断是否需要工具 ◄──── 是 ────▶ 返回工具调用请求（JSON）
    │                              │
    │ 否                           ▼
    │                        执行本地函数
    ▼                              │
直接生成回答                       ▼
                              获得执行结果
                                   │
                                   ▼
                              大模型根据结果生成回答
                                   │
                                   ▼
                              返回给用户
```

### 1.3 为什么 LLM 需要 Function Calling？

| LLM 局限 | Function Calling 解决方案 |
|---------|------------------------|
| 知识有截止日期 | 调用搜索引擎/API 获取实时信息 |
| 数学计算不准确 | 调用计算器工具获得精确结果 |
| 无法访问私有数据 | 查询数据库或内部 API |
| 无法执行实际操作 | 调用系统命令或第三方服务 |
| 容易产生幻觉 | 基于真实数据生成回答 |

---

## 2. Tools Schema 设计

### 2.1 Schema 结构（OpenAI 格式）

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的天气信息",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称，如：北京、上海"
        },
        "date": {
          "type": "string",
          "description": "日期，格式 YYYY-MM-DD"
        }
      },
      "required": ["location"]
    }
  }
}
```

### 2.2 字段说明

| 字段 | 类型 | 说明 | 最佳实践 |
|------|------|------|---------|
| `type` | string | 固定值 `"function"` | - |
| `function.name` | string | 函数名称 | 使用小写字母和下划线，如 `get_weather` |
| `function.description` | string | 函数功能描述 | **关键！** LLM 根据此描述决定何时调用 |
| `function.parameters` | object | 参数定义（JSON Schema） | 详细定义每个参数的类型和说明 |
| `parameters.required` | array | 必需参数列表 | 明确哪些参数必须提供 |

### 2.3 参数类型支持

```json
{
  "type": "object",
  "properties": {
    "string_param": {"type": "string", "description": "字符串参数"},
    "integer_param": {"type": "integer", "description": "整数参数"},
    "number_param": {"type": "number", "description": "浮点数参数"},
    "boolean_param": {"type": "boolean", "description": "布尔参数"},
    "array_param": {
      "type": "array",
      "items": {"type": "string"},
      "description": "字符串数组"
    },
    "object_param": {
      "type": "object",
      "properties": {
        "key1": {"type": "string"},
        "key2": {"type": "integer"}
      },
      "description": "嵌套对象"
    },
    "enum_param": {
      "type": "string",
      "enum": ["option1", "option2", "option3"],
      "description": "枚举类型参数"
    }
  },
  "required": ["string_param"]
}
```

### 2.4 常见 Tools 类型

| 类别 | 示例 | 用途 |
|------|------|------|
| **信息查询** | `get_weather`, `search_news` | 获取实时信息 |
| **计算工具** | `calculator`, `unit_converter` | 精确计算 |
| **数据处理** | `query_database`, `read_file` | 访问结构化数据 |
| **系统操作** | `send_email`, `create_calendar_event` | 执行实际操作 |
| **外部 API** | `call_third_party_api` | 集成第三方服务 |

---

## 3. Function Calling 实现详解

### 3.1 OpenAI API 调用流程

```python
import openai
import json

# 1. 定义工具
functions = [
    {
        "name": "get_weather",
        "description": "获取城市天气",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "date": {"type": "string"}
            },
            "required": ["location"]
        }
    }
]

# 2. 第一次调用：让模型决定是否调用工具
response = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    functions=functions,  # 提供可用工具
    function_call="auto"  # 让模型自动决定
)

# 3. 解析响应
message = response.choices[0].message

if message.function_call:
    # 模型决定调用工具
    function_name = message.function_call.name
    function_args = json.loads(message.function_call.arguments)
    
    # 4. 执行工具函数
    result = get_weather(**function_args)
    
    # 5. 第二次调用：将结果返回给模型
    second_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
            message,  # 助手消息（工具调用请求）
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(result)
            }
        ]
    )
    
    print(second_response.choices[0].message.content)
```

### 3.2 关键 API 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `functions` | array | 可用工具列表 |
| `function_call` | string/object | 工具调用控制方式 |

**function_call 选项：**
- `"auto"`：让模型自动决定是否调用（默认）
- `"none"`：不调用任何工具
- `{"name": "function_name"}`：强制调用指定工具

### 3.3 响应格式解析

**情况 1：需要调用工具**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "function_call": {
        "name": "get_weather",
        "arguments": "{\"location\": \"北京\"}"
      }
    }
  }]
}
```

**情况 2：直接回答**

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "这是一个直接回答，不需要调用工具。"
    }
  }]
}
```

---

## 4. 工具函数设计最佳实践

### 4.1 命名规范

```python
# ✅ 好的命名
get_weather           # 获取天气
search_products       # 搜索产品
create_order          # 创建订单
calculate_mortgage    # 计算房贷

# ❌ 避免的命名
weather               # 太模糊
do_search             # 无意义前缀
make_it               # 不清晰
func_1                # 无描述性
```

### 4.2 Description 编写技巧

**原则：清晰、具体、包含使用场景**

```python
# ❌ 差的描述
"获取天气数据"

# ✅ 好的描述
"获取指定城市的当前天气信息，包括温度、湿度、天气状况等。
当用户询问某个城市的天气、温度、是否下雨时使用此函数。"
```

**进阶技巧：**

```python
# 包含使用示例
description="""
查询数据库中的用户信息。

使用场景：
- 用户询问"查询张三的信息"
- 需要获取特定条件的用户列表
- 统计某个城市的用户数量

示例调用：
- query_database("users", {"name": "张三"})
- query_database("users", {"city": "北京"})
"""
```

### 4.3 参数设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **原子性** | 每个函数只做一件事 | `get_weather` 只获取天气，不获取新闻 |
| **幂等性** | 多次调用结果一致 | 查询类操作天然幂等 |
| **安全性** | 危险操作需要确认 | 删除数据前二次确认 |
| **可组合** | 函数可以链式调用 | A 的输出作为 B 的输入 |

---

## 5. 高级应用模式

### 5.1 并行工具调用

当多个工具调用相互独立时，可以并行执行：

```python
# 模型返回多个 tool_calls
tool_calls = response_message.tool_calls

# 并行执行
with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(execute_tool, tool_call)
        for tool_call in tool_calls
    ]
    results = [f.result() for f in futures]
```

### 5.2 工具链（Tool Chain）

```
用户："帮我找北京明天天气，如果下雨就提醒我带伞"
    │
    ▼
[Tool 1] get_weather(location="北京", date="明天")
    │
    ▼
结果：{"condition": "小雨", "temp": 20}
    │
    ▼
[Tool 2] create_reminder(
    title="带伞提醒",
    condition="明天北京有小雨，记得带伞"
)
    │
    ▼
完成
```

### 5.3 工具调用循环

某些任务需要多次工具调用：

```python
max_iterations = 5

while iteration < max_iterations:
    response = llm.chat_completion(messages)
    
    if not response.tool_calls:
        break  # 任务完成
    
    # 执行工具
    results = execute_tools(response.tool_calls)
    
    # 将结果加入对话
    messages.extend(build_tool_messages(results))
    
    iteration += 1
```

---

## 6. 与 ReAct Agent 集成

### 6.1 架构对比

| 特性 | Pure Function Calling | ReAct | 混合模式 |
|------|----------------------|-------|---------|
| **决策逻辑** | LLM 内置 | 显式推理链 | 结合两者 |
| **可解释性** | 较低 | 高 | 中等 |
| **多步推理** | 依赖迭代 | 原生支持 | 优化 |
| **工具调用** | 单次/并行 | 顺序执行 | 灵活 |
| **适用场景** | 简单任务 | 复杂推理 | 通用 |

### 6.2 混合架构设计

```
┌─────────────────────────────────────┐
│         用户输入                    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     意图分类器（路由层）             │
│  - 简单查询 → Function Calling       │
│  - 复杂任务 → ReAct 模式            │
│  - 需要推理 → ReAct 模式            │
└─────────────┬───────────────────────┘
              │
       ┌──────┴──────┐
       ▼             ▼
┌──────────┐   ┌──────────┐
│ Function │   │  ReAct   │
│ Calling  │   │  Agent   │
└────┬─────┘   └────┬─────┘
     │              │
     └──────┬───────┘
            ▼
┌─────────────────────────────────────┐
│         结果整合与输出               │
└─────────────────────────────────────┘
```

### 6.3 集成代码示例

```python
class HybridAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.function_caller = FunctionCallingAgent(tools)
        self.react_agent = ReActAgent(llm, tools)
    
    def run(self, query):
        # 意图分类
        intent = self.classify_intent(query)
        
        if intent == "simple":
            # 简单任务：使用 Function Calling
            return self.function_caller.run(query)
        else:
            # 复杂任务：使用 ReAct
            return self.react_agent.run(query)
    
    def classify_intent(self, query):
        """
        判断任务复杂度
        
        简单任务特征：
        - 单次信息查询
        - 直接计算
        - 明确的数据操作
        
        复杂任务特征：
        - 需要多步推理
        - 包含条件判断
        - 需要规划
        """
        # 使用轻量级分类器或 LLM 判断
        pass
```

---

## 7. 安全性与错误处理

### 7.1 安全防护措施

| 风险 | 防护方案 |
|------|---------|
| **注入攻击** | 参数校验、SQL 参数化 |
| **敏感操作** | 二次确认、权限检查 |
| **无限循环** | 最大迭代次数限制 |
| **信息泄露** | 工具访问范围控制 |
| **恶意调用** | 输入过滤、白名单机制 |

### 7.2 错误处理策略

```python
def execute_tool_safely(tool_call):
    try:
        # 1. 参数校验
        validate_parameters(tool_call.arguments)
        
        # 2. 权限检查
        check_permissions(tool_call.name)
        
        # 3. 执行工具
        result = execute_tool(tool_call)
        
        # 4. 结果格式化
        return format_success(result)
        
    except ValidationError as e:
        return format_error("参数错误", str(e))
    except PermissionError as e:
        return format_error("权限不足", str(e))
    except ToolExecutionError as e:
        return format_error("执行失败", str(e))
    except Exception as e:
        return format_error("未知错误", str(e))
```

---

## 8. 性能优化

### 8.1 优化策略

| 策略 | 实现方式 | 效果 |
|------|---------|------|
| **工具结果缓存** | Redis/Memcached | 减少重复调用 |
| **并行执行** | ThreadPoolExecutor | 加速独立工具调用 |
| **流式响应** | SSE/WebSocket | 提升用户体验 |
| **工具预加载** | 启动时初始化 | 减少首次调用延迟 |
| **智能选择** | 工具调用预测 | 减少不必要的调用 |

### 8.2 缓存设计示例

```python
from functools import lru_cache
import hashlib

class CachedToolExecutor:
    def __init__(self, cache_ttl=300):
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    def execute(self, tool_name, arguments):
        # 生成缓存键
        cache_key = self._make_key(tool_name, arguments)
        
        # 检查缓存
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        # 执行工具
        result = execute_tool(tool_name, arguments)
        
        # 存入缓存
        self.cache[cache_key] = (result, time.time())
        
        return result
```

---

## 9. 实际应用场景

### 9.1 智能客服系统

```
用户："我的订单 #12345 什么时候到？"
    │
    ▼
[Tool 1] extract_order_id("我的订单 #12345")
    │
    ▼
结果：{"order_id": "12345"}
    │
    ▼
[Tool 2] query_order_status(order_id="12345")
    │
    ▼
结果：{"status": "配送中", "eta": "明天下午"}
    │
    ▼
生成回答："您的订单 #12345 目前正在配送中，预计明天下午送达。"
```

### 9.2 数据分析助手

```
用户："分析一下上个月的销售额趋势"
    │
    ▼
[Tool 1] query_database(
    table="sales",
    date_range={"start": "2024-01-01", "end": "2024-01-31"}
)
    │
    ▼
[Tool 2] calculate_statistics(data)
    │
    ▼
[Tool 3] generate_chart(type="line", data=data)
    │
    ▼
生成分析报告
```

### 9.3 个人助理

```
用户："下周三下午帮我安排一个会议，邀请张三和李四"
    │
    ▼
[Tool 1] check_calendar(date="下周三", time="下午")
    │
    ▼
[Tool 2] create_event(
    date="下周三",
    time="下午",
    attendees=["张三", "李四"]
)
    │
    ▼
[Tool 3] send_invitation(event_id, attendees)
```

---

## 10. 学习路径与进阶

### 10.1 你已经掌握的内容

✅ **本课完成：**
- Function Calling 核心概念
- Tools Schema 设计
- API 调用流程
- 工具函数实现
- 与 Agent 框架集成

### 10.2 推荐进阶方向

| 方向 | 内容 | 资源 |
|------|------|------|
| **Multi-Agent** | 多智能体工具共享与协作 | AutoGen, CrewAI |
| **开源模型** | 本地部署支持 Function Calling 的模型 | Llama 2/3, Qwen |
| **工具市场** | 构建可复用的工具生态系统 | LangChain Hub |
| **可视化** | 工具调用链的可视化展示 | LangSmith |
| **安全性** | 工具调用的沙箱与权限管理 | 相关论文 |

### 10.3 下一步学习建议

1. **实战项目**：构建一个支持多种工具的对话机器人
2. **API 接入**：接入真实的 OpenAI/Claude/国产大模型 API
3. **性能优化**：实现工具缓存、并行调用
4. **Multi-Agent**：学习多个 Agent 如何共享工具

---

## 总结

Function Calling 是大模型应用开发的核心能力，它让 AI 从"只会说"进化为"能做"。通过本课的学习，你已经掌握了：

1. **理论知识**：Function Calling 的工作原理
2. **实践技能**：Tools Schema 设计、API 调用、工具实现
3. **架构思维**：与 ReAct Agent 的集成方案
4. **工程意识**：安全性、错误处理、性能优化

**关键要点：**
- Schema 设计要清晰、具体、包含使用场景
- 工具函数要单一职责、安全可控
- 错误处理要全面，给出有用的错误信息
- 性能优化要考虑缓存、并行、流式

准备好继续学习 **Multi-Agent 系统** 了吗？🚀
