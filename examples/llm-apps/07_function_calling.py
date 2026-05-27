#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第28课：Function Calling 与 Tools 使用
======================================

课程目标：
- 理解 Function Calling 的核心概念和工作原理
- 掌握如何定义和注册 Tools（工具函数）
- 学习 OpenAI Function Calling API 的使用
- 实现一个能调用外部工具的 AI Agent
- 将 Tools 集成到现有 Agent 框架中

应用场景：
- 计算器：让 AI 能进行精确数学计算
- 天气查询：获取实时天气信息
- 搜索引擎：获取最新知识
- 数据库查询：操作外部数据源
- API调用：与第三方服务交互

"""

import os
import sys
import json
import math
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("🚀 第28课：Function Calling 与 Tools 使用")
print("=" * 70)
print()

# =============================================================================
# 第一部分：Function Calling 核心概念
# =============================================================================

print("📚 第一部分：Function Calling 核心概念")
print("-" * 70)

concept_explanation = """
什么是 Function Calling（函数调用）？

Function Calling 是大模型的一种能力，允许模型在需要时调用外部函数/API 来获取
信息或执行操作，而不是仅依赖预训练的知识。

核心流程：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   用户提问   │────▶│  大模型分析  │────▶│ 判断是否需要 │
│"今天北京天气"│     │  理解意图   │     │  调用工具   │
└─────────────┘     └──────┬──────┘     └──────┬──────┘
                           │                    │
                           ▼                    │
                    ┌─────────────┐            │
                    │  识别需要   │            │
                    │  weather()  │◄───────────┘
                    │  函数调用   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ 返回函数调用 │
                    │ 请求（JSON） │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  执行函数   │
                    │ weather(    │
                    │   "北京")   │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ 返回结果    │
                    │ "晴，25°C"  │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐     ┌─────────────┐
                    │ 大模型生成  │────▶│ 返回给用户  │
                    │ 自然语言回答│     │ "今天北京..."│
                    └─────────────┘     └─────────────┘

为什么需要 Function Calling？
1. 解决知识截止时间问题（获取实时信息）
2. 执行精确计算（避免 LLM 数学错误）
3. 与外部系统交互（数据库、API、文件系统）
4. 增强 AI 的实际行动能力
"""

print(concept_explanation)
print()

# =============================================================================
# 第二部分：定义 Tools（工具函数）
# =============================================================================

print("🔧 第二部分：定义 Tools（工具函数）")
print("-" * 70)

# 模拟天气查询工具
def get_weather(location: str, date: Optional[str] = None) -> Dict[str, Any]:
    """
    获取指定地区的天气信息（模拟实现）
    
    Args:
        location: 城市名称，如"北京"、"上海"
        date: 日期，格式"YYYY-MM-DD"，默认为今天
        
    Returns:
        包含天气信息的字典
    """
    # 模拟天气数据
    mock_weather_db = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 45, "wind": "东北风3级"},
        "上海": {"temp": 28, "condition": "多云", "humidity": 65, "wind": "东南风2级"},
        "广州": {"temp": 32, "condition": "小雨", "humidity": 80, "wind": "南风1级"},
        "深圳": {"temp": 31, "condition": "阴", "humidity": 75, "wind": "无持续风向"},
        "杭州": {"temp": 26, "condition": "晴", "humidity": 55, "wind": "东风2级"},
    }
    
    if location in mock_weather_db:
        data = mock_weather_db[location].copy()
        data["location"] = location
        data["date"] = date or datetime.now().strftime("%Y-%m-%d")
        return data
    else:
        return {"error": f"未找到 {location} 的天气数据", "location": location}


# 计算器工具
def calculator(expression: str) -> Dict[str, Any]:
    """
    执行数学计算（安全版本）
    
    Args:
        expression: 数学表达式，如"2 + 2", "sqrt(16)", "10 * 5"
        
    Returns:
        计算结果
    """
    try:
        # 安全评估：只允许基本数学运算
        allowed_names = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        # 简单表达式求值
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return {
            "expression": expression,
            "result": round(result, 6) if isinstance(result, float) else result,
            "success": True
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


# 搜索工具（模拟）
def search(query: str, num_results: int = 3) -> List[Dict[str, str]]:
    """
    模拟搜索引擎（实际应用中调用 Bing/Google API）
    
    Args:
        query: 搜索关键词
        num_results: 返回结果数量
        
    Returns:
        搜索结果列表
    """
    # 模拟知识库
    mock_search_db = {
        "Python": [
            {"title": "Python 官方文档", "snippet": "Python 是一种解释型、高级编程语言..."},
            {"title": "Python 教程 - 菜鸟教程", "snippet": "Python 是一种解释型、面向对象..."},
        ],
        "Transformer": [
            {"title": "Attention Is All You Need", "snippet": "Transformer 是一种基于注意力机制的..."},
            {"title": "Hugging Face Transformers", "snippet": "开源自然语言处理库..."},
        ],
        "OpenAI": [
            {"title": "OpenAI 官网", "snippet": "OpenAI 是一家人工智能研究公司..."},
            {"title": "GPT-4 技术报告", "snippet": "GPT-4 是大型多模态语言模型..."},
        ],
    }
    
    # 查找匹配的结果
    results = []
    for key, items in mock_search_db.items():
        if key.lower() in query.lower() or query.lower() in key.lower():
            results.extend(items)
    
    # 如果没有直接匹配，返回默认结果
    if not results:
        results = [
            {"title": f"关于 '{query}' 的搜索结果", "snippet": f"这是关于 {query} 的模拟搜索结果..."},
            {"title": f"{query} - 百度百科", "snippet": f"{query} 是一个常见的技术术语..."},
        ]
    
    return results[:num_results]


# 数据库查询工具（模拟）
def query_database(table: str, filters: Optional[Dict] = None) -> Dict[str, Any]:
    """
    模拟数据库查询
    
    Args:
        table: 表名，如"users", "orders"
        filters: 过滤条件
        
    Returns:
        查询结果
    """
    mock_db = {
        "users": [
            {"id": 1, "name": "张三", "age": 25, "city": "北京"},
            {"id": 2, "name": "李四", "age": 30, "city": "上海"},
            {"id": 3, "name": "王五", "age": 28, "city": "广州"},
        ],
        "orders": [
            {"id": 101, "user_id": 1, "product": "iPhone", "amount": 5999},
            {"id": 102, "user_id": 2, "product": "MacBook", "amount": 12999},
            {"id": 103, "user_id": 1, "product": "AirPods", "amount": 1299},
        ]
    }
    
    if table not in mock_db:
        return {"error": f"表 '{table}' 不存在"}
    
    data = mock_db[table]
    
    # 应用过滤
    if filters:
        for key, value in filters.items():
            data = [item for item in data if item.get(key) == value]
    
    return {
        "table": table,
        "count": len(data),
        "data": data
    }


print("✅ 已定义以下工具函数：")
print()
print("1. get_weather(location, date)")
print("   功能：获取城市天气信息")
print("   示例：get_weather('北京')")
print()
print("2. calculator(expression)")
print("   功能：数学计算器")
print("   示例：calculator('sqrt(16) + 10')")
print()
print("3. search(query, num_results)")
print("   功能：搜索引擎（模拟）")
print("   示例：search('Python 教程')")
print()
print("4. query_database(table, filters)")
print("   功能：数据库查询（模拟）")
print("   示例：query_database('users', {'city': '北京'})")
print()

# =============================================================================
# 第三部分：Tools Schema 定义（OpenAI 格式）
# =============================================================================

print("📋 第三部分：Tools Schema 定义（OpenAI 格式）")
print("-" * 70)

# 定义每个工具的 JSON Schema（OpenAI Function Calling 格式）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息，包括温度、天气状况、湿度等",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海、广州"
                    },
                    "date": {
                        "type": "string",
                        "description": "日期，格式为 YYYY-MM-DD，默认为今天"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算，支持基本运算和常用数学函数",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如：'2 + 2', 'sqrt(16)', '10 * 5 + 3'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索互联网信息，获取最新的知识和数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认为3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "查询数据库，获取用户信息或订单数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": "表名，可选值：'users'（用户表）、'orders'（订单表）"
                    },
                    "filters": {
                        "type": "object",
                        "description": "过滤条件，例如：{'city': '北京'}"
                    }
                },
                "required": ["table"]
            }
        }
    }
]

print("Tools Schema（JSON 格式）：")
print(json.dumps(TOOLS_SCHEMA, ensure_ascii=False, indent=2))
print()

print("Schema 结构说明：")
print("  - type: 'function' 表示这是一个函数工具")
print("  - function.name: 函数名称（必须唯一）")
print("  - function.description: 函数描述（LLM 根据此描述决定何时调用）")
print("  - function.parameters: 参数定义（JSON Schema 格式）")
print("    - properties: 每个参数的详细定义")
print("    - required: 必需参数列表")
print()

# =============================================================================
# 第四部分：模拟 Function Calling 流程
# =============================================================================

print("🤖 第四部分：模拟 Function Calling 流程")
print("-" * 70)

# 工具函数映射表
AVAILABLE_TOOLS: Dict[str, Callable] = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search": search,
    "query_database": query_database,
}


class MockLLM:
    """
    模拟支持 Function Calling 的大语言模型
    
    实际应用中，这里会调用 OpenAI、Claude、文心等真实 API
    """
    
    def __init__(self, tools_schema: List[Dict]):
        self.tools_schema = tools_schema
        self.tools_map = {tool["function"]["name"]: tool for tool in tools_schema}
    
    def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> Dict:
        """
        模拟 LLM API 调用
        
        实际 API 响应格式：
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_xxx",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "北京"}'
                        }
                    }]
                }
            }]
        }
        """
        # 获取最后一条用户消息
        user_message = messages[-1]["content"]
        
        # 模拟 LLM 决策逻辑（实际由模型完成）
        tool_call = self._simulate_tool_decision(user_message)
        
        if tool_call:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    }
                }]
            }
        else:
            # 不需要调用工具，直接回答
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": self._simulate_direct_answer(user_message)
                    }
                }]
            }
    
    def _simulate_tool_decision(self, query: str) -> Optional[Dict]:
        """模拟 LLM 判断是否调用工具"""
        query_lower = query.lower()
        
        # 天气查询
        if any(word in query_lower for word in ["天气", "温度", "下雨", "晴"]):
            # 提取城市名（简化处理）
            cities = ["北京", "上海", "广州", "深圳", "杭州"]
            for city in cities:
                if city in query:
                    return {
                        "id": "call_weather_001",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": city})
                        }
                    }
        
        # 数学计算
        if any(word in query_lower for word in ["计算", "等于", "多少", "sqrt", "平方", "加", "减", "乘", "除"]):
            # 提取表达式（简化处理）
            import re
            # 匹配数学表达式
            patterns = [
                r'(\d+\s*[\+\-\*\/]\s*\d+)',
                r'sqrt\(\d+\)',
                r'(\d+)\s*的平方',
                r'(\d+)\s*\+\s*(\d+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, query)
                if match:
                    expr = match.group(0)
                    # 转换中文表达
                    expr = expr.replace("的平方", "**2").replace("加", "+").replace("减", "-").replace("乘", "*").replace("除", "/")
                    return {
                        "id": "call_calc_001",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": json.dumps({"expression": expr})
                        }
                    }
        
        # 搜索
        if any(word in query_lower for word in ["搜索", "查找", "什么是", "介绍一下"]):
            # 提取搜索词
            search_terms = ["Python", "Transformer", "OpenAI"]
            for term in search_terms:
                if term in query:
                    return {
                        "id": "call_search_001",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": term})
                        }
                    }
        
        # 数据库查询
        if any(word in query_lower for word in ["用户", "订单", "数据库", "查询"]):
            if "北京" in query and "用户" in query:
                return {
                    "id": "call_db_001",
                    "type": "function",
                    "function": {
                        "name": "query_database",
                        "arguments": json.dumps({"table": "users", "filters": {"city": "北京"}})
                    }
                }
        
        return None
    
    def _simulate_direct_answer(self, query: str) -> str:
        """模拟直接回答（不调用工具）"""
        return f"这是一个模拟回答。实际应用中，这里会返回 LLM 的生成内容。"


# 创建模拟 LLM 实例
mock_llm = MockLLM(TOOLS_SCHEMA)

print("模拟 LLM Function Calling 流程：")
print()

# 测试用例
test_queries = [
    "北京今天天气怎么样？",
    "计算 16 的平方根加上 10",
    "搜索一下 Python 的相关信息",
    "查询北京的用户有哪些？",
    "你好，请介绍一下自己",  # 不需要调用工具
]

for query in test_queries:
    print(f"👤 用户：{query}")
    
    # 构建消息
    messages = [
        {"role": "system", "content": "你是一个有用的助手，可以调用工具来获取信息。"},
        {"role": "user", "content": query}
    ]
    
    # 调用 LLM
    response = mock_llm.chat_completion(messages, tools=TOOLS_SCHEMA)
    
    # 解析响应
    message = response["choices"][0]["message"]
    
    if message.get("tool_calls"):
        print("🤖 LLM 决定调用工具：")
        for tool_call in message["tool_calls"]:
            func = tool_call["function"]
            print(f"   工具：{func['name']}")
            print(f"   参数：{func['arguments']}")
            
            # 执行工具函数
            tool_name = func["name"]
            tool_args = json.loads(func["arguments"])
            
            if tool_name in AVAILABLE_TOOLS:
                result = AVAILABLE_TOOLS[tool_name](**tool_args)
                print(f"   结果：{json.dumps(result, ensure_ascii=False)}")
            else:
                print(f"   ❌ 工具 {tool_name} 不存在")
    else:
        print(f"🤖 LLM 直接回答：{message.get('content', '')}")
    
    print("-" * 50)

print()

# =============================================================================
# 第五部分：完整的 Function Calling Agent
# =============================================================================

print("🎯 第五部分：完整的 Function Calling Agent")
print("-" * 70)


class FunctionCallingAgent:
    """
    支持 Function Calling 的智能代理
    
    整合 LLM + Tools，实现自动工具调用循环
    """
    
    def __init__(self, tools: Dict[str, Callable], tools_schema: List[Dict]):
        self.tools = tools
        self.tools_schema = tools_schema
        self.llm = MockLLM(tools_schema)
        self.conversation_history: List[Dict] = []
    
    def run(self, user_query: str, max_iterations: int = 5) -> str:
        """
        执行用户查询，自动处理工具调用
        
        Args:
            user_query: 用户输入
            max_iterations: 最大工具调用轮数（防止无限循环）
            
        Returns:
            最终回答
        """
        print(f"📝 用户查询：{user_query}")
        print()
        
        # 初始化对话
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_query}
        ]
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"🔄 第 {iteration} 轮对话")
            
            # 调用 LLM
            response = self.llm.chat_completion(messages, tools=self.tools_schema)
            message = response["choices"][0]["message"]
            
            # 检查是否需要调用工具
            if not message.get("tool_calls"):
                # LLM 直接回答了
                print("✅ LLM 生成最终回答")
                return message.get("content", "")
            
            # 处理工具调用
            print(f"🔧 LLM 请求调用 {len(message['tool_calls'])} 个工具")
            
            tool_results = []
            for tool_call in message["tool_calls"]:
                func = tool_call["function"]
                tool_name = func["name"]
                tool_args = json.loads(func["arguments"])
                
                print(f"   调用 {tool_name}({tool_args})")
                
                # 执行工具
                if tool_name in self.tools:
                    try:
                        result = self.tools[tool_name](**tool_args)
                        result_str = json.dumps(result, ensure_ascii=False)
                        print(f"   结果：{result_str[:100]}...")
                    except Exception as e:
                        result_str = f"错误：{str(e)}"
                        print(f"   ❌ {result_str}")
                else:
                    result_str = f"错误：工具 {tool_name} 不存在"
                    print(f"   ❌ {result_str}")
                
                # 构建工具响应消息
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": tool_name,
                    "content": result_str
                })
            
            # 将工具调用和结果添加到对话历史
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message["tool_calls"]
            })
            
            for tool_result in tool_results:
                messages.append(tool_result)
            
            print()
        
        return "达到最大迭代次数，无法完成请求。"
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个智能助手，可以调用工具来获取信息或执行操作。

可用的工具包括：
1. get_weather - 查询天气
2. calculator - 数学计算
3. search - 搜索信息
4. query_database - 数据库查询

当用户问题需要实时信息、精确计算或外部数据时，请使用相应的工具。
调用工具后，根据结果生成自然语言回答。"""


# 创建 Function Calling Agent
print("创建 Function Calling Agent...")
agent = FunctionCallingAgent(AVAILABLE_TOOLS, TOOLS_SCHEMA)
print("✅ Agent 创建成功\n")

# 测试完整流程
test_cases = [
    "北京和上海哪个城市温度更高？",
    "帮我计算半径为5的圆的面积",
    "搜索 Transformer 的相关信息",
    "查询用户张三的订单信息",
]

print("=" * 70)
print("Function Calling Agent 测试")
print("=" * 70)
print()

for test in test_cases:
    print(f"{'='*70}")
    result = agent.run(test)
    print(f"💡 最终结果：{result}")
    print()

# =============================================================================
# 第六部分：与现有 Agent 框架集成
# =============================================================================

print("🔌 第六部分：与现有 Agent 框架集成")
print("-" * 70)

integration_guide = """
如何将 Function Calling 集成到之前的 ReAct Agent 框架？

方案1：将 Tools 作为 ReAct 的 Actions
----------------------------
在 ReAct 框架中，Tools 就是具体的 Action 实现：

观察(Observation) ──▶ 思考(Thought) ──▶ 行动(Action)
                                         │
                                         ▼
                                    调用 Tool 函数
                                         │
                                         ▼
                                    获得 Observation
                                         │
                                         ▼
                                    继续循环...

方案2：混合模式
------------
- 简单任务：直接使用 Function Calling（单次调用）
- 复杂任务：使用 ReAct 模式（多步推理）

代码示例：

class AdvancedAgent(ReActAgent):
    def __init__(self, llm, tools):
        super().__init__(llm)
        self.tools = tools
        self.function_caller = FunctionCallingAgent(tools, tools_schema)
    
    def think_and_act(self, observation):
        # 判断是否需要工具
        if self.needs_tool(observation):
            # 使用 Function Calling 快速获取信息
            tool_result = self.function_caller.run(observation)
            return Action("observe", tool_result)
        else:
            # 使用 ReAct 推理
            return super().think_and_act(observation)

优势对比：
- Function Calling：快速、高效、单次调用
- ReAct：可解释性强、支持多步推理、适合复杂任务
- 混合模式：兼顾效率和灵活性
"""

print(integration_guide)
print()

# =============================================================================
# 第七部分：实际 API 调用示例（OpenAI）
# =============================================================================

print("🌐 第七部分：实际 API 调用示例（OpenAI）")
print("-" * 70)

openai_example = '''
# 实际使用 OpenAI API 进行 Function Calling

import openai

# 设置 API 密钥
openai.api_key = "your-api-key"

# 定义工具（与之前相同的 schema）
tools = TOOLS_SCHEMA

# 用户查询
messages = [
    {"role": "user", "content": "北京今天天气怎么样？"}
]

# 第一次调用：让模型决定是否调用工具
response = openai.chat.completions.create(
    model="gpt-4-1106-preview",  # 或 gpt-3.5-turbo-1106
    messages=messages,
    tools=tools,
    tool_choice="auto"  # 让模型自动决定是否调用工具
)

# 检查响应
response_message = response.choices[0].message

if response_message.tool_calls:
    # 模型决定调用工具
    messages.append(response_message)  # 添加助手消息
    
    # 执行工具调用
    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # 调用本地函数
        function_response = AVAILABLE_TOOLS[function_name](**function_args)
        
        # 添加工具响应到对话
        messages.append({
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": json.dumps(function_response)
        })
    
    # 第二次调用：让模型根据工具结果生成回答
    second_response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages
    )
    
    final_answer = second_response.choices[0].message.content
    print(final_answer)
else:
    # 模型直接回答了
    print(response_message.content)

关键参数说明：
- tools: 可用的工具列表
- tool_choice: 
  - "auto": 让模型自动决定
  - "none": 不调用任何工具
  - {"type": "function", "function": {"name": "xxx"}}: 强制调用指定工具
'''

print(openai_example)
print()

# =============================================================================
# 第七部分（续）：DeepSeek API Function Calling 示例
# =============================================================================

print("🌐 第七部分（续）：DeepSeek API Function Calling 示例")
print("-" * 70)

deepseek_example = '''
# 使用 DeepSeek API 进行 Function Calling
# DeepSeek 支持 OpenAI 兼容的接口格式

import openai
import json

# 配置 DeepSeek API
# 支持国内直接访问，性价比高
client = openai.OpenAI(
    api_key="your-deepseek-api-key",  # 从 https://platform.deepseek.com 获取
    base_url="https://api.deepseek.com/v1"  # DeepSeek API 端点
)

# 定义工具（与 OpenAI 相同的 schema 格式）
tools = [
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
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如：2 + 2, sqrt(16)"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]

# 工具函数映射
available_functions = {
    "get_weather": get_weather,
    "calculator": calculator
}

def chat_with_function_calling(user_message, model="deepseek-chat"):
    """
    使用 DeepSeek 进行 Function Calling 的完整流程
    
    DeepSeek 支持的模型：
    - deepseek-chat (推荐，通用对话)
    - deepseek-coder (代码专用)
    - deepseek-reasoner (推理增强)
    """
    
    messages = [
        {"role": "system", "content": "你是一个智能助手，可以使用工具帮助用户。"},
        {"role": "user", "content": user_message}
    ]
    
    # 第一次调用：让模型决定是否调用工具
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto"  # 让模型自动决定
    )
    
    response_message = response.choices[0].message
    
    # 检查是否需要调用工具
    if response_message.tool_calls:
        print(f"🤖 DeepSeek 决定调用 {len(response_message.tool_calls)} 个工具")
        
        # 添加助手消息到对话
        messages.append({
            "role": "assistant",
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in response_message.tool_calls
            ]
        })
        
        # 执行所有工具调用
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"   调用: {function_name}({function_args})")
            
            # 执行本地函数
            if function_name in available_functions:
                function_response = available_functions[function_name](**function_args)
                
                # 添加工具响应到对话
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(function_response, ensure_ascii=False)
                })
            else:
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps({"error": f"函数 {function_name} 不存在"})
                })
        
        # 第二次调用：让模型根据工具结果生成回答
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        return second_response.choices[0].message.content
    else:
        # 模型直接回答了
        return response_message.content

# 测试示例
test_queries = [
    "北京今天天气怎么样？",
    "帮我计算 2 的 10 次方",
]

for query in test_queries:
    print(f"\\n👤 用户: {query}")
    answer = chat_with_function_calling(query)
    print(f"🤖 DeepSeek: {answer}")

DeepSeek Function Calling 特点：
1. 兼容 OpenAI API 格式，迁移成本低
2. 支持中文理解，对中文工具描述友好
3. 价格优势明显（约为 GPT-4 的 1/10）
4. 国内访问稳定，无需代理
5. 支持 deepseek-chat/deepseek-coder/deepseek-reasoner 多个模型

注意事项：
- 需要申请 API Key: https://platform.deepseek.com
- 免费额度：新用户有 10 元免费额度
- 速率限制：注意查看官方文档的 RPM/TPM 限制
- 工具描述建议用中文，效果更好
'''

print(deepseek_example)
print()

# 创建实际的 DeepSeek Function Calling Agent 类
class DeepSeekFunctionAgent:
    """
    基于 DeepSeek API 的 Function Calling Agent
    
    使用方法：
    1. 设置环境变量 DEEPSEEK_API_KEY
    2. 初始化 Agent: agent = DeepSeekFunctionAgent(tools, tools_schema)
    3. 运行查询: result = agent.run("北京天气怎么样？")
    """
    
    def __init__(self, tools: Dict[str, Callable], tools_schema: List[Dict], api_key: Optional[str] = None):
        self.tools = tools
        self.tools_schema = tools_schema
        
        # 获取 API Key
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            print("⚠️ 未设置 DEEPSEEK_API_KEY，将使用模拟模式")
            self.client = None
        else:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1"
            )
    
    def run(self, user_query: str, model: str = "deepseek-chat", max_iterations: int = 3) -> str:
        """
        执行带 Function Calling 的对话
        
        Args:
            user_query: 用户查询
            model: 模型名称，默认 deepseek-chat
            max_iterations: 最大工具调用轮数
            
        Returns:
            模型生成的回答
        """
        if not self.client:
            # 模拟模式
            return f"[模拟模式] DeepSeek 处理查询: {user_query}\\n（请设置 DEEPSEEK_API_KEY 使用真实 API）"
        
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": user_query}
        ]
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # 调用 DeepSeek API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=self.tools_schema,
                    tool_choice="auto"
                )
                
                response_message = response.choices[0].message
                
                # 检查是否需要调用工具
                if not response_message.tool_calls:
                    return response_message.content
                
                # 添加助手消息
                messages.append({
                    "role": "assistant",
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })
                
                # 执行工具调用
                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    # 执行本地函数
                    if func_name in self.tools:
                        try:
                            result = self.tools[func_name](**func_args)
                            result_content = json.dumps(result, ensure_ascii=False)
                        except Exception as e:
                            result_content = json.dumps({"error": str(e)})
                    else:
                        result_content = json.dumps({"error": f"工具 {func_name} 不存在"})
                    
                    # 添加工具响应
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": result_content
                    })
                
            except Exception as e:
                return f"调用 DeepSeek API 出错: {str(e)}"
        
        return "达到最大迭代次数"
    
    def _get_system_prompt(self) -> str:
        return """你是一个智能助手，可以调用工具来获取信息或执行操作。

可用的工具包括：
1. get_weather - 查询天气
2. calculator - 数学计算
3. search - 搜索信息
4. query_database - 数据库查询

当用户问题需要实时信息、精确计算或外部数据时，请使用相应的工具。
调用工具后，根据结果生成自然语言回答。"""


# 演示 DeepSeek Agent 的使用
print("=" * 70)
print("🚀 DeepSeek Function Calling Agent 演示")
print("=" * 70)
print()
print("创建 DeepSeek Agent...")
deepseek_agent = DeepSeekFunctionAgent(AVAILABLE_TOOLS, TOOLS_SCHEMA)

# 检查是否有真实的 API Key
if os.getenv("DEEPSEEK_API_KEY"):
    print("✅ Agent 创建成功（使用真实 DeepSeek API）")
    print()
    print("📝 测试 DeepSeek Function Calling：")
    print("-" * 70)
    
    # 使用真实 API 运行测试
    test_queries = [
        "北京今天天气怎么样？",
        "帮我计算 16 的平方根加上 10",
        "搜索一下 Python 的相关信息",
        "查询北京的天气和上海的天气，告诉我哪个城市更热？",
    ]
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{i}. 👤 用户: {test_query}")
        print(f"   🤖 DeepSeek 正在处理...")
        try:
            result = deepseek_agent.run(test_query)
            print(f"   💡 回答: {result[:200]}..." if len(result) > 200 else f"   💡 回答: {result}")
        except Exception as e:
            print(f"   ❌ 错误: {str(e)}")
    
else:
    print("⚠️ Agent 创建成功（模拟模式，未检测到 DEEPSEEK_API_KEY）")
    print()
    print("使用方法：")
    print("  1. 访问 https://platform.deepseek.com 获取 API Key")
    print("  2. 设置环境变量: export DEEPSEEK_API_KEY='your-key'")
    print("  3. 运行: result = deepseek_agent.run('北京天气怎么样？')")
    print()
    print("💡 模拟运行示例（请配置 API Key 后体验真实效果）：")
    test_query = "北京和上海哪个城市温度更高？"
    result = deepseek_agent.run(test_query)
    print(f"查询: {test_query}")
    print(f"结果: {result}")

print()

# =============================================================================
# 第八部分：总结与最佳实践
# =============================================================================

print("=" * 70)
print("📋 总结与最佳实践")
print("=" * 70)

summary = """
Function Calling 核心要点：

1. 定义清晰的 Tools Schema
   - name: 使用有意义的英文名称
   - description: 详细描述功能和使用场景（LLM靠此决定调用）
   - parameters: 明确定义每个参数的类型和说明

2. 错误处理
   - 工具调用失败时返回错误信息
   - LLM 可以根据错误信息调整策略

3. 安全性
   - 对敏感操作添加确认步骤
   - 使用沙箱环境执行代码
   - 限制工具访问范围

4. 性能优化
   - 并行执行独立的工具调用
   - 缓存常用工具结果
   - 设置超时时间防止阻塞

5. 调试技巧
   - 打印完整的工具调用链
   - 记录工具执行时间和结果
   - 分析 LLM 的工具选择决策

下一步学习建议：
- 接入真实的 LLM API（OpenAI/Claude/国产模型）
- 实现更复杂的工具链（工具 A 的输出作为工具 B 的输入）
- 学习 LangChain 的 Tools 封装
- 探索 Multi-Agent 场景下的工具共享
"""

print(summary)

print()
print("=" * 70)
print("🎉 第28课完成！你已经掌握了 Function Calling 技术！")
print("=" * 70)
print()
print("【下节课预告】")
print("   第29课：Multi-Agent 系统设计与实现")
print("   - 多智能体协作架构")
print("   - Agent 之间的通信机制")
print("   - 角色分工与任务分配")
print()
print("准备好继续吗？😊")
