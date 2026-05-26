#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent 开发实战 - ReAct 智能代理
=====================================
项目名称：基于大语言模型的自主智能代理
任务：实现 ReAct (Reasoning + Acting) 模式的 AI Agent，能够思考、行动、观察并完成任务

学习目标：
- 理解 AI Agent 的核心概念（感知-思考-行动循环）
- 掌握 ReAct 模式（推理+行动）的实现
- 学习工具调用（Tool Use）机制
- 实现记忆管理（短期记忆+长期记忆）
- 掌握 Agent 的自我反思和迭代优化

AI Agent 架构：
┌─────────────┐
│   用户输入   │
└──────┬──────┘
       ↓
┌─────────────┐     ┌─────────────┐
│  思考(Reason)│←───→│  记忆(Memory)│
└──────┬──────┘     └─────────────┘
       ↓
┌─────────────┐     ┌─────────────┐
│  行动(Act)   │←───→│  工具(Tools) │
└──────┬──────┘     └─────────────┘
       ↓
┌─────────────┐
│  观察(Observe)│
└──────┬──────┘
       ↓
    [循环直到完成]
"""

import os

# ========== 配置 Hugging Face 镜像源 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

print("=" * 70)
print("AI Agent 开发实战 - ReAct 智能代理")
print("=" * 70)

# ============================================================
# 第一部分：AI Agent 基础概念
# ============================================================
print("\n" + "=" * 70)
print("第一部分：AI Agent 基础概念")
print("=" * 70)

print("""
【什么是 AI Agent？】

AI Agent = 大语言模型(LLM) + 工具(Tools) + 记忆(Memory) + 规划(Planning)

核心循环（感知-思考-行动）：
1. 感知(Perception): 接收用户输入和环境信息
2. 思考(Thought): 分析当前状态，制定计划
3. 行动(Action): 选择并执行工具
4. 观察(Observation): 获取工具返回的结果
5. [循环] 直到任务完成

【ReAct 模式】
ReAct = Reasoning + Acting

论文: "ReAct: Synergizing Reasoning and Acting in Language Models"

传统方式:
  用户: "2024年诺贝尔文学奖得主是谁？"
  AI: [直接猜测] 可能是村上春树...

ReAct方式:
  思考: 我需要搜索2024年诺贝尔文学奖的最新信息
  行动: 调用 search_tool("2024年诺贝尔文学奖得主")
  观察: 2024年诺贝尔文学奖授予了韩江(Han Kang)
  回答: 2024年诺贝尔文学奖得主是韩国作家韩江

【工具(Tools)】
工具是 Agent 扩展能力的手段:
- search: 搜索引擎，获取实时信息
- calculator: 计算器，执行数学运算
- weather: 天气查询
- code_interpreter: 代码执行
- database: 数据库查询
""")

# ============================================================
# 第二部分：工具(Tools)定义
# ============================================================
print("\n" + "=" * 70)
print("第二部分：工具(Tools)定义")
print("=" * 70)

import json
import random
import datetime
from typing import Dict, List, Any, Callable
from dataclasses import dataclass, field


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: Dict[str, Any]
    func: Callable
    
    def execute(self, **kwargs) -> str:
        """执行工具"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {str(e)}"


# 定义工具函数
def search_tool(query: str) -> str:
    """模拟搜索工具 - 实际应用中调用搜索引擎API"""
    # 模拟搜索结果
    knowledge_base = {
        "2024年诺贝尔文学奖": "2024年诺贝尔文学奖授予韩国作家韩江(Han Kang)，表彰其'以强烈的诗意散文直面历史创伤，揭示人类生命的脆弱'。",
        "中国的首都是哪里": "中国的首都是北京，位于华北地区。",
        "爱因斯坦": "阿尔伯特·爱因斯坦(Albert Einstein, 1879-1955)是德裔美籍物理学家，创立了相对论。",
        "Python": "Python是一种高级编程语言，由Guido van Rossum于1991年创建，以简洁易读著称。",
        "深度学习": "深度学习是机器学习的一个分支，使用多层神经网络模拟人脑的工作方式。",
    }
    
    # 模糊匹配
    for key, value in knowledge_base.items():
        if key in query or query in key:
            return value
    
    return f"搜索结果: 关于'{query}'的信息（这是模拟结果）"


def calculator_tool(expression: str) -> str:
    """计算器工具"""
    try:
        # 安全计算 - 只允许基本运算
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "错误: 表达式包含非法字符"
        
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def weather_tool(location: str) -> str:
    """模拟天气查询工具"""
    weathers = ["晴天", "多云", "阴天", "小雨", "大雨", "雪"]
    temp = random.randint(-5, 35)
    weather = random.choice(weathers)
    return f"{location}当前天气: {weather}, 温度 {temp}°C"


def datetime_tool() -> str:
    """获取当前时间"""
    now = datetime.datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}"


# 工具注册表
TOOLS = {
    "search": Tool(
        name="search",
        description="搜索引擎，用于查询事实性信息、新闻、百科知识等",
        parameters={
            "query": "搜索关键词，str类型"
        },
        func=search_tool
    ),
    "calculator": Tool(
        name="calculator",
        description="计算器，用于执行数学运算。支持 + - * / 和括号",
        parameters={
            "expression": "数学表达式，str类型，例如 '123 + 456' 或 '(100 - 20) * 5'"
        },
        func=calculator_tool
    ),
    "weather": Tool(
        name="weather",
        description="天气查询，获取指定城市的当前天气",
        parameters={
            "location": "城市名称，str类型"
        },
        func=weather_tool
    ),
    "datetime": Tool(
        name="datetime",
        description="获取当前日期和时间",
        parameters={},
        func=datetime_tool
    ),
}

print("\n🛠️  已注册工具:")
for name, tool in TOOLS.items():
    print(f"   • {name}: {tool.description}")

# ============================================================
# 第三部分：记忆(Memory)管理
# ============================================================
print("\n" + "=" * 70)
print("第三部分：记忆(Memory)管理")
print("=" * 70)

print("""
【Agent 记忆类型】

1. 短期记忆(Short-term Memory):
   - 当前对话的上下文
   - ReAct循环中的思考-行动-观察链
   - 随着新对话开始而清空

2. 长期记忆(Long-term Memory):
   - 用户的偏好设置
   - 历史对话的关键信息
   - 需要显式存储和检索

3. 工作记忆(Working Memory):
   - 当前任务的中间结果
   - 工具调用的返回数据
""")


class Memory:
    """简单的记忆管理类"""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.conversation_history: List[Dict[str, str]] = []
        self.user_preferences: Dict[str, Any] = {}
    
    def add_turn(self, role: str, content: str):
        """添加一轮对话"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })
        # 保持最大轮数
        if len(self.conversation_history) > self.max_turns:
            self.conversation_history = self.conversation_history[-self.max_turns:]
    
    def get_context(self) -> str:
        """获取当前上下文"""
        context = []
        for turn in self.conversation_history:
            context.append(f"{turn['role']}: {turn['content']}")
        return "\n".join(context)
    
    def clear(self):
        """清空短期记忆"""
        self.conversation_history = []
    
    def set_preference(self, key: str, value: Any):
        """设置用户偏好"""
        self.user_preferences[key] = value
    
    def get_preference(self, key: str) -> Any:
        """获取用户偏好"""
        return self.user_preferences.get(key)


memory = Memory(max_turns=10)
print("\n💾 记忆管理器初始化完成")

# ============================================================
# 第四部分：ReAct Agent 实现
# ============================================================
print("\n" + "=" * 70)
print("第四部分：ReAct Agent 实现")
print("=" * 70)


class ReActAgent:
    """
    ReAct Agent 实现
    
    ReAct = Reasoning + Acting
    循环: Thought → Action → Observation → ... → Answer
    """
    
    def __init__(self, tools: Dict[str, Tool], memory: Memory, max_iterations: int = 5):
        self.tools = tools
        self.memory = memory
        self.max_iterations = max_iterations
        
        # 构建工具描述
        self.tools_description = self._build_tools_description()
    
    def _build_tools_description(self) -> str:
        """构建工具描述文本"""
        descriptions = []
        for name, tool in self.tools.items():
            params = ", ".join([f"{k}={v}" for k, v in tool.parameters.items()])
            descriptions.append(f"- {name}({params}): {tool.description}")
        return "\n".join(descriptions)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return f"""你是一个智能助手，可以使用以下工具来完成任务：

{self.tools_description}

请使用 ReAct 模式（思考-行动-观察）来解决问题：

格式要求：
1. 思考(Thought): 分析问题，制定计划
2. 行动(Action): 选择工具调用，格式: Action: tool_name(param1=value1, param2=value2)
3. 观察(Observation): 工具返回的结果（由系统自动提供）
4. 最终回答(Final Answer): 任务完成后的总结回答

重要规则：
- 每次只能调用一个工具
- 如果不需要工具，直接给出 Final Answer
- 最多执行 {self.max_iterations} 轮行动
- 保持思考过程清晰、有条理
"""
    
    def _parse_action(self, text: str) -> tuple:
        """从文本中解析 Action"""
        import re
        # 匹配 Action: tool_name(param=value)
        pattern = r'Action:\s*(\w+)\s*\(([^)]+)\)'
        match = re.search(pattern, text)
        
        if match:
            tool_name = match.group(1)
            params_str = match.group(2)
            
            # 解析参数
            params = {}
            for param in params_str.split(','):
                if '=' in param:
                    key, value = param.split('=', 1)
                    params[key.strip()] = value.strip().strip('"\'')
            
            return tool_name, params
        
        return None, None
    
    def _generate_thought(self, user_input: str, context: str = "") -> str:
        """
        模拟 LLM 生成思考过程
        实际应用中应调用 GPT-4/Claude 等大模型 API
        """
        # 这里使用规则-based 模拟，实际应调用 LLM
        thought_chain = []
        
        # 简单规则：判断是否需要工具
        if "搜索" in user_input or "是谁" in user_input or "什么是" in user_input:
            thought_chain.append("Thought: 这是一个查询类问题，我需要使用搜索工具获取最新信息。")
            # 提取查询关键词
            query = user_input.replace("搜索", "").replace("是谁", "").replace("什么是", "").strip()
            thought_chain.append(f"Action: search(query=\"{query}\")")
            
        elif "计算" in user_input or any(op in user_input for op in ['+', '-', '*', '/']):
            thought_chain.append("Thought: 这是一个数学计算问题，我需要使用计算器。")
            # 提取表达式
            import re
            expr_match = re.search(r'[\d\+\-\*\/\(\)\.\s]+', user_input)
            if expr_match:
                expr = expr_match.group().strip()
                thought_chain.append(f"Action: calculator(expression=\"{expr}\")")
                
        elif "天气" in user_input:
            thought_chain.append("Thought: 用户想查询天气，我需要使用天气工具。")
            # 提取城市名（简单规则）
            import re
            cities = ["北京", "上海", "广州", "深圳", "杭州", "成都"]
            for city in cities:
                if city in user_input:
                    thought_chain.append(f"Action: weather(location=\"{city}\")")
                    break
            else:
                thought_chain.append('Action: weather(location="北京")')
                
        elif "时间" in user_input or "几点" in user_input:
            thought_chain.append("Thought: 用户询问当前时间，我需要使用时间工具。")
            thought_chain.append("Action: datetime()")
            
        else:
            thought_chain.append("Thought: 这是一个可以直接回答的通用问题。")
            thought_chain.append("Final Answer: 感谢您的提问。作为AI助手，我可以帮您搜索信息、进行计算、查询天气等。请问有什么具体任务需要我协助吗？")
        
        return "\n".join(thought_chain)
    
    def run(self, user_input: str) -> str:
        """
        执行 ReAct 循环
        """
        print(f"\n👤 用户: {user_input}")
        print("-" * 50)
        
        # 添加到记忆
        self.memory.add_turn("user", user_input)
        
        # ReAct 循环
        iteration = 0
        final_answer = None
        current_thought = self._generate_thought(user_input)
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 第 {iteration} 轮:")
            
            # 输出思考过程
            print(current_thought)
            
            # 检查是否已有最终答案
            if "Final Answer:" in current_thought:
                final_answer = current_thought.split("Final Answer:")[1].strip()
                break
            
            # 解析 Action
            tool_name, params = self._parse_action(current_thought)
            
            if tool_name and tool_name in self.tools:
                # 执行工具
                tool = self.tools[tool_name]
                observation = tool.execute(**params)
                
                print(f"\n👁️  Observation: {observation}")
                
                # 生成下一轮思考（实际应用中会基于观察结果继续）
                # 这里简化处理：直接给出最终答案
                final_answer = observation
                break
            else:
                # 没有可执行的工具
                if not final_answer:
                    final_answer = "我需要更多信息来回答这个问题。"
                break
        
        # 添加助手回复到记忆
        self.memory.add_turn("assistant", final_answer or "未能获取有效回答")
        
        print(f"\n🤖 最终回答: {final_answer}")
        return final_answer


# 初始化 Agent
agent = ReActAgent(tools=TOOLS, memory=memory, max_iterations=5)

print("\n🤖 ReAct Agent 初始化完成")
print(f"   工具数量: {len(TOOLS)}")
print(f"   最大迭代次数: {agent.max_iterations}")

# ============================================================
# 第五部分：Agent 测试
# ============================================================
print("\n" + "=" * 70)
print("第五部分：Agent 测试")
print("=" * 70)

test_cases = [
    "2024年诺贝尔文学奖得主是谁？",
    "计算 (100 + 200) * 3 等于多少",
    "北京今天天气怎么样",
    "现在几点了",
]

for test_input in test_cases:
    print("\n" + "=" * 50)
    agent.run(test_input)
    print("=" * 50)

# ============================================================
# 第六部分：高级 Agent 功能
# ============================================================
print("\n" + "=" * 70)
print("第六部分：高级 Agent 功能")
print("=" * 70)

print("""
【进阶功能】

1. 多步推理 Multi-hop Reasoning:
   Q: "2024年诺贝尔文学奖得主的作品有哪些？"
   Step 1: 搜索得主 → 韩江
   Step 2: 搜索韩江的作品 → 《素食者》等
   Step 3: 整合回答

2. 自我反思 Self-Reflection:
   - 检查结果是否合理
   - 发现错误时重新规划
   - 优化行动策略

3. 工具链 Tool Chaining:
   - 多个工具串联使用
   - 前一个工具的输出作为后一个的输入
   - 例如: 搜索 → 计算 → 格式化

4. 规划 Planning:
   - 复杂任务分解为子任务
   - 制定执行计划
   - 按优先级排序

【实际部署建议】

1. 使用真实 LLM API:
   - OpenAI GPT-4
   - Anthropic Claude
   - 国产: 文心一言、通义千问、ChatGLM

2. 集成真实工具:
   - Serper API (Google搜索)
   - Wolfram Alpha (数学计算)
   - OpenWeatherMap (天气)
   - Python解释器 (代码执行)

3. 添加向量数据库:
   - 长期记忆存储
   - 相似案例检索
   - 知识库增强
""")

# ============================================================
# 第七部分：多轮对话演示
# ============================================================
print("\n" + "=" * 70)
print("第七部分：多轮对话演示")
print("=" * 70)

print("\n📝 多轮对话示例（使用记忆）:\n")

# 创建新的 Agent 实例
chat_agent = ReActAgent(tools=TOOLS, memory=Memory(max_turns=5), max_iterations=3)

conversation = [
    "你好，我想查询北京明天的天气",
    "那上海呢？",  # 指代消解，需要上下文
    "帮我计算一下 25 * 4",
]

for msg in conversation:
    print(f"\n{'='*50}")
    chat_agent.run(msg)
    print(f"\n💾 当前记忆:\n{chat_agent.memory.get_context()}")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print("""
✅ 本项目完成了：

1. AI Agent 核心架构
   - ReAct 模式 (Reasoning + Acting)
   - 感知-思考-行动-观察循环
   - 多轮迭代直到任务完成

2. 工具系统(Tools)
   - 搜索引擎: 获取事实性信息
   - 计算器: 执行数学运算
   - 天气查询: 获取实时数据
   - 时间查询: 获取当前时间
   - 工具注册和调用机制

3. 记忆管理(Memory)
   - 短期记忆: 对话历史
   - 长期记忆: 用户偏好
   - 上下文窗口管理

4. Agent 执行流程
   - Thought: 分析问题和制定计划
   - Action: 选择并执行工具
   - Observation: 获取工具结果
   - Final Answer: 给出最终回答

🚀 进阶方向：
   1. 接入真实 LLM API (GPT-4/Claude)
   2. 集成更多工具 (代码执行、数据库、API调用)
   3. 实现多步推理和复杂任务规划
   4. 添加向量数据库作为长期记忆
   5. 实现 Agent 自我反思和错误修正
   6. 开发可视化界面或Web服务
   7. 学习 LangChain/LlamaIndex 的 Agent 模块
   8. 实现 Multi-Agent 协作系统

📚 推荐资源：
   - ReAct 论文: arxiv.org/abs/2210.03629
   - LangChain Agents: python.langchain.com/docs/modules/agents
   - AutoGPT: github.com/Significant-Gravitas/AutoGPT
   - BabyAGI: github.com/yoheinakajima/babyagi
""")

print("\n" + "=" * 70)
print("AI Agent 开发实战完成！")
print("=" * 70)
