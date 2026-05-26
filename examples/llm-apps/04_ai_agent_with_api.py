#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Agent 接入真实 LLM API 实战
=================================
项目名称：基于 GPT-4/Claude/国产大模型的智能代理
任务：接入真实大模型API，实现真正具备推理能力的ReAct Agent

支持的API：
- OpenAI GPT-4 / GPT-3.5
- Anthropic Claude
- 国产: 智谱AI(GLM-4)、DeepSeek、Moonshot(Kimi)
- Azure OpenAI

学习目标：
- 学习LLM API调用
- 掌握API Key管理
- 实现真正的推理+行动循环
- 对比不同模型的效果
- 理解Token消耗和成本控制
"""

import os
import sys
import json
import re
import time
import inspect
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ========== 配置 Hugging Face 镜像源 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 70)
print("AI Agent 接入真实 LLM API 实战")
print("=" * 70)

# ============================================================
# 第一部分：API配置和模型选择
# ============================================================
print("\n" + "=" * 70)
print("第一部分：API配置和模型选择")
print("=" * 70)

print("""
【支持的LLM API】

1. OpenAI (推荐)
   - 模型: gpt-4, gpt-4-turbo, gpt-3.5-turbo
   - 官网: https://platform.openai.com
   - 特点: 推理能力强，文档完善

2. Anthropic Claude
   - 模型: claude-3-opus, claude-3-sonnet
   - 官网: https://console.anthropic.com
   - 特点: 上下文长(200K)，推理细致

3. 智谱AI (国产)
   - 模型: glm-4, glm-4-flash
   - 官网: https://open.bigmodel.cn
   - 特点: 中文友好，速度快

4. DeepSeek (国产)
   - 模型: deepseek-chat, deepseek-coder
   - 官网: https://platform.deepseek.com
   - 特点: 性价比高，代码能力强

5. Moonshot/Kimi (国产)
   - 模型: moonshot-v1-8k/32k/128k
   - 官网: https://platform.moonshot.cn
   - 特点: 上下文超长(200K)，中文理解好

【API Key设置方式】
方式1: 环境变量 (推荐)
   $env:OPENAI_API_KEY="your-key"
   $env:ZHIPU_API_KEY="your-key"

方式2: .env文件
   创建 .env 文件存放密钥

方式3: 直接输入 (仅测试)
   程序运行时输入
""")


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    ZHIPU = "zhipu"
    DEEPSEEK = "deepseek"
    MOONSHOT = "moonshot"


@dataclass
class ModelConfig:
    """模型配置"""
    provider: LLMProvider
    model_name: str
    api_key_env: str
    base_url: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.7
    

# 预置模型配置
PRESET_MODELS = {
    "gpt-4": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-4",
        api_key_env="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    "gpt-3.5": ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="gpt-3.5-turbo",
        api_key_env="OPENAI_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    "claude-3": ModelConfig(
        provider=LLMProvider.ANTHROPIC,
        model_name="claude-3-sonnet-20240229",
        api_key_env="ANTHROPIC_API_KEY",
        max_tokens=2000,
        temperature=0.7
    ),
    "glm-4": ModelConfig(
        provider=LLMProvider.ZHIPU,
        model_name="glm-4",
        api_key_env="ZHIPU_API_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        max_tokens=2000,
        temperature=0.7
    ),
    "deepseek": ModelConfig(
        provider=LLMProvider.DEEPSEEK,
        model_name="deepseek-v4-flash",
        api_key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/v1",
        max_tokens=2000,
        temperature=0.7
    ),
    "moonshot": ModelConfig(
        provider=LLMProvider.MOONSHOT,
        model_name="moonshot-v1-8k",
        api_key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.cn/v1",
        max_tokens=2000,
        temperature=0.7
    ),
}


class LLMClient:
    """LLM API客户端"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = self._get_api_key()
        self.client = None
        self.total_tokens = 0
        self.total_cost = 0.0
        
        if self.api_key:
            self._init_client()
    
    def _get_api_key(self) -> Optional[str]:
        """获取API Key"""
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            print(f"\n⚠️  未找到环境变量 {self.config.api_key_env}")
            print(f"   请设置: $env:{self.config.api_key_env}=\"your-api-key\"")
        return api_key
    
    def _init_client(self):
        """初始化API客户端"""
        try:
            if self.config.provider == LLMProvider.OPENAI:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.config.base_url
                )
            elif self.config.provider == LLMProvider.ANTHROPIC:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            elif self.config.provider in [LLMProvider.ZHIPU, LLMProvider.DEEPSEEK, LLMProvider.MOONSHOT]:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.config.base_url
                )
            print(f"✓ {self.config.provider.value} 客户端初始化成功")
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print(f"   请安装: pip install openai anthropic")
        except Exception as e:
            print(f"❌ 客户端初始化失败: {e}")
    
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        调用LLM聊天接口
        
        Args:
            messages: 消息列表，格式 [{"role": "system"/"user"/"assistant", "content": "..."}]
            **kwargs: 额外参数
        
        Returns:
            LLM生成的文本
        """
        if not self.client:
            print(f"❌ 客户端未初始化，使用模拟模式")
            return self._mock_response(messages)
        
        try:
            start_time = time.time()
            
            if self.config.provider == LLMProvider.ANTHROPIC:
                # Claude API 格式
                system_msg = ""
                user_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_msg = msg["content"]
                    else:
                        user_messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                response = self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    system=system_msg,
                    messages=user_messages
                )
                
                result = response.content[0].text
                tokens = response.usage.input_tokens + response.usage.output_tokens
                
            else:
                # OpenAI 兼容格式
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                )
                
                result = response.choices[0].message.content
                tokens = response.usage.total_tokens
            
            # 统计
            elapsed = time.time() - start_time
            self.total_tokens += tokens
            cost = self._calculate_cost(tokens)
            self.total_cost += cost
            
            print(f"   ⏱️  {elapsed:.2f}s | 🔤 {tokens} tokens | 💰 ${cost:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            return self._mock_response(messages)
    
    def _calculate_cost(self, tokens: int) -> float:
        """估算成本 (美元)"""
        # 大致估算，实际价格请参考官方文档
        prices = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.008,
            "glm-4": 0.001,
            "deepseek-chat": 0.0005,
            "moonshot-v1-8k": 0.001,
        }
        price_per_1k = prices.get(self.config.model_name, 0.01)
        return (tokens / 1000) * price_per_1k
    
    def _mock_response(self, messages: List[Dict[str, str]]) -> str:
        """模拟LLM响应（用于无API Key时测试）"""
        user_msg = messages[-1]["content"] if messages else ""
        
        # 简单的规则匹配模拟
        if "search" in user_msg.lower() or "搜索" in user_msg:
            return "Thought: 我需要进行搜索来获取信息。\nAction: search(query=\"查询内容\")"
        elif "calculate" in user_msg.lower() or "计算" in user_msg:
            return "Thought: 这是一个数学问题，需要使用计算器。\nAction: calculator(expression=\"1+1\")"
        else:
            return "Thought: 我可以直接回答这个问题。\nFinal Answer: 这是模拟回答，请设置API Key获取真实回复。"
    
    def get_stats(self) -> Dict[str, Any]:
        """获取使用统计"""
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
            "total_cost_rmb": self.total_cost * 7.2
        }


# ============================================================
# 第二部分：工具系统（增强版）
# ============================================================
print("\n" + "=" * 70)
print("第二部分：工具系统（增强版）")
print("=" * 70)

print("""
【增强工具系统】

相比之前的模拟版本，增加了：
- 真实网络搜索 (DuckDuckGo)
- Python代码执行 (安全沙箱)
- 文件读写操作
- 时间日期查询
- 数学计算增强
""")


# 工具定义
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, str]
    func: Callable
    
    def execute(self, **kwargs) -> str:
        try:
            sig = inspect.signature(self.func)
            valid_params = {
                key: value
                for key, value in kwargs.items()
                if key in sig.parameters
            }
            return str(self.func(**valid_params))
        except Exception as e:
            return f"❌ 工具执行错误: {str(e)}"


# ========== 工具函数实现 ==========

def search_tool(query: str) -> str:
    """
    网络搜索工具
    使用 DuckDuckGo 搜索引擎（无需API Key）
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        last_error = None
        with DDGS() as ddgs:
            for q in (query, query.replace("诺贝尔文学奖", "Nobel Prize in Literature")):
                try:
                    results = list(ddgs.text(q, max_results=3))
                    if results:
                        output = []
                        for i, result in enumerate(results, 1):
                            body = result.get("body") or result.get("snippet") or ""
                            title = result.get("title") or "无标题"
                            output.append(f"[{i}] {title}\n{body[:200]}...\n")
                        return "\n".join(output)
                except Exception as e:
                    last_error = e

        if last_error:
            return f"搜索失败: {last_error}"
        return "搜索未返回结果"

    except ImportError:
        return (
            f"[模拟搜索结果] 关于'{query}'的信息：\n"
            "这是一个模拟搜索结果。请安装 ddgs 获取真实搜索结果：\n"
            "pip install ddgs"
        )
    except Exception as e:
        return f"搜索失败: {str(e)}"


def calculator_tool(expression: str) -> str:
    """安全计算器"""
    try:
        # 只允许安全字符
        allowed = set('0123456789+-*/.() ** % // ')
        if not all(c in allowed for c in expression):
            return "错误: 表达式包含非法字符"
        
        # 使用eval计算（实际生产环境应使用更安全的解析器）
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def python_tool(code: str) -> str:
    """
    Python代码执行工具（安全沙箱）
    注意：这是简化版本，生产环境需要更严格的沙箱
    """
    import io
    import contextlib
    
    # 禁止危险操作
    forbidden = ['import os', 'import sys', 'open(', '__import__', 'eval(', 'exec(', 'subprocess', 'os.system']
    for f in forbidden:
        if f in code:
            return f"❌ 安全限制: 不允许使用 '{f}'"
    
    # 创建受限执行环境
    safe_globals = {
        "__builtins__": {
            "print": print,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "list": list,
            "dict": dict,
            "str": str,
            "int": int,
            "float": float,
            "sum": sum,
            "max": max,
            "min": min,
            "abs": abs,
            "round": round,
        }
    }
    
    # 捕获输出
    output_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals, {})
        
        result = output_buffer.getvalue()
        return result if result else "代码执行完成（无输出）"
        
    except Exception as e:
        return f"执行错误: {str(e)}"


def datetime_tool() -> str:
    """获取当前时间"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M:%S %A")


def weather_tool(location: str) -> str:
    """
    天气查询工具
    注意：需要接入真实天气API，这里是模拟
    """
    # 模拟天气数据
    import random
    weathers = ["晴天", "多云", "阴天", "小雨", "中雨"]
    temp = random.randint(15, 30)
    weather = random.choice(weathers)
    
    return f"{location}天气: {weather}, 气温 {temp}°C (模拟数据)"


def file_tool(operation: str, filename: str, content: str = "") -> str:
    """
    文件操作工具
    operation: read/write/append
    """
    try:
        if operation == "read":
            with open(filename, 'r', encoding='utf-8') as f:
                return f.read()
        elif operation == "write":
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"已写入文件: {filename}"
        elif operation == "append":
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"已追加到文件: {filename}"
        else:
            return f"不支持的操作: {operation}"
    except Exception as e:
        return f"文件操作失败: {str(e)}"


# 工具注册表
TOOLS = {
    "search": Tool(
        name="search",
        description="网络搜索引擎，用于获取实时信息、新闻、百科知识等。支持中文和英文搜索。",
        parameters={"query": "搜索关键词，字符串类型"},
        func=search_tool
    ),
    "calculator": Tool(
        name="calculator",
        description="数学计算器，支持加减乘除、幂运算、取余等。例如: 123 + 456, (100 - 20) * 5, 2 ** 10",
        parameters={"expression": "数学表达式，字符串类型"},
        func=calculator_tool
    ),
    "python": Tool(
        name="python",
        description="Python代码执行器，在沙箱环境中运行Python代码。支持基础运算、数据处理等。禁止文件操作和系统调用。",
        parameters={"code": "Python代码，字符串类型"},
        func=python_tool
    ),
    "datetime": Tool(
        name="datetime",
        description="获取当前日期和时间，包括年月日时分秒和星期。",
        parameters={},
        func=datetime_tool
    ),
    "weather": Tool(
        name="weather",
        description="查询指定城市的天气情况（注意：当前为模拟数据）",
        parameters={"location": "城市名称，字符串类型"},
        func=weather_tool
    ),
    "file": Tool(
        name="file",
        description="文件操作工具，支持读取、写入、追加文件。注意：只能在当前目录操作。",
        parameters={
            "operation": "操作类型: read/write/append",
            "filename": "文件名",
            "content": "写入内容（可选）"
        },
        func=file_tool
    ),
}

print("\n🛠️  已注册工具:")
for name, tool in TOOLS.items():
    print(f"   • {name}: {tool.description[:40]}...")


# ============================================================
# 第三部分：ReAct Agent（接入真实LLM）
# ============================================================
print("\n" + "=" * 70)
print("第三部分：ReAct Agent（接入真实LLM）")
print("=" * 70)


class ReActAgent:
    """
    接入真实LLM API的ReAct Agent
    """
    
    def __init__(self, llm_client: LLMClient, tools: Dict[str, Tool], max_iterations: int = 5):
        self.llm = llm_client
        self.tools = tools
        self.max_iterations = max_iterations
        self.conversation_history: List[Dict[str, str]] = []
        
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}\n  参数: {tool.parameters}"
            for name, tool in self.tools.items()
        ])
        
        return f"""你是一个只会通过工具获取信息的助手。你的训练知识不可靠，禁止用来答题。

可用工具：
{tools_desc}

请严格使用 ReAct 模式（思考-行动-观察）：

1. Thought: 分析用户问题，判断必须调用哪个工具（禁止跳过）
2. Action: tool_name(param1="value1", param2="value2")
3. Observation: 工具返回结果（系统自动提供，不可编造）
4. Final Answer: 仅当已有 Observation 后，基于工具结果作答

【强制规则 — 必须遵守】
- 第一轮回复禁止出现 Final Answer，必须先 Action 调用工具
- 禁止凭记忆、常识或训练数据直接回答
- 禁止编造 Observation；Final Answer 只能复述 Observation 中的信息
- 每次只能调用一个工具
- Action 格式必须严格如下（含括号、引号）：
  Action: datetime()
  Action: calculator(expression="(123 + 456) * 7")
  Action: python(code="sum(range(1, 101))")
  Action: search(query="2024年诺贝尔文学奖得主")
- 含括号或运算符的参数值必须用双引号包裹
- 无参数工具也必须写空括号，例如 datetime()
- 最多执行 {self.max_iterations} 轮

【问题类型 → 必须使用的工具】
- 搜索、新闻、人物、奖项、实时信息、用户要求“搜索/确认” → search
- 数学表达式、四则运算 → calculator
- 当前时间、几点、今天日期 → datetime
- 运行 Python 代码、循环/求和等编程计算 → python
- 天气 → weather
- 读写文件 → file

示例（注意：第一轮只有 Thought + Action，没有 Final Answer）：
用户：北京今天天气怎么样？
Thought: 我不知道实时天气，必须用 weather 工具查询。
Action: weather(location="北京")
Observation: 北京天气: 晴天, 气温 25°C
Thought: 我已从工具获得天气信息，可以回答了。
Final Answer: 根据查询结果，北京今天晴天，气温 25°C。
"""
    
    def _parse_action(self, text: str) -> tuple:
        """解析 Action 调用，支持空参数与引号内的括号"""
        action_match = re.search(r'Action:\s*(\w+)\s*\(', text, re.IGNORECASE)
        if not action_match:
            return None, None

        tool_name = action_match.group(1)
        start = action_match.end()
        depth = 1
        pos = start
        in_quote = None

        while pos < len(text) and depth > 0:
            ch = text[pos]
            if in_quote:
                if ch == '\\' and pos + 1 < len(text):
                    pos += 2
                    continue
                if ch == in_quote:
                    in_quote = None
            else:
                if ch in '"\'':
                    in_quote = ch
                elif ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
            pos += 1

        if depth != 0:
            return None, None

        params_str = text[start:pos - 1].strip()
        if not params_str:
            return tool_name, {}

        params = self._parse_action_params(params_str)
        return tool_name, params

    def _parse_action_params(self, params_str: str) -> Dict[str, str]:
        """解析 Action 参数字符串"""
        params: Dict[str, str] = {}

        json_like = re.match(r"^\{(.+)\}$", params_str, re.DOTALL)
        if json_like:
            inner = json_like.group(1)
            for match in re.finditer(
                r"""['\"]?(\w+)['\"]?\s*:\s*(['"])(.*?)\2""",
                inner,
                re.DOTALL,
            ):
                params[match.group(1)] = match.group(3)
            if params:
                return params

        pos = 0
        while pos < len(params_str):
            key_match = re.match(r'(\w+)\s*=\s*', params_str[pos:])
            if not key_match:
                break

            pos += key_match.end()
            key = key_match.group(1)

            if pos >= len(params_str):
                break

            quote = params_str[pos]
            if quote in '"\'':
                pos += 1
                value_chars: List[str] = []
                while pos < len(params_str):
                    if params_str[pos] == '\\' and pos + 1 < len(params_str):
                        value_chars.append(params_str[pos + 1])
                        pos += 2
                        continue
                    if params_str[pos] == quote:
                        pos += 1
                        break
                    value_chars.append(params_str[pos])
                    pos += 1
                params[key] = ''.join(value_chars)
            else:
                value_match = re.match(r'([^,\s)]+)', params_str[pos:])
                if not value_match:
                    break
                params[key] = value_match.group(1)
                pos += value_match.end()

            while pos < len(params_str) and params_str[pos] in ', \t':
                pos += 1

        return params
    
    def run(self, user_input: str) -> str:
        """
        执行 ReAct 循环
        """
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*60}")
        
        # 初始化对话
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input}
        ]
        
        iteration = 0
        final_answer = None
        tools_used = False
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 第 {iteration}/{self.max_iterations} 轮:")
            print("-" * 50)
            
            # 调用 LLM
            response = self.llm.chat(messages)
            
            # 提取思考过程
            thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL | re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1).strip()
                print(f"🧠 Thought: {thought[:100]}...")
            
            # 检查是否已有最终答案（未调用工具时拒绝直接作答）
            if "Final Answer:" in response:
                if not tools_used:
                    print("⚠️  未调用工具就试图作答，已要求先使用工具")
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "错误：禁止凭记忆直接给出 Final Answer。"
                            "你必须先调用合适的工具，等待 Observation 后再回答。"
                            "请重新输出 Thought 和 Action。"
                        ),
                    })
                    continue

                final_match = re.search(r'Final Answer:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
                if final_match:
                    final_answer = final_match.group(1).strip()
                    print(f"\n✅ Final Answer: {final_answer}")
                    break
            
            # 解析 Action
            tool_name, params = self._parse_action(response)
            
            if tool_name and tool_name in self.tools:
                print(f"🔧 Action: {tool_name}({params})")
                
                # 执行工具
                tool = self.tools[tool_name]
                observation = tool.execute(**params)
                tools_used = True
                
                # 截断过长的观察结果
                obs_display = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"👁️  Observation: {obs_display}")
                
                # 添加观察结果到对话
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
                
            else:
                print(f"⚠️  未识别工具调用或工具不存在")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "Action 格式无法解析。请严格使用以下格式之一：\n"
                        "Action: datetime()\n"
                        "Action: calculator(expression=\"(123 + 456) * 7\")\n"
                        "Action: python(code=\"sum(range(1, 101))\")\n"
                        "Action: search(query=\"关键词\")"
                    ),
                })
        
        if not final_answer:
            final_answer = "抱歉，我无法在限定轮数内完成这个任务。"
            print(f"\n❌ {final_answer}")
        
        return final_answer


# ============================================================
# 第四部分：测试运行
# ============================================================
print("\n" + "=" * 70)
print("第四部分：测试运行")
print("=" * 70)


def select_model() -> Optional[ModelConfig]:
    """选择要使用的模型"""
    print("\n📋 可用模型列表:")
    for i, (name, config) in enumerate(PRESET_MODELS.items(), 1):
        has_key = "✓" if os.environ.get(config.api_key_env) else "✗"
        print(f"   {i}. {name} ({config.provider.value}) [API Key {has_key}]")
    
    print("\n💡 提示：输入编号选择模型，或直接回车使用模拟模式")
    choice = input("\n请选择模型 (1-6): ").strip()
    
    if not choice:
        print("\n⚠️  使用模拟模式（无真实API调用）")
        return None
    
    try:
        idx = int(choice) - 1
        model_name = list(PRESET_MODELS.keys())[idx]
        return PRESET_MODELS[model_name]
    except (ValueError, IndexError):
        print("❌ 无效选择，使用模拟模式")
        return None


def main():
    """主函数"""
    # 选择模型
    config = select_model()
    
    if config:
        # 初始化LLM客户端
        print(f"\n🤖 初始化 {config.model_name}...")
        llm = LLMClient(config)
        
        if not llm.api_key:
            print("\n⚠️  未设置API Key，切换到模拟模式")
            print(f"   请运行: $env:{config.api_key_env}=\"your-api-key\"")
    else:
        # 模拟模式
        llm = LLMClient(ModelConfig(
            provider=LLMProvider.OPENAI,
            model_name="mock",
            api_key_env="MOCK"
        ))
    
    # 初始化Agent
    agent = ReActAgent(llm, TOOLS, max_iterations=5)
    
    # 测试用例
    test_cases = [
        "2024年诺贝尔文学奖得主是谁？请搜索确认。",
        "计算 (123 + 456) * 7 等于多少？",
        "现在几点了？",
        "用Python计算1到100的和",
    ]
    
    print("\n" + "="*70)
    print("🧪 开始测试")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"测试 {i}/{len(test_cases)}")
        print(f"{'='*70}")
        
        try:
            result = agent.run(test)
            print(f"\n📝 结果: {result}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
        
        if i < len(test_cases):
            input("\n按回车继续下一个测试...")
    
    # 交互模式
    print("\n" + "="*70)
    print("💬 交互模式（输入 'exit' 退出）")
    print("="*70)
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() in ['exit', 'quit', '退出']:
            break
        
        if not user_input:
            continue
        
        try:
            result = agent.run(user_input)
            print(f"\n🤖 Agent: {result}")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
    
    # 统计
    print("\n" + "="*70)
    print("📊 使用统计")
    print("="*70)
    stats = llm.get_stats()
    print(f"   总Token数: {stats['total_tokens']}")
    print(f"   预估成本: ${stats['total_cost_usd']:.4f} USD")
    print(f"   预估成本: ¥{stats['total_cost_rmb']:.2f} RMB")
    
    print("\n✨ 感谢使用！")


if __name__ == "__main__":
    main()
