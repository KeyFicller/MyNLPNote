#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 核心模块 - 第25/26课共享
包含 LLM 客户端、工具系统、ReAct Agent
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

# HuggingFace 镜像（在导入 transformers 相关库之前）
_LLM_APPS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.abspath(os.path.join(_LLM_APPS_DIR, ".."))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)
from hf_mirror import setup_hf_mirror

setup_hf_mirror()


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


class ReActAgent:
    """接入真实 LLM API 的 ReAct Agent"""

    def __init__(
        self,
        llm_client: LLMClient,
        tools: Dict[str, Tool],
        max_iterations: int = 5,
        extra_system_rules: str = "",
        action_format_hints: Optional[List[str]] = None,
    ):
        self.llm = llm_client
        self.tools = tools
        self.max_iterations = max_iterations
        self.conversation_history: List[Dict[str, str]] = []
        self.extra_system_rules = extra_system_rules
        self.action_format_hints = action_format_hints or [
            'Action: datetime()',
            'Action: calculator(expression="(123 + 456) * 7")',
            'Action: python(code="sum(range(1, 101))")',
            'Action: search(query="关键词")',
        ]

    def _build_system_prompt(self) -> str:
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
- 含括号或运算符的参数值必须用双引号包裹
- 无参数工具也必须写空括号，例如 datetime()
- 最多执行 {self.max_iterations} 轮
{self.extra_system_rules}
示例（注意：第一轮只有 Thought + Action，没有 Final Answer）：
用户：北京今天天气怎么样？
Thought: 我不知道实时天气，必须用 weather 工具查询。
Action: weather(location="北京")
Observation: 北京天气: 晴天, 气温 25°C
Thought: 我已从工具获得天气信息，可以回答了。
Final Answer: 根据查询结果，北京今天晴天，气温 25°C。
"""

    def _extract_final_answer(self, text: str) -> Optional[str]:
        """从回复中提取最终答案，兼容多种写法"""
        patterns = [
            r'Final Answer:\s*(.+)',
            r'最终答案[：:]\s*(.+)',
            r'【Final Answer】\s*(.+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # 去掉常见的 markdown 包裹
                answer = re.sub(r'^[\*`_\s]+|[\*`_]+$', '', answer)
                return answer.strip()
        return None

    def _parse_action(self, text: str) -> tuple:
        # 兼容 markdown 代码块中的 Action
        code_block = re.search(r'```(?:\w*\n)?(.*?)```', text, re.DOTALL)
        if code_block and 'Action:' in code_block.group(1):
            text = code_block.group(1)

        action_match = re.search(r'Action:\s*(\w+)\s*\(', text, re.IGNORECASE)
        if not action_match:
            return None, None
        tool_name = action_match.group(1).lower()
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
        return tool_name, self._parse_action_params(params_str)

    def _parse_action_params(self, params_str: str) -> Dict[str, str]:
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
        print(f"\n{'='*60}")
        print(f"👤 用户: {user_input}")
        print(f"{'='*60}")
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_input},
        ]
        iteration = 0
        final_answer = None
        tools_used = False
        last_observation = ""
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n🔄 第 {iteration}/{self.max_iterations} 轮:")
            print("-" * 50)
            response = self.llm.chat(messages)
            thought_match = re.search(
                r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)',
                response,
                re.DOTALL | re.IGNORECASE,
            )
            if thought_match:
                thought = thought_match.group(1).strip()
                print(f"🧠 Thought: {thought[:100]}...")

            extracted = self._extract_final_answer(response)
            if extracted:
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
                final_answer = extracted
                print(f"\n✅ Final Answer: {final_answer}")
                break

            tool_name, params = self._parse_action(response)
            if tool_name and tool_name in self.tools:
                print(f"🔧 Action: {tool_name}({params})")
                observation = self.tools[tool_name].execute(**params)
                tools_used = True
                last_observation = observation
                obs_display = observation[:200] + "..." if len(observation) > 200 else observation
                print(f"👁️  Observation: {obs_display}")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        f"Observation: {observation}\n\n"
                        "请基于以上 Observation 直接作答，不要再调用工具。"
                        "下一回复格式：\nThought: 简要总结\nFinal Answer: 你的答案"
                    ),
                })
            elif tools_used:
                # 已有工具结果：模型可能用非标准格式直接回答
                plain = response.strip()
                if plain and len(plain) > 20 and not re.search(r'Action:\s*\w+\s*\(', plain, re.I):
                    final_answer = plain
                    if thought_match:
                        final_answer = re.sub(
                            r'^Thought:.*?(?=Final Answer:|$)',
                            '',
                            plain,
                            flags=re.DOTALL | re.IGNORECASE,
                        ).strip() or plain
                    print(f"\n✅ Final Answer: {final_answer[:200]}...")
                    break
                print("⚠️  已有 Observation，请输出 Final Answer")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "你已收到工具结果，不要再次调用工具。"
                        "请严格输出：\nThought: ...\nFinal Answer: ..."
                        f"\n\n参考 Observation:\n{last_observation[:500]}"
                    ),
                })
            else:
                print("⚠️  未识别工具调用或工具不存在")
                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": (
                        "Action 格式无法解析。请严格使用以下格式之一：\n"
                        + "\n".join(self.action_format_hints)
                    ),
                })
        if not final_answer:
            final_answer = "抱歉，我无法在限定轮数内完成这个任务。"
            print(f"\n❌ {final_answer}")
        return final_answer


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
        return PRESET_MODELS[list(PRESET_MODELS.keys())[idx]]
    except (ValueError, IndexError):
        print("❌ 无效选择，使用模拟模式")
        return None


def create_llm_client(config: Optional[ModelConfig]) -> LLMClient:
    """根据配置创建 LLM 客户端"""
    if config:
        print(f"\n🤖 初始化 {config.model_name}...")
        llm = LLMClient(config)
        if not llm.api_key:
            print("\n⚠️  未设置 API Key，切换到模拟模式")
            print(f'   请设置环境变量: {config.api_key_env}')
        return llm
    return LLMClient(ModelConfig(
        provider=LLMProvider.OPENAI,
        model_name="mock",
        api_key_env="MOCK",
    ))
