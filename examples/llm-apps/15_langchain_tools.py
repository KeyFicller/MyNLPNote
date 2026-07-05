#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 进阶：Prompt 模板
"""

from datetime import datetime
from email import message
import os

from langchain_core.messages.tool import tool_call
from langchain_core.messages.utils import _convert_to_openai_tool_calls
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, prompt
from langchain_deepseek import ChatDeepSeek
from openai.types.responses import response
from rich import print as rprint
from langchain_core.utils.function_calling import convert_to_openai_tool

MODEL = "deepseek-v4-pro"


def _api_base() -> str | None:
    return os.getenv("DEEPSEEK_BASE_URL")


def _api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")


def _chat_deepseek() -> ChatDeepSeek:
    kwargs: dict = {"model": MODEL, "api_key": _api_key()}
    if base := _api_base():
        kwargs["api_base"] = base


    return ChatDeepSeek(**kwargs)

@tool
def get_current_time(city: str) -> str:
    """获取当前时间"""
    return f"当前{city}时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

@tool(description="加法函数", parse_docstring=True)
def add(a: int, b: int) -> int:
    """
    计算 a + b 的和

    Args:
        a: 加数
        b: 加数

    Returns:
        int: 和
    """
    return a + b

def _test_tool_call_with_tool_specification():
    print("测试 get_current_time 工具: invoke")
    result = get_current_time.invoke({"city":"北京"})
    print(result)

    print("-" * 60)
    model_with_tools = _chat_deepseek().bind_tools([get_current_time])
    messages = [
        HumanMessage(content="现在上海是什么时间？")
    ]
    response = model_with_tools.invoke(messages)
    #print(response.content)
    if response.tool_calls:
        print("Tool calls is triggered")
        rprint(response.tool_calls[0])
        if response.tool_calls[0]["name"] == "get_current_time":
            tool_message = get_current_time.invoke(response.tool_calls[0])
            messages.extend([response, tool_message])
            final_response = model_with_tools.invoke(messages)
            print(final_response.content)

    else:
        print("Tool calls is not triggered")

'''
注意:
   空行分隔（即使包含空白字符也不可以）
'''
def get_weather(city: str) -> str:
    """
    获取某城市天气

    Args:
        city: 城市名称

    Returns:
        返回城市的天气
    """
    return f"当前{city}天气：晴天"

def _test_tool_call_without_tool_specification():
    print("测试 get_weather 工具: invoke")
    converted_tool = convert_to_openai_tool(get_weather)
    rprint(converted_tool)

    messages = [
        HumanMessage(content="现在上海是什么天气？")
    ]
    #model_with_tools = _chat_deepseek().bind_tools([converted_tool])
    model_with_tools = _chat_deepseek().bind_tools([get_weather])
    response = model_with_tools.invoke(messages)
    rprint(response)

def _test_tool_call_with_tool_modiifier():
    rprint(convert_to_openai_tool(add))

    messages = [
        HumanMessage(content="计算 1 + 2 的和")
    ]
    response = _chat_deepseek().bind_tools([add]).invoke(messages)
    rprint(response)

def main() -> None:
    if not _api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    print("=" * 60)

    #_test_tool_call_with_tool_specification()
    #_test_tool_call_without_tool_specification()
    _test_tool_call_with_tool_modiifier()


if __name__ == "__main__":
    main()
