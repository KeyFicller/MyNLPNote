#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第38课：LangChain Tool Choices
================================

运行方式：
    python examples/langchain/07_langchain_tool_choices.py

相关笔记：
    notes/phase-langchain/07_LangChain_Tool_Choices.md

连续四次 invoke 对比 tool_choice。
"""

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from rich import print as rprint

from deepseek_client import api_key, chat_deepseek

@tool
def add(a: int, b: int) -> int:
    """
    计算 a + b 的和
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """
    计算 a - b 的差
    """
    return a - b


def main() -> None:
    if not api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    print("=" * 60)

    messages = [
        HumanMessage(content="1+1=?")
    ]

    model_with_tools = chat_deepseek().bind_tools([add, subtract], tool_choice="none")
    rprint(model_with_tools.invoke(messages))

    model_with_tools = chat_deepseek().bind_tools([add, subtract], tool_choice="auto")
    rprint(model_with_tools.invoke(messages))

    # some tool_choice is not supported in thinking mode
    model_with_tools = chat_deepseek().bind_tools([add, subtract], tool_choice="required")
    rprint(model_with_tools.invoke(messages,extra_body={"thinking": {"type": "disabled"}}))

    # some tool_choice is not supported in thinking mode
    model_with_tools = chat_deepseek().bind_tools([add, subtract], tool_choice="subtract")
    rprint(model_with_tools.invoke(messages,extra_body={"thinking": {"type": "disabled"}}))


if __name__ == "__main__":
    main()
