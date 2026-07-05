#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Tool Schema 模板
"""

from datetime import datetime
from email import message
import os
from typing import Literal

from langchain_core.messages.tool import tool_call
from langchain_core.messages.utils import _convert_to_openai_tool_calls
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, prompt
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, json_schema
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
    if not _api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    print("=" * 60)

    messages = [
        HumanMessage(content="1+1=?")
    ]

    model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="none")
    rprint(model_with_tools.invoke(messages))

    model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="auto")
    rprint(model_with_tools.invoke(messages))

    # some tool_choice is not supported in thinking mode
    model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="required")
    rprint(model_with_tools.invoke(messages,extra_body={"thinking": {"type": "disabled"}}))

    # some tool_choice is not supported in thinking mode
    model_with_tools = _chat_deepseek().bind_tools([add, subtract], tool_choice="subtract")
    rprint(model_with_tools.invoke(messages,extra_body={"thinking": {"type": "disabled"}}))


if __name__ == "__main__":
    main()
