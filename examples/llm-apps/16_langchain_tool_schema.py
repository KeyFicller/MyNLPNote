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

class WeatherInput(BaseModel):
    city: str = Field(
        description="城市名称",
        default="北京"
    )
    unit: Literal["C", "F"]
    fore_cast : bool = Field(
        description="是否需要天气预报",
        default=False
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str, fore_cast: bool = False) -> str:
    """
    获取某城市天气
    """
    return f"当前{city}天气：晴天，温度：20{unit}。 - Forecast: {fore_cast}"

json_schema = {
    'properties': {
        'city': {
            'default': '北京',
            'description': '城市名称',
            'type': 'string'
        },
        'unit': {'enum': ['C', 'F'], 'type': 'string'},
        'fore_cast': {
            'default': False,
            'description': '是否需要天气预报',
            'type': 'boolean'
        }
    },
    'required': ['unit'],
    'type': 'object'
}

@tool(args_schema=json_schema)
def get_weather2(city: str, unit: str, fore_cast: bool = False) -> str:
    """
    获取某城市天气
    """
    return f"当前{city}天气：晴天，温度：20{unit}。 - Forecast: {fore_cast}"

def main() -> None:
    if not _api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    print("=" * 60)

    rprint(convert_to_openai_tool(get_weather))

    messages = [
        HumanMessage(content="明天上海是什么天气？")
    ]
    response = _chat_deepseek().bind_tools([get_weather]).invoke(messages)
    rprint(response)

    rprint(convert_to_openai_tool(get_weather2))
    messages = [
        HumanMessage(content="明天上海是什么天气？")
    ]
    response = _chat_deepseek().bind_tools([get_weather2]).invoke(messages)
    rprint(response)


if __name__ == "__main__":
    main()
