#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LangChain ChatDeepSeek 公共配置与工厂函数。"""

import os

from langchain_deepseek import ChatDeepSeek

MODEL = "deepseek-v4-pro"


def api_base() -> str | None:
    return os.getenv("DEEPSEEK_BASE_URL")


def api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")


def chat_deepseek() -> ChatDeepSeek:
    kwargs: dict = {"model": MODEL, "api_key": api_key()}
    if base := api_base():
        kwargs["api_base"] = base
    return ChatDeepSeek(**kwargs)
