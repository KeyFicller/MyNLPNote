#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangSmith / LangChain 进阶：DeepSeek 直连（ChatDeepSeek）
"""

import os

from langchain_deepseek import ChatDeepSeek

MODEL = "deepseek-v4-pro"


def _api_base() -> str | None:
    return os.getenv("DEEPSEEK_BASE_URL")


def _api_key() -> str | None:
    return os.getenv("DEEPSEEK_API_KEY")

def _langsmith_config() -> dict:
    return {
        "project": os.getenv("LANGSMITH_PROJECT"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT"),
        "tracing": os.getenv("LANGSMITH_TRACING"),
    }


def _chat_deepseek() -> ChatDeepSeek:
    kwargs: dict = {"model": MODEL, "api_key": _api_key()}
    if base := _api_base():
        kwargs["api_base"] = base


    return ChatDeepSeek(**kwargs)


def demo_api_specifications() -> None:
    print("\n" + "=" * 60)
    print("API Specifications — ChatDeepSeek")
    print("=" * 60)
    llm = _chat_deepseek()
    config = {
        "run_name": "Hello LangSmith",
        "tags": ["langchain", "deepseek"],
    }
    response = llm.invoke("Introduce yourself with single sentence.", config=config)
    print(response.content)


def main() -> None:
    if not _api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    config = _langsmith_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    demo_api_specifications()


if __name__ == "__main__":
    main()
