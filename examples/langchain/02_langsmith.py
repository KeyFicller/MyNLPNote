#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第33课：LangSmith 追踪与可观测性
================================

运行方式：
    python examples/langchain/02_langsmith.py

相关笔记：
    notes/phase-langchain/02_LangSmith追踪与可观测性.md
"""

import os

from deepseek_client import api_key, chat_deepseek

def _langsmith_config() -> dict:
    return {
        "project": os.getenv("LANGSMITH_PROJECT"),
        "api_key": os.getenv("LANGSMITH_API_KEY"),
        "endpoint": os.getenv("LANGSMITH_ENDPOINT"),
        "tracing": os.getenv("LANGSMITH_TRACING"),
    }


def demo_api_specifications() -> None:
    print("\n" + "=" * 60)
    print("API Specifications — ChatDeepSeek")
    print("=" * 60)
    llm = chat_deepseek()
    config = {
        "run_name": "Hello LangSmith",
        "tags": ["langchain", "deepseek"],
    }
    response = llm.invoke("Introduce yourself with single sentence.", config=config)
    print(response.content)


def main() -> None:
    if not api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    config = _langsmith_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    print("=" * 60)
    demo_api_specifications()


if __name__ == "__main__":
    main()
