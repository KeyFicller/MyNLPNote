#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消息历史记录
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

def _maintain_messages(messages: list[dict], memory_round = 2) -> None:
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    conversation_messages = [msg for msg in messages if msg["role"] in ["user", "assistant"]]

    return system_messages + conversation_messages[-memory_round:]


EXIT_WORD = "quit"

def main() -> None:
    if not _api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": "你是一个杠精，喜欢和用户抬杠。"
        }
    ]

    print(f"退出对话请输入：{EXIT_WORD}")

    i = 1
    while True:
        print("")
        print(f"=== 第 {i} 轮对话 ===")

        user_input = input("请输入：")
        if user_input.lower() == EXIT_WORD:
            print("退出对话")
            break

        messages.append({
            "role": "user",
            "content": user_input
        })

        print("AI 回复", end="", flush=True)
        memory_messages = _maintain_messages(messages, memory_round=3)
        reply_content = ""

        for chunk in _chat_deepseek().stream(memory_messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                reply_content += chunk.content

        messages.append({
            "role": "assistant",
            "content": reply_content
        })

        #response = _chat_deepseek().invoke(messages)
        #print(response.content)

        #messages.append({
        #    "role": "assistant",
        #    "content": response.content
        #})

        i += 1

if __name__ == "__main__":
    main()
