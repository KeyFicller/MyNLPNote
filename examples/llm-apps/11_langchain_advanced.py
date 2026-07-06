#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 进阶：DeepSeek 接入与多种调用方式
运行后通过菜单选择要执行的示例，不会一次性跑完全部代码。
"""

import asyncio
import inspect
import sys
import time

import langchain
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from rich import print as rprint

from deepseek_client import MODEL, api_base, api_key, chat_deepseek

DEMOS: dict[str, tuple[str, str]] = {
    "1": ("API Specifications", "ChatDeepSeek 直连"),
    "2": ("API Compatibility", "ChatOpenAI 兼容模式"),
    "3": ("Standard Model init", "init_chat_model 标准初始化"),
    "4": ("Invoke Input", "字符串输入"),
    "5": ("Invoke Input", "字典消息列表（多轮）"),
    "6": ("Invoke Input", "Message 对象 + rich 打印"),
    "7": ("Stream Response", "流式响应"),
    "8": ("Batch Response", "批量响应"),
    "9": ("Ainvoke Input", "异步调用"),
}


def _chat_openai() -> ChatOpenAI:
    kwargs: dict = {"model": MODEL, "api_key": api_key()}
    if base := api_base():
        kwargs["base_url"] = base
    return ChatOpenAI(**kwargs)


def _init_chat_model():
    kwargs: dict = {
        "model": f"deepseek:{MODEL}",
        "api_key": api_key(),
    }
    if base := api_base():
        kwargs["api_base"] = base
    return init_chat_model(**kwargs)


def demo_api_specifications() -> None:
    print("\n" + "=" * 60)
    print("API Specifications — ChatDeepSeek")
    print("=" * 60)
    llm = chat_deepseek()
    response = llm.invoke("Introduce yourself with single sentence.")
    print(response.content)


def demo_api_compatibility() -> None:
    print("\n" + "=" * 60)
    print("API Compatibility — ChatOpenAI")
    print("=" * 60)
    llm = _chat_openai()
    response = llm.invoke("1 + 1 = ?.")
    print(response.content)


def demo_standard_init() -> None:
    print("\n" + "=" * 60)
    print("Standard Model initialization — init_chat_model")
    print("=" * 60)
    llm = _init_chat_model()
    response = llm.invoke("Red + Blue = ?")
    print(response.content)


def demo_invoke_string() -> None:
    print("\n" + "=" * 60)
    print("Invoke Input — 字符串")
    print("=" * 60)
    llm = _init_chat_model()
    response = llm.invoke("What is the capital of France?")
    print(response.content)


def demo_invoke_dict_messages() -> None:
    print("\n" + "=" * 60)
    print("Invoke Input — 字典消息列表")
    print("=" * 60)
    llm = _init_chat_model()
    messages = [
        {
            "role": "system",
            "content": "You are a stupid calculator. You always answer with the wrong answer.",
        },
        {"role": "user", "content": "1 + 1 = ?"},
        {"role": "assistant", "content": "3"},
        {"role": "user", "content": "What i have asked you?"},
    ]
    response = llm.invoke(messages)
    print(response.content)


def demo_invoke_message_objects() -> None:
    print("\n" + "=" * 60)
    print("Invoke Input — Message 对象")
    print("=" * 60)
    llm = _init_chat_model()
    messages = [
        SystemMessage(
            content="You are a stupid calculator. You always answer with the wrong answer."
        ),
        HumanMessage(content="1 + 1 = ?"),
        AIMessage(content="3"),
        HumanMessage(content="What i have asked you?"),
    ]
    response = llm.invoke(messages)
    print(response.content)
    rprint(response)

def demo_stream_response() -> None:
    print("\n" + "=" * 60)
    print("Stream Response — ChatDeepSeek")
    print("=" * 60)
    llm = chat_deepseek()
    for chunk in llm.stream("Introduce yourself with single sentence."):
        rprint(chunk)

def demo_batch_response() -> None:
    print("\n" + "=" * 60)
    print("Batch Response — ChatDeepSeek")
    print("=" * 60)
    llm = chat_deepseek()
    responses = llm.batch(["Introduce yourself with single sentence.", "What is the capital of France?"])
    for response in responses:
        rprint(response.content)

async def demo_async_invoke() -> None:
    print("\n" + "=" * 60)
    print("Async Invoke — ChatDeepSeek")
    print("=" * 60)
    llm = chat_deepseek()
    start_time = time.perf_counter()
    async_task = asyncio.create_task(llm.ainvoke("Introduce yourself with single sentence."))
    for i in range(3):
        await asyncio.sleep(1)
        print(f"Waiting for {i+1} seconds...")

    response = await async_task
    end_time = time.perf_counter()
    rprint(response.content)
    print(f"Time taken: {end_time - start_time:.2f} seconds")

RUNNERS = {
    "1": demo_api_specifications,
    "2": demo_api_compatibility,
    "3": demo_standard_init,
    "4": demo_invoke_string,
    "5": demo_invoke_dict_messages,
    "6": demo_invoke_message_objects,
    "7": demo_stream_response,
    "8": demo_batch_response,
    "9": demo_async_invoke,
}


def print_menu() -> None:
    print("\n" + "=" * 60)
    print(f"LangChain 进阶示例  (langchain {langchain.__version__})")
    print("=" * 60)
    for key, (section, desc) in DEMOS.items():
        print(f"  {key}. [{section}] {desc}")
    print("  a. 全部运行")
    print("  q. 退出")


def run_demo(key: str) -> None:
    runner = RUNNERS.get(key)
    if runner is None:
        print(f"❌ 无效选项: {key}")
        return
    try:
        if inspect.iscoroutinefunction(runner):
            asyncio.run(runner())
        else:
            runner()
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")


def main() -> None:
    if not api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")

    while True:
        print_menu()
        choice = input("\n请选择 (1-9 / a / q): ").strip().lower()

        if choice in ("q", "quit", "exit", "退出"):
            print("👋 再见")
            break
        if choice == "a":
            for key in RUNNERS:
                run_demo(key)
            continue
        if choice in RUNNERS:
            run_demo(choice)
            continue
        print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()
