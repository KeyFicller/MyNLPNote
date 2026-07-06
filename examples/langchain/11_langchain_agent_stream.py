#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第43课：LangChain Agent 流式输出
======================================

运行方式：
    python examples/langchain/09_langchain_agnet.py

相关笔记：
    notes/phase-langchain/09_LangChain_Agent.md

"""

from langchain.agents import create_agent
from dotenv import load_dotenv
from IPython.display import Image, display
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from deepseek_client import chat_deepseek
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from rich import print as rprint

load_dotenv(override=True)

@tool()
def get_current_weather(city: str) -> str:
    '''获取当前天气'''
    return f"当前天气为：{city} 晴朗"

@tool()
def get_current_time() -> str:
    '''获取当前时间'''
    return f"当前时间：2026-07-06 10:00:00"

@tool()
def get_current_location() -> str:
    '''获取当前位置'''
    return f"当前位置：武汉"

def invoke_agent_test(stream_mode: str, print_chunk : bool = False):
    agent = create_agent(
        model = "deepseek:deepseek-v4-pro",
        tools = [get_current_weather, get_current_time, get_current_location]
    )

    messages = [
        HumanMessage(content="当前的位置、时间、天气是什么？")
    ]

    for chunk in agent.stream({"messages" : messages}, stream_mode=stream_mode):
        if (print_chunk):
            rprint(chunk)
        # if ("messages" in chunk and chunk["messages"] != None):
        #     print(f"message count streamed: {len(chunk['messages'])}")
        #     rprint(chunk["messages"][-1].pretty_print())
        # elif ("content" in chunk and chunk["content"] != None):
        #     print(f"content streamed: {chunk['content']}")
        # else:
        #     print("No messages streamed")
        print("-" * 100)

def main():
    invoke_agent_test("values", print_chunk=True)
    #invoke_agent_test("updates", print_chunk=True)
    #invoke_agent_test("messages", print_chunk=True)
    #invoke_agent_test("tasks", print_chunk=True)
    #invoke_agent_test("debug", print_chunk=True)
    #invoke_agent_test("checkpoints", print_chunk=True)

if __name__ == "__main__":
    main()