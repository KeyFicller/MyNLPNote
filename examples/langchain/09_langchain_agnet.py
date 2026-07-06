#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第40课：LangChain Agent
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

def create_agent_test():

    agent = create_agent(
        model = "deepseek:deepseek-v4-pro"
    )

    print(type(agent))

    display(Image(agent.get_graph().draw_mermaid_png()))

    agent = create_agent(
        model = init_chat_model(model = "deepseek:deepseek-v4-pro")
    )

    print(type(agent))

    display(Image(agent.get_graph().draw_mermaid_png()))

    agent = create_agent(
        model = chat_deepseek()
    )

    print(type(agent))

    display(Image(agent.get_graph().draw_mermaid_png()))

def invoke_agent_test():
    agent = create_agent(
        model = "deepseek:deepseek-v4-pro"
    )

    messages = [
        HumanMessage(content="你好")
    ]

    response = agent.invoke({"messages" : messages})

    rprint(response)
    rprint(messages)

@tool(parse_docstring=True)
def say_yes() -> str:
    '''回答yes'''
    print("tool invoked yes.")
    return "yes"

@tool(parse_docstring=True)
def say_no() -> str:
    '''回答no'''
    print("tool invoked no.")
    return "no"

def static_tools_agent_test():
    agent = create_agent(
        model = "deepseek:deepseek-v4-pro",
        tools = [say_yes, say_no]
    )

    messages = [
        HumanMessage(content="北京是中国的首都吗？")
    ]

    response = agent.invoke({"messages" : messages})

    rprint(response)
    rprint(response.get("messages")[-1].content)

    print("-" * 100)

    agent = create_agent(
        model="deepseek:deepseek-v4-pro",
        tools=[DuckDuckGoSearchRun()],
        system_prompt="你是助手。遇到天气、新闻等实时信息时，请使用搜索工具查询后再回答。",
    )

    messages = [
        HumanMessage(content="今天武汉的天气怎么样？")
    ]

    response = agent.invoke({"messages": messages})

    rprint(response)
    rprint(response.get("messages")[-1].content)

def main() -> None:
    #create_agent_test()
    #invoke_agent_test()
    static_tools_agent_test()

if __name__ == "__main__":
    main()
