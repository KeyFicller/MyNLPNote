#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 进阶：Prompt 模板
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, prompt

from deepseek_client import api_key, chat_deepseek

def _test_chat_prompt_format():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一个{role}，喜欢和用户{behavior}。"),
            ("user", "{input}"),
        ]
    )

    prompt = prompt_template.invoke({
        "role": "杠精",
        "behavior": "抬杠",
        "input": "你好，你是谁？",
    })
    #print(prompt)
    print(type(prompt))
    response = chat_deepseek().invoke(prompt)
    print(response.content)

    prompt = prompt_template.format(
        role="马屁精",
        behavior="溜须拍马",
        input="你好，你是谁？",
    )
    #print(prompt)
    response = chat_deepseek().invoke(prompt)
    print(response.content)

    prompt = prompt_template.format_messages(
        role="社恐",
        behavior="社恐",
        input="你好，你是谁？",
    )
    #print(prompt)
    response = chat_deepseek().invoke(prompt)
    print(response.content)

def _test_chat_prompt_initialization():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一个杠精，喜欢和用户抬杠。"),
        ("user", "你好，{user_input}")
    ])

    prompt = prompt_template.invoke({
        "user_input": "你是最牛逼的ai模型吗？",
    })
    #print(prompt)
    response = chat_deepseek().invoke(prompt)
    print(response.content)

    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("你是一个杠精，喜欢和用户抬杠。"),
        HumanMessagePromptTemplate.from_template("你好，{user_input}"),
    ])

    prompt = prompt_template.invoke({
        "user_input": "你是最牛逼的ai模型吗？",
    })
    #print(prompt)
    response = chat_deepseek().invoke(prompt)
    print(response.content)

    prompt_template = ChatPromptTemplate.from_messages([
        ChatPromptTemplate.from_messages(("system", "你是一个杠精，喜欢和用户抬杠。")),
        ChatPromptTemplate.from_messages(("user", "你好，你是谁？")),
    ])
    #print(prompt)
    prompt = prompt_template.invoke({})
    response = chat_deepseek().invoke(prompt)
    print(response.content)

def _test_chat_prompt_partial():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是{department} 部门的{role}。"),
        ("user", "{user_input}"),
    ])
    
    IT_department = prompt_template.partial(department="IT", role="工程师")
    SALING_department = prompt_template.partial(department="销售", role="销售员")

    prompt = IT_department.invoke({
        "user_input": "我的鼠标为什么坏了？",
    })
    response = chat_deepseek().invoke(prompt)
    print(response.content)
    print("-" * 60)
    prompt = SALING_department.invoke({
        "user_input": "我的鼠标为什么坏了？",
    })
    response = chat_deepseek().invoke(prompt)
    print(response.content)
    print("-" * 60)

def _test_chat_prompt_placeholder():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一名教授。"),
        ("placeholder", "{conversation}"),
    ])

    prompt = prompt_template.invoke({
        # Here you can use tuple
        "conversation": [
            ("user", "你好，你是谁？"),
            ("assistant", "我是主攻土木工程的教授。"),
            ("user", "我没听清楚，你主攻哪个方向来着？"),
        ]
    })
    response = chat_deepseek().invoke(prompt)
    print(response.content)

    print("-" * 60)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一名教授。"),
        MessagesPlaceholder(variable_name="conversation")
    ])

    prompt = prompt_template.invoke({
        # Here you can use tuple
        "conversation": [
            HumanMessage(content="你好，你是谁？"),
            AIMessage(content="我是主攻土木工程的教授。"),
            HumanMessage(content="我没听清楚，你主攻哪个方向来着？"),
        ]
    })
    response = chat_deepseek().invoke(prompt)
    print(response.content)


    print("-" * 60)
    prompt_template1 = ChatPromptTemplate.from_messages([
        ("system", "你是一名教授。"),
    ])
    prompt_template2 = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="conversation"),
    ])
    prompt_template = prompt_template1 + prompt_template2
    prompt = prompt_template.invoke({
        # Here you can use tuple
        "conversation": [
            HumanMessage(content="你好，你是谁？"),
            AIMessage(content="我是主攻土木工程的教授。"),
            HumanMessage(content="我没听清楚，你主攻哪个方向来着？"),
        ]
    })
    response = chat_deepseek().invoke(prompt)
    print(response.content)


def main() -> None:
    if not api_key():
        print("⚠️  未设置 DEEPSEEK_API_KEY，请在 .env 或环境变量中配置")
    print("=" * 60)

    #_test_chat_prompt_format()
    #_test_chat_prompt_initialization()
    #_test_chat_prompt_partial()
    _test_chat_prompt_placeholder()


if __name__ == "__main__":
    main()
