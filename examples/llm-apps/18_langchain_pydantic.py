#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain Pydantic 模板
"""

from typing import Literal, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel,Field, ValidationError
from rich import print as rprint
from enum import Enum

from deepseek_client import api_key, chat_deepseek

class PydanticTest(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    work: str = Field(description="The work of the person")

def pydantic_test():
    llm =chat_deepseek().with_structured_output(PydanticTest)
    messages = [
        HumanMessage(content="张三是一名30岁的程序员")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

class PydanticTest2(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(default=18, description="The age of the person")
    work: str = Field(description="The work of the person")
    city: Optional[str] = Field(description="The city of the person")

def pydantic_test2():
    llm =chat_deepseek().with_structured_output(PydanticTest2)
    messages = [
        HumanMessage(content="张三是一名的程序员")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

    messages = [
        HumanMessage(content="李四是一名销售，在杭州工作")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

class Gender(str, Enum):
    MALE = "男",
    FEMALE = "女",
    UNKNOWN = "未知"

class PydanticTest3(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    gender: Gender = Field(default=Gender.UNKNOWN, description="The gender of the person")
    city: Literal["北京", "上海", "广州", "深圳", "杭州", "成都", "重庆", "武汉", "西安", "南京", "天津", "青岛", "济南", "郑州", "长沙", "石家庄", "太原", "沈阳", "大连", "长春", "哈尔滨", "厦门", "福州", "南昌", "南宁", "海口", "三亚", "昆明", "贵阳", "拉萨", "西安", "兰州", "西宁", "银川", "乌鲁木齐", "香港", "澳门", "台湾"] = Field(default="北京", description="The city of the person")

def pydantic_test3():
    llm =chat_deepseek().with_structured_output(PydanticTest3)
    messages = [
        HumanMessage(content="张三是一名30岁的女程序员")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

    llm =chat_deepseek().with_structured_output(PydanticTest3)
    messages = [
        HumanMessage(content="李四是一名30岁的程序员,在沿海地区经济最发达的地区工作")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

class PydanticTest4(BaseModel):
    elements : list[str] = Field(description="The elements of the list")

def pydantic_test4():
    llm =chat_deepseek().with_structured_output(PydanticTest4)
    messages = [
        HumanMessage(content="请列举出地球上最常见的五种化学元素")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

class Address(BaseModel):
    '''
    地址信息
    '''
    city : str = Field(description="The city of the address")
    district : str = Field(description="The district of the address")

class Company(BaseModel):
    '''
    公司信息
    '''
    name : str = Field(description="The name of the company")
    address : Address = Field(description="The address of the company")

def pydantic_test5():
    llm =chat_deepseek().with_structured_output(Company)
    messages = [
        HumanMessage(content="小米科技有限公司总部的地址是什么")
    ]
    response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
    rprint(response)

class User(BaseModel):
    '''
    用户信息
    '''
    name : str = Field(description="The name of the user", min_length=2, max_length=10)
    age : int = Field(description="The age of the user", ge=12, le=27)

def pydantic_test6():
    try:
        user = User(name="雷军1234", age=30)
    except ValidationError as e:
        rprint(e)

    llm =chat_deepseek().with_structured_output(User)
    messages = [
        HumanMessage(content="介绍雷军")
    ]
    try:
        response = llm.invoke(messages, extra_body={"thinking": {"type": "disabled"}})
        rprint(response)
    except ValidationError as e:
        rprint(e)
        print("用户信息验证失败")

def main() -> None:
    #pydantic_test()
    #pydantic_test2()
    #pydantic_test3()
    #pydantic_test4()
    #pydantic_test5()
    pydantic_test6()

if __name__ == "__main__":
    main()
