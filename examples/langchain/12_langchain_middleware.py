#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第43课：LangChain Agent Middleware
======================================

参考课件：尚硅谷-08-中间件.pdf

运行方式：
    python examples/langchain/12_langchain_middleware.py

相关笔记：
    notes/phase-langchain/12_LangChain_Middleware.md

"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    after_agent,
    after_model,
    before_agent,
    before_model,
)
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command
from rich import print as rprint

from deepseek_client import MODEL, api_base, api_key

load_dotenv(override=True)

DEEPSEEK_MODEL = f"deepseek:{MODEL}"
CUSTOM_PROFILE = {"max_input_tokens": 128_000}


def _summary_model():
    kwargs: dict = {
        "model": DEEPSEEK_MODEL,
        "profile": CUSTOM_PROFILE,
        "api_key": api_key(),
    }
    if base := api_base():
        kwargs["base_url"] = base
    return init_chat_model(**kwargs)


def _pretty_messages(response: dict) -> None:
    for msg in response.get("messages", []):
        msg.pretty_print()


# ---------------------------------------------------------------------------
# §2.1 SummarizationMiddleware — 压缩上下文
# ---------------------------------------------------------------------------

def summarization_middleware_test() -> None:
    """课件 2.1.2：trigger / keep 触发摘要，摘要 HumanMessage 插入头部。"""
    messages = [
        SystemMessage("你是个非常友好的AI助手"),
        HumanMessage("你好啊，我是老王，你是谁？"),
        AIMessage("你好老王，我是小王"),
        HumanMessage("好的小王，很高兴认识你"),
        AIMessage("你高兴得太早了"),
        HumanMessage("呵呵，你什么意思"),
    ]

    agent = create_agent(
        model=DEEPSEEK_MODEL,
        middleware=[
            SummarizationMiddleware(
                model=_summary_model(),
                trigger=[("tokens", 100), ("messages", 6), ("fraction", 0.001)],
                keep=("messages", 2),
            )
        ],
    )

    response = agent.invoke({"messages": messages})
    print("==== §2.1 SummarizationMiddleware ====")
    _pretty_messages(response)
    print("-" * 100)


def summarization_custom_prompt_test() -> None:
    """课件 2.1.3：自定义 summary_prompt。"""
    messages = [
        SystemMessage("你是个非常友好的AI助手"),
        HumanMessage("你好啊，我是老王，你是谁？"),
        AIMessage("你好老王，我是小王"),
        HumanMessage("好的小王，很高兴认识你"),
        AIMessage("你高兴得太早了"),
        HumanMessage("呵呵，你什么意思，你是谁？"),
    ]

    agent = create_agent(
        model=DEEPSEEK_MODEL,
        middleware=[
            SummarizationMiddleware(
                model=_summary_model(),
                trigger=[("tokens", 100), ("messages", 6), ("fraction", 0.0001)],
                keep=("messages", 2),
                summary_prompt="对历史消息摘要，消息列表如下\n{messages}",
            )
        ],
    )

    response = agent.invoke({"messages": messages})
    print("==== §2.1.3 自定义 summary_prompt ====")
    _pretty_messages(response)
    print("-" * 100)


# ---------------------------------------------------------------------------
# §2.2 HumanInTheLoopMiddleware — 人工审批
# ---------------------------------------------------------------------------

@tool
def get_weather(city: str, is_forcast: bool = False) -> str:
    """查询指定城市天气。"""
    res = f"{city}今天天气不错"
    if is_forcast:
        res += "\n明天下雨"
    return res


@tool
def get_news() -> str:
    """查询当日新闻。"""
    return "中方三艘油轮通过霍尔木兹海峡"


@tool
def read_email_tool(email_id: str) -> str:
    """通过邮件 ID 读取内容。"""
    return f"邮件ID：{email_id}\n是空的"


@tool
def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """发送邮件。"""
    print(">>> 真的执行发送邮件工具了")
    return f"发送给 {recipient} 的邮件标题是：{subject}，内容：{body}"


def human_in_the_loop_test() -> None:
    """课件 2.2：工具调用前中断，Command(resume=...) 审批后继续。"""
    agent = create_agent(
        model=DEEPSEEK_MODEL,
        tools=[get_weather, get_news, read_email_tool, send_email_tool],
        checkpointer=InMemorySaver(),
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "get_weather": True,
                    "get_news": True,
                    "read_email_tool": False,
                    "send_email_tool": {
                        "allowed_decisions": ["approve", "reject"],
                        "description": "发送邮件中断啦",
                    },
                },
                description_prefix="中断啦",
            )
        ],
    )

    config = {"configurable": {"thread_id": "middleware-hitl-1"}}
    user_content = (
        "请帮我查询今天北京的天气"
        "查询今日新闻"
        "查看ID为 'sk2131421' 的邮件内容，"
        "向15641685664@qq.com发送邮件，标题是'哈哈哈'，内容是：'你好啊'"
        "同时做这四件事"
    )

    response = agent.invoke({"messages": [HumanMessage(content=user_content)]}, config=config)
    print("==== §2.2 第一次 invoke（中断） ====")
    _pretty_messages(response)

    interrupts = response.get("__interrupt__", [])
    print("========== interrupts ==========")
    rprint(interrupts)

    if not interrupts:
        print("未触发中断，跳过 resume 演示")
        print("-" * 100)
        return

    action_requests = interrupts[0].value["action_requests"]
    decisions: dict[str, list] = {"decisions": []}
    for action_request in action_requests:
        if action_request["name"] == "get_weather":
            decisions["decisions"].append(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "get_weather",
                        "args": {"city": "中国北京市", "is_forcast": True},
                    },
                }
            )
        elif action_request["name"] == "get_news":
            decisions["decisions"].append({"type": "approve"})
        elif action_request["name"] == "send_email_tool":
            decisions["decisions"].append({"type": "approve"})

    resumed = agent.invoke(Command(resume=decisions), config=config)
    print("==== §2.2 审批后继续执行 ====")
    _pretty_messages(resumed)
    print("-" * 100)


# ---------------------------------------------------------------------------
# §2.3 PIIMiddleware — 敏感信息保护
# ---------------------------------------------------------------------------

def pii_middleware_test() -> None:
    """课件 2.3.2：内置检测器 + 多种 strategy。"""
    agent = create_agent(
        model=DEEPSEEK_MODEL,
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
            PIIMiddleware("url", strategy="hash", apply_to_input=True),
            PIIMiddleware("mac_address", strategy="mask", apply_to_input=True),
            PIIMiddleware("ip", strategy="block", apply_to_input=True),
        ],
    )

    print("==== §2.3.2 PII 脱敏 invoke ====")
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    """帮我向 156168188@qq.com 发送一封邮件
同时查看银行卡号： 5105-1051-0510-5100 的余额
访问 https://localhost:12345
确认这是不是 MAC地址： 11-11-11-11-11-11"""
                )
            ]
        }
    )
    _pretty_messages(response)

    print("==== §2.3.2 PII block invoke ====")
    try:
        agent.invoke({"messages": [HumanMessage("看看这个 IP 能不能 ping 通：192.168.10.1")]})
    except Exception as e:
        print("=" * 30, "-> 抛异常 <-", "=" * 30)
        print(f"检测到IP，抛出异常：{e}")
    print("-" * 100)


def detect_phone_number(content: str) -> list[dict[str, int | str]]:
    """课件 2.3.3：自定义手机号检测器。"""
    return [
        {"text": m.group(0), "start": m.start(), "end": m.end()}
        for m in re.finditer(r"[0-9]{11}", content)
    ]


def pii_custom_detector_test() -> None:
    """课件 2.3.3：自定义 detector（正则 + 函数）。"""
    agent = create_agent(
        model=DEEPSEEK_MODEL,
        tools=[],
        middleware=[
            PIIMiddleware(
                "api_key",
                strategy="hash",
                apply_to_input=True,
                detector=r"sk-[a-zA-Z0-9]+",
            ),
            PIIMiddleware(
                "phone_number",
                strategy="mask",
                apply_to_input=True,
                detector=detect_phone_number,
            ),
        ],
    )

    print("==== §2.3.3 自定义 PII 检测器 ====")
    response = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    """这是不是有效的 API_KEY： sk-awef23AFEfaafaefa
帮我给这个号码打电话： 12345612345
访问 https://localhost:12345"""
                )
            ]
        }
    )
    _pretty_messages(response)
    print("-" * 100)


# ---------------------------------------------------------------------------
# §2.4 TodoListMiddleware — 任务规划
# ---------------------------------------------------------------------------

TODO_WORKSPACE = Path(__file__).resolve().parent / "todo_workspace"


@tool
def list_files(path: str = ".") -> str:
    """列出工作区指定目录下的文件和子目录。"""
    target = (TODO_WORKSPACE / path).resolve()
    workspace_root = TODO_WORKSPACE.resolve()
    if not str(target).startswith(str(workspace_root)):
        return "错误：只允许访问工作区内的目录。"
    if not target.exists():
        return f"错误：目录不存在: {path}"
    if not target.is_dir():
        return f"错误：不是目录: {path}"
    items = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    if not items:
        return f"目录为空: {path}"
    lines = []
    for item in items:
        rel = item.relative_to(workspace_root)
        kind = "[DIR]" if item.is_dir() else "[FILE]"
        lines.append(f"{kind} {rel.as_posix()}")
    return "\n".join(lines)


@tool
def read_file(path: str) -> str:
    """读取工作区中的文本文件内容。"""
    file_path = (TODO_WORKSPACE / path).resolve()
    if not str(file_path).startswith(str(TODO_WORKSPACE.resolve())):
        return "错误：只允许读取工作区内的文件。"
    if not file_path.exists():
        return f"错误：文件不存在: {path}"
    return file_path.read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """写入工作区中的文本文件。"""
    file_path = (TODO_WORKSPACE / path).resolve()
    if not str(file_path).startswith(str(TODO_WORKSPACE.resolve())):
        return "错误：只允许写入工作区内的文件。"
    file_path.write_text(content, encoding="utf-8")
    return f"已写入文件: {path}"


@tool
def run_tests() -> str:
    """在工作区运行 pytest -q，并返回输出。"""
    try:
        result = subprocess.run(
            ["pytest", "-q"],
            cwd=str(TODO_WORKSPACE),
            capture_output=True,
            text=True,
            timeout=20,
        )
        return (
            f"returncode={result.returncode}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )
    except Exception as e:
        return f"运行测试失败: {e}"


def todo_list_middleware_test() -> None:
    """课件 2.4：TodoListMiddleware 修复 my_add.py。"""
    TODO_WORKSPACE.mkdir(exist_ok=True)
    (TODO_WORKSPACE / "my_add.py").write_text(
        'def add(a: int, b: int) -> int:\n    """返回两个整数的和"""\n    return a - b\n',
        encoding="utf-8",
    )
    (TODO_WORKSPACE / "test_my_add.py").write_text(
        "from my_add import add\n\n"
        "def test_add():\n"
        '    """测试加法功能"""\n'
        "    assert add(2, 3) == 5\n"
        "    assert add(-1, 1) == 0\n"
        "    assert add(0, 0) == 0\n"
        "    assert add(10, -5) == 5\n",
        encoding="utf-8",
    )

    agent = create_agent(
        model=DEEPSEEK_MODEL,
        tools=[list_files, read_file, write_file, run_tests],
        middleware=[TodoListMiddleware()],
        system_prompt=(
            "你是一个代码修复助手。遇到多步骤任务时，先使用 write_todos 制定待办事项；"
            "然后读取文件、修复代码并运行测试。工作全部在工作区下进行。"
        ),
    )

    print("==== §2.4 TodoListMiddleware ====")
    print("正在执行 Agent 任务...")
    final_state = agent.invoke(
        {"messages": [HumanMessage(content="请测试并修复工作区下 my_add.py 文件中的代码")]}
    )
    rprint(final_state.get("todos", "（无 todos 字段）"))
    if final_state.get("messages"):
        print(final_state["messages"][-1].content)
    print("-" * 100)


# ---------------------------------------------------------------------------
# §3.1 ModelCallLimitMiddleware — 调用限额
# ---------------------------------------------------------------------------

def model_call_limit_test() -> None:
    """课件 3.1：thread_limit 限制整个会话的 model 调用次数。"""
    agent = create_agent(
        model=DEEPSEEK_MODEL,
        checkpointer=InMemorySaver(),
        tools=[],
        middleware=[ModelCallLimitMiddleware(thread_limit=2, exit_behavior="end")],
    )
    config = {"configurable": {"thread_id": "middleware-mcl-1"}}

    print("==== §3.1 ModelCallLimitMiddleware ====")
    for i in range(3):
        print(f"--- 第 {i + 1} 次 invoke ---")
        response = agent.invoke({"messages": [HumanMessage(content=f"第{i + 1}次问好")]}, config=config)
        _pretty_messages(response)
    print("-" * 100)


# ---------------------------------------------------------------------------
# §4 多个 Middleware 执行顺序（洋葱模型）
# ---------------------------------------------------------------------------

class Middleware1(AgentMiddleware):
    def before_model(self, state, runtime):
        print("[中间件1] before_model")
        return None

    def after_model(self, state, runtime):
        print("[中间件1] after_model")
        return None


class Middleware2(AgentMiddleware):
    def before_model(self, state, runtime):
        print("[中间件2] before_model")
        return None

    def after_model(self, state, runtime):
        print("[中间件2] after_model")
        return None


class Middleware3(AgentMiddleware):
    def before_model(self, state, runtime):
        print("[中间件3] before_model")
        return None

    def after_model(self, state, runtime):
        print("[中间件3] after_model")
        return None


def middleware_order_test() -> None:
    """课件 §4：before_model 正序 1→2→3，after_model 逆序 3→2→1。"""
    agent = create_agent(
        model=DEEPSEEK_MODEL,
        tools=[],
        middleware=[Middleware1(), Middleware2(), Middleware3()],
    )
    print("==== §4 中间件执行顺序 ====")
    print("执行一次调用，观察顺序：")
    agent.invoke({"messages": [{"role": "user", "content": "测试"}]})
    print("\n关键点：")
    print("  - before_model: 正序执行（1→2→3）")
    print("  - after_model:  逆序执行（3→2→1）")
    print("  - 类似洋葱模型：1→2→3→模型→3→2→1")
    print("-" * 100)


# ---------------------------------------------------------------------------
# §5 自定义 Middleware — 装饰器 / 类
# ---------------------------------------------------------------------------

@before_model
def before_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_model <- "
    return None


@after_model
def after_model_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_model <- "
    return None


@before_agent
def before_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> before_agent <- "
    return None


@after_agent
def after_agent_middleware(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    state["messages"][-1].content += " -> after_agent <- "
    return None


def custom_decorator_middleware_test() -> None:
    """课件 5.3.1：装饰器实现 Node-style hooks。"""
    agent = create_agent(
        model=DEEPSEEK_MODEL,
        middleware=[
            before_model_middleware,
            after_model_middleware,
            before_agent_middleware,
            after_agent_middleware,
        ],
    )
    print("==== §5.3.1 装饰器自定义 Middleware ====")
    response = agent.invoke({"messages": [HumanMessage("你好啊")]})
    _pretty_messages(response)
    print("-" * 100)


class MyMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> before_model <- "
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> after_model <- "
        return None

    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> before_agent <- "
        return None

    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        state["messages"][-1].content += " -> after_agent <- "
        return None


def custom_class_middleware_test() -> None:
    """课件 5.3.1：AgentMiddleware 子类实现。"""
    agent = create_agent(model=DEEPSEEK_MODEL, middleware=[MyMiddleware()])
    print("==== §5.3.1 类自定义 Middleware ====")
    response = agent.invoke({"messages": [HumanMessage("你好啊")]})
    _pretty_messages(response)
    print("-" * 100)


def main() -> None:
    # §2 常用内置中间件
    #summarization_middleware_test()
    #summarization_custom_prompt_test()
    human_in_the_loop_test()
    #pii_middleware_test()
    # pii_custom_detector_test()
    # todo_list_middleware_test()

    # §3 其它内置中间件
    # model_call_limit_test()

    # §4 组合与顺序
    # middleware_order_test()

    # §5 自定义中间件
    # custom_decorator_middleware_test()
    # custom_class_middleware_test()


if __name__ == "__main__":
    main()
