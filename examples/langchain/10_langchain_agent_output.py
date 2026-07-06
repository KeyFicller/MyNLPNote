from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import AutoStrategy, ProviderStrategy, ToolStrategy
from pydantic import BaseModel, Field, field_validator
from deepseek_client import chat_deepseek
from langchain.messages import HumanMessage
from rich import print as rprint
from langchain.chat_models import init_chat_model

load_dotenv(override=True)

_PLACEHOLDER_NAMES = {"不详", "未提供", "未知", "无", "N/A", "n/a"}


class ContractInfo(BaseModel):
    """用户的联系方式"""
    name: str = Field(description="用户的姓名", min_length=1)
    phone: str = Field(description="用户的11位手机号码", min_length=11, max_length=11)
    email: str = Field(
        description="用户的电子邮箱"
    )
    '''
    @field_validator("name")
    @classmethod
    def reject_placeholder_name(cls, v: str) -> str:
        if v.strip() in _PLACEHOLDER_NAMES:
            raise ValueError("姓名不能为占位值")
        return v.strip()

    @field_validator("phone")
    @classmethod
    def reject_fake_phone(cls, v: str) -> str:
        if set(v) == {"0"}:
            raise ValueError("手机号不能为虚构的全零号码")
        return v

    @field_validator("email")
    @classmethod
    def reject_placeholder_email(cls, v: str) -> str:
        lower = v.lower()
        if "placeholder" in lower or lower.endswith("@example.com"):
            raise ValueError("邮箱不能为占位值")
        return v
    '''

def main() -> None:
    agent = create_agent(
        model = init_chat_model("deepseek:deepseek-v4-pro", extra_body={"thinking": {"type": "disabled"}}),
        # Deepseek doesn't support ProviderStrategy, so we use ToolStrategy instead.
        response_format = ToolStrategy(
            ContractInfo,
            tool_message_content="成功提取内容")
        # response_format = ProviderStrategy(ContractInfo)
        # response_format = AutoStrategy(ContractInfo)
    )

    messages = [
        HumanMessage(content="从这段话中提取信息，小明的邮箱地址为：xiaoming@163.com，手机号：12345678912")
    ]

    response = agent.invoke({"messages" : messages})

    for msg in response.get("messages"):
        msg.pretty_print()

    rprint(response.get("messages")[-1].content)
    rprint(response.get("structured_response"))

    agent_with_error_handling = create_agent(
        model=init_chat_model(
            "deepseek:deepseek-v4-pro",
            extra_body={"thinking": {"type": "disabled"}},
        ),
        response_format=ToolStrategy(
            ContractInfo,
            tool_message_content="成功提取内容",
            handle_errors="校验失败：姓名、11位手机号、邮箱必须严格来自原文，禁止编造或使用占位值。",
        ),
        system_prompt=(
            "你是信息抽取助手。只抽取文本中明确出现的姓名、11位手机号和邮箱，捏造行为将视为严重违法"
        ),
    )

    messages = [
        HumanMessage(
            content="从这段话中提取信息：邮箱地址缺失，手机号只有三位：123"
        )
    ]

    config = {
        "recursion_limit": 5
    }
    response = agent_with_error_handling.invoke({"messages" : messages}, config=config)

    for msg in response.get("messages"):
        msg.pretty_print()

    rprint(response.get("messages")[-1].content)
    rprint(response.get("structured_response"))


if __name__ == "__main__":
    main()
