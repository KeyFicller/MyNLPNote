#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享 API 客户端 — 被 toy 下各游戏复用。
"""

import os
from typing import Optional
import openai

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"


def create_client() -> Optional[openai.OpenAI]:
    """
    创建并返回 OpenAI 兼容客户端实例（DeepSeek）。

    Returns:
        openai.OpenAI 实例，如果 API Key 未配置则返回 None
    """
    if not API_KEY:
        print("⚠️ 警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("   请在 .vscode/settings.json 或环境变量中配置")
        return None

    try:
        return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    except Exception as e:
        print(f"❌ 创建客户端失败: {e}")
        return None
