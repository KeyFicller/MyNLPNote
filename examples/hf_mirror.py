#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuggingFace 镜像与缓存配置（中国大陆用户）

用法（必须在 import transformers / sentence_transformers 之前调用）：
    from hf_mirror import setup_hf_mirror
    setup_hf_mirror()
"""

import os
from typing import Optional

# 项目根目录（examples/ 的上一级）
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_MIRROR = "https://hf-mirror.com"


def setup_hf_mirror(
    mirror: str = DEFAULT_MIRROR,
    cache_dir: Optional[str] = None,
    download_timeout: str = "600",
) -> str:
    """
    配置 HuggingFace 国内镜像与本地缓存目录。

    Returns:
        HF_HOME 绝对路径
    """
    hf_home = cache_dir or os.path.join(_PROJECT_ROOT, ".cache", "huggingface")
    hf_home = os.path.abspath(hf_home)
    hub_cache = os.path.join(hf_home, "hub")

    os.environ["HF_ENDPOINT"] = mirror
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = download_timeout
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    os.makedirs(hub_cache, exist_ok=True)
    return hf_home
