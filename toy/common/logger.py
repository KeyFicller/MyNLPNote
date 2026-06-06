#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享日志系统 — 被 toy 下各游戏复用。

提供：
- setup_game_logging(game_name, log_dir) — 开启双写（控制台 + 文件）
- game_print(*args, **kwargs)        — 替代内置 print
- close_game_logging()               — 关闭日志文件
"""

import os
import re
import atexit
import builtins
from datetime import datetime
from typing import TextIO

_LOG_FILE: TextIO | None = None
_LOG_PATH: str | None = None
_ORIGINAL_PRINT = builtins.print

# 公开别名 — 供游戏在 logging 尚未开启或需要绕过日志时使用
original_print = _ORIGINAL_PRINT

ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """去除 ANSI 颜色代码"""
    return ANSI_ESCAPE_RE.sub("", text)


def setup_game_logging(game_name: str = "game", log_dir: str | None = None) -> str:
    """
    复写 builtins.print，将输出写入带时间戳的日志文件（同时保留控制台输出）。

    Args:
        game_name: 游戏名，用于日志文件名前缀（如 "undercover" / "liars_bar"）
        log_dir:   日志目录，默认为当前文件所在目录的 ../logs

    Returns:
        日志文件的完整路径
    """
    global _LOG_FILE, _LOG_PATH

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(log_dir, f"{game_name}_{ts}.log")
    _LOG_FILE = open(_LOG_PATH, "w", encoding="utf-8")
    _LOG_FILE.write(f"=== {game_name} {datetime.now().isoformat(timespec='seconds')} ===\n")
    _LOG_FILE.flush()

    builtins.print = game_print
    atexit.register(close_game_logging)
    return _LOG_PATH


def game_print(*args, **kwargs):
    """
    替代内置 print：写入日志；默认仍输出到控制台。

    额外关键字参数：
        log_plain: 可选，写入日志的纯文本（无 ANSI）；控制台仍用 args 拼接结果
    """
    log_plain = kwargs.pop("log_plain", None)
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    text = sep.join(str(a) for a in args) + end

    if _LOG_FILE and not _LOG_FILE.closed:
        if log_plain is not None:
            log_text = log_plain if str(log_plain).endswith(end) else str(log_plain) + end
        else:
            log_text = _strip_ansi(text)
        _LOG_FILE.write(log_text)
        _LOG_FILE.flush()

    target = kwargs.get("file")
    if target is not None:
        _ORIGINAL_PRINT(*args, **kwargs)
    else:
        _ORIGINAL_PRINT(*args, **{k: v for k, v in kwargs.items() if k != "file"})


def close_game_logging():
    """关闭日志文件"""
    global _LOG_FILE
    if _LOG_FILE and not _LOG_FILE.closed:
        _LOG_FILE.write(f"\n=== END {datetime.now().isoformat(timespec='seconds')} ===\n")
        _LOG_FILE.close()
        _LOG_FILE = None
