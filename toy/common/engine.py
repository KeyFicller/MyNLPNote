#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多 AI 文字对抗性游戏框架
========================

提供 BasePlayer 和 BaseGame 两个抽象基类，封装回合制 AI 对战游戏的通用模式：

- 玩家管理（存活/淘汰）
- 回合生命周期（初始化 → 准备 → 执行 → 胜负判定 → 结算）
- 历史记录
- 模板方法 run() — 子类只需填充各阶段逻辑

用法：
    class MyGame(BaseGame):
        def init_players(self): ...
        def setup_round(self) -> bool: ...
        def run_round(self) -> bool: ...
        def check_end_condition(self) -> bool: ...
        def get_winner(self) -> BasePlayer | None: ...
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

import openai

from common.client import create_client


# ── 数据结构 ────────────────────────────────────────────


@dataclass
class GameResult:
    """游戏结算结果"""
    winner: str = ""                # 胜者标识（玩家名 / 阵营名）
    rounds: int = 0                 # 总局数
    duration: float = 0.0           # 耗时（秒）
    details: Dict[str, Any] = field(default_factory=dict)


# ── 玩家基类 ────────────────────────────────────────────


class BasePlayer(ABC):
    """
    AI 玩家基类。

    子类需实现 make_decision() 定义该游戏的决策逻辑。
    通用属性 index / name / alive / client 由基类管理。
    """

    def __init__(
        self,
        index: int,
        name: str = "",
        client: Optional[openai.OpenAI] = None,
    ):
        self.index = index
        self.name = name or f"玩家{index + 1}"
        self.alive = True
        self.client = client or create_client()

    def __repr__(self) -> str:
        status = "存活" if self.alive else "淘汰"
        return f"<{self.name} #{self.index} [{status}]>"

    @abstractmethod
    def make_decision(self, game: "BaseGame", **kwargs) -> Dict[str, Any]:
        """
        根据当前游戏状态做出决策。

        Returns:
            决策 dict，至少包含 {"action": str}。具体字段由子游戏定义。
        """
        ...


# ── 游戏基类 ────────────────────────────────────────────


class BaseGame(ABC):
    """
    多 AI 对抗游戏基类 — 模板方法模式。

    子类必须实现 5 个抽象方法：
        init_players()         — 创建玩家列表
        setup_round()          — 每轮初始化（发牌/出题等）；返回 False 则终止
        run_round()            — 执行一轮游戏逻辑；返回 False 则终止
        check_end_condition()  — 判定胜负，设置 self.ended / self.result
        get_winner()           — 返回胜者（或 None）

    通用钩子（可选覆盖）：
        print_intro()   — 开场动画
        print_summary() — 终局总结
    """

    def __init__(self):
        self.players: List[BasePlayer] = []
        self.current_round: int = 0
        self.history: List[Dict[str, Any]] = []
        self.ended: bool = False
        self.start_time: Optional[datetime] = None

    # ── 抽象方法（子类必须实现） ──

    @abstractmethod
    def init_players(self) -> None:
        """创建玩家并填充 self.players"""
        ...

    @abstractmethod
    def setup_round(self) -> bool:
        """
        准备新一轮。
        通常包括：递增轮次、发牌/出题、重置回合状态。

        Returns:
            True 可继续；False 无法开局（如存活人数不足）
        """
        ...

    @abstractmethod
    def run_round(self) -> bool:
        """
        执行一轮完整的游戏逻辑。
        通常包括：玩家依次行动 → 结算 → 记录历史。

        Returns:
            True 本轮正常结束；False 整局游戏终止
        """
        ...

    @abstractmethod
    def check_end_condition(self) -> bool:
        """
        检查游戏是否满足终止条件。
        若终止，设置 self.ended = True 并填充 self.result。

        Returns:
            True 游戏已结束
        """
        ...

    @abstractmethod
    def get_winner(self) -> Optional[BasePlayer]:
        """返回胜者；平局或无胜者返回 None"""
        ...

    # ── 钩子（可选覆盖） ──

    def print_intro(self) -> None:
        """游戏开场信息"""
        print(f"\n{'='*60}")
        print(f"🎮 游戏开始！")
        print(f"{'='*60}")

    def print_summary(self) -> None:
        """游戏结束总结 — 子类可覆盖以输出游戏特有信息"""
        print(f"\n{'='*60}")
        print("📋 游戏总结")
        print(f"{'='*60}")
        winner = self.get_winner()
        print(f"  获胜方: {winner.name if winner else '平局'}")
        print(f"  总轮数: {self.current_round}")
        if self.start_time:
            duration = (datetime.now() - self.start_time).total_seconds()
            print(f"  游戏时长: {duration:.2f}秒")

    # ── 通用工具方法 ──

    def get_alive_players(self) -> List[BasePlayer]:
        """返回存活玩家列表"""
        return [p for p in self.players if p.alive]

    def get_alive_indices(self) -> List[int]:
        """返回存活玩家索引列表"""
        return [i for i, p in enumerate(self.players) if p.alive]

    def record_event(self, event_type: str, detail: str) -> None:
        """向历史记录追加一条事件"""
        self.history.append({
            "round": self.current_round,
            "event": event_type,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })

    # ── 模板方法 ──

    def start(self) -> bool:
        """
        初始化游戏。调用 init_players() 并记录开始时间。

        Returns:
            True 启动成功
        """
        self.start_time = datetime.now()
        self.print_intro()
        self.init_players()
        return True

    def run(self, max_rounds: int = 20) -> Optional[Any]:
        """
        运行完整游戏循环（模板方法）。

        流程：
            start() → while 未结束且未超轮次:
                         setup_round() → run_round() → check_end_condition()
                     → _finalize() → print_summary()

        Args:
            max_rounds: 最大轮数上限（防止无限循环）

        Returns:
            游戏结算数据（类型由子类 _finalize() 决定，通常为 GameResult 或 None）
        """
        if not self.start():
            print("❌ 游戏启动失败")
            return None

        while not self.ended and self.current_round < max_rounds:
            if not self.setup_round():
                break
            if not self.run_round():
                break
            self.check_end_condition()

        result = self._finalize()
        self.print_summary()
        return result

    def _finalize(self) -> Optional[Any]:
        """
        内部结算钩子 — 子类可覆盖。

        标记结束、计算耗时、生成该游戏专用的结算数据。
        返回的结算数据同时作为 run() 的返回值。
        """
        self.ended = True
        return None
