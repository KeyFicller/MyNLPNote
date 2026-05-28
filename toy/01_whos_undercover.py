#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
谁是卧底 - AI 对战版
====================

多名 AI 玩家参与的"谁是卧底"游戏，使用 DeepSeek API 进行推理和决策。

游戏规则：
- 多数玩家为平民（真词），少数为卧底（假词）
- 每轮轮流描述自己的词（不能直接说出词），再投票淘汰一人
- 所有卧底被淘汰 → 平民胜；存活卧底人数 > 存活平民人数 → 卧底胜

人数与卧底数量可通过 GameConfig 或环境变量配置（见 DEFAULT_GAME_CONFIG）。

特点：
- 完整的 AI 思考过程展示
- Function Calling 投票机制
- 结构化游戏历史记录
"""

from random import shuffle
from typing import List, Dict, Any, Optional, TextIO
from dataclasses import dataclass, field
from datetime import datetime
import openai
import os
import json
import atexit
import builtins
import re


# =====================================================
# 游戏日志系统（类似骗子酒馆）
# =====================================================

_LOG_FILE: TextIO | None = None
_LOG_PATH: str | None = None
_ORIGINAL_PRINT = builtins.print

ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """去除 ANSI 颜色代码"""
    return ANSI_ESCAPE_RE.sub("", text)


def setup_game_logging(log_dir: str | None = None) -> str:
    """
    设置游戏日志系统
    
    复写 print，将输出写入带时间戳的日志文件（同时保留控制台输出）
    
    Args:
        log_dir: 日志目录，默认为当前文件所在目录的 logs 子目录
        
    Returns:
        日志文件的完整路径
    """
    global _LOG_FILE, _LOG_PATH

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(log_dir, f"undercover_{ts}.log")
    _LOG_FILE = open(_LOG_PATH, "w", encoding="utf-8")
    _LOG_FILE.write(f"=== 谁是卧底 {datetime.now().isoformat(timespec='seconds')} ===\n")
    _LOG_FILE.flush()

    builtins.print = game_print
    atexit.register(close_game_logging)
    return _LOG_PATH


def game_print(*args, **kwargs):
    """
    替代内置 print：写入日志；默认仍输出到控制台

    Args:
        *args: 要打印的内容
        **kwargs: 其他参数
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


# =====================================================
# 配置常量
# =====================================================

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
VOTE_SUMMARY_MAX_LEN = 100
VOTE_MAX_RETRIES = 2
DESCRIPTION_MAX_RETRIES = 3
DESCRIPTION_SIMILARITY_THRESHOLD = 0.72


@dataclass
class GameConfig:
    """游戏人数与胜负规则配置（勿在业务逻辑中写死人数）"""

    player_num: int = 8
    undercover_num: int = 3
    max_rounds: int = 20

    def __post_init__(self) -> None:
        if self.player_num < 3:
            raise ValueError(f"player_num 至少为 3，当前为 {self.player_num}")
        if self.undercover_num < 1:
            raise ValueError(f"undercover_num 至少为 1，当前为 {self.undercover_num}")
        if self.undercover_num >= self.player_num:
            raise ValueError(
                f"undercover_num ({self.undercover_num}) 必须小于 player_num ({self.player_num})"
            )
        if self.civilian_num <= self.undercover_num:
            raise ValueError(
                f"开局平民数 ({self.civilian_num}) 必须大于卧底数 ({self.undercover_num})，"
                "否则平民无法通过淘汰卧底获胜"
            )
        if self.max_rounds < 1:
            raise ValueError(f"max_rounds 至少为 1，当前为 {self.max_rounds}")

    @property
    def civilian_num(self) -> int:
        return self.player_num - self.undercover_num

    @classmethod
    def from_env(cls) -> "GameConfig":
        """从环境变量读取配置（未设置则用默认值）"""
        return cls(
            player_num=int(os.getenv("UNDERCOVER_PLAYER_NUM", "8")),
            undercover_num=int(os.getenv("UNDERCOVER_UNDERCOVER_NUM", "3")),
            max_rounds=int(os.getenv("UNDERCOVER_MAX_ROUNDS", "20")),
        )


DEFAULT_GAME_CONFIG = GameConfig.from_env()


# =====================================================
# 数据类定义
# =====================================================

@dataclass
class TurnRecord:
    """单轮游戏记录"""
    round_num: int
    descriptions: Dict[int, str] = field(default_factory=dict)
    votes: Dict[int, int] = field(default_factory=dict)
    eliminated: Optional[int] = None
    reason: str = ""


@dataclass
class GameResult:
    """游戏结果"""
    winner: str  # "civilian" 或 "undercover"
    undercover_indices: List[int]
    true_word: str
    fake_word: str
    rounds: int
    duration: float


# =====================================================
# OpenAI 客户端（共享）
# =====================================================

def create_client() -> Optional[openai.OpenAI]:
    """
    创建并返回 OpenAI 客户端实例
    
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


# =====================================================
# 第一部分：玩家类
# =====================================================

class Player:
    """
    AI 玩家类
    
    每个玩家通过 DeepSeek API 进行推理和决策，包括：
    - 词语描述（模糊但准确）
    - 投票决策（基于历史信息推理）
    """
    
    def __init__(
        self,
        index: int,
        client: Optional[openai.OpenAI],
        config: GameConfig,
    ):
        """
        初始化玩家
        
        Args:
            index: 玩家编号（0 起）
            client: OpenAI 客户端实例
            config: 游戏配置
        """
        self.index = index
        self.config = config
        self.word: Optional[str] = None
        self.is_alive = True
        self.think_history: List[str] = []  # 思考历史
        self.description_history: List[str] = []  # 发言历史
        self.client = client
        self.suspicion_level = 0  # 怀疑度（被投票次数）
    
    def set_word(self, word: str) -> None:
        """设置玩家的词"""
        self.word = word
    
    @staticmethod
    def _normalize_description_text(text: str) -> str:
        """归一化描述文本，便于相似度比较"""
        return re.sub(r"[\s，。、；：！？\"'「」『』（）\-\—]", "", text.strip())

    @staticmethod
    def _description_similarity(a: str, b: str) -> float:
        """计算两条描述的相似度（0~1，越高越像）"""
        na = Player._normalize_description_text(a)
        nb = Player._normalize_description_text(b)
        if not na or not nb:
            return 0.0
        if na == nb:
            return 1.0
        if len(na) >= 8 and (na in nb or nb in na):
            return 0.95
        set_a, set_b = set(na), set(nb)
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    def _is_description_too_similar(self, description: str, others: List[str]) -> bool:
        """是否与已有描述高度重复"""
        for other in others:
            if self._description_similarity(description, other) >= DESCRIPTION_SIMILARITY_THRESHOLD:
                return True
        return False

    def _build_description_prompt(
        self,
        public_history: str,
        alive_players: List[int],
        current_round: int,
        recent_others: List[tuple[int, str]],
    ) -> str:
        """
        构建描述阶段的系统提示词
        
        Args:
            public_history: 格式化的各轮公开历史（发言/投票/淘汰）
            alive_players: 存活玩家列表
            current_round: 当前轮次
            
        Returns:
            系统提示词字符串
        """
        my_past = self.description_history[:-1]  # 不含本轮即将生成的
        my_past_str = (
            "\n".join(f"  第{i + 1}次: 「{d}」" for i, d in enumerate(my_past))
            if my_past
            else "  （本轮为首次发言）"
        )
        
        if recent_others:
            others_block = "\n".join(
                f"  - 玩家{pid}: 「{desc}」" for pid, desc in recent_others
            )
            diversity_hint = (
                "【他人已说的描述 — 严禁照抄】\n"
                f"{others_block}\n\n"
                "你的描述必须与以上每一条都明显不同：换角度（习俗/口感/外观/制作/场景/寓意等），"
                "禁止套用同一句式或只替换个别词语。"
            )
        else:
            diversity_hint = "（本轮你是第一个发言，可从任意合理角度描述。）"

        cfg = self.config
        return f"""你是"谁是卧底"游戏中的玩家{self.index}号。

【游戏规则】
- 共{cfg.player_num}名玩家：{cfg.civilian_num}名平民（相同真词）、{cfg.undercover_num}名卧底（相似假词）
- 每轮轮流描述自己的词，不能直接说出词本身
- 描述后投票淘汰一名玩家
- 所有卧底被淘汰 → 平民胜；存活卧底人数 > 存活平民人数 → 卧底胜

【当前状态】
- 你的词："{self.word}"
- 存活玩家：{alive_players}
- 你是玩家：{self.index}号
- 当前轮次：第{current_round}轮

【各轮公开历史 — 务必参考】
{public_history}

{diversity_hint}

【你过往的发言】
{my_past_str}

【你过往的思考】
{self.think_history[-3:] if self.think_history else "暂无"}

【任务要求】
1. 先分析：根据历史发言判断自己是不是卧底
2. 再描述：用一句话描述这个词的特征（不能直接说出词）
3. 描述要有个人特色：即使词相同，也要从【不同侧面】表述，避免与他人撞车

【身份判断】
- 如果你的词和其他人的描述方向一致 → 你是平民，正常描述
- 如果你的词和其他人的描述方向不同 → 你是卧底，必须伪装

【卧底伪装策略 - 关键】
你是卧底时，绝对不能描述你词的独特特征！要这样伪装：
1. 仔细分析平民的描述，找到他们词的共同特征
2. 找出你的词和平民词的【共同点】（功能、类别、使用场景等）
3. 只描述这些共同点，绝口不提你的词独有的特征

【正确示例 - 平民词"蜡烛"，卧底词"灯泡"】
平民描述："用蜡做的，点燃后会融化"（蜡烛特征）
卧底伪装："可以提供光亮的东西，人们常用来照明"（共同点：照明功能）
✓ 卧底说了"照明"这个共同点，没说"通电""玻璃"等灯泡独有特征

【错误示例 - 会暴露】
平民描述："用蜡做的，点燃后会融化"
卧底说："需要通电才能亮，有玻璃外壳"（× 暴露！说了灯泡独有特征）

【输出格式】
总结: <一句话总结你的推理和策略（100字以内）>
描述: <一句话描述这个词的特征，不要直接说出词>

【描述多样性示例 — 词都是「汤圆」时】
- 玩家A: 正月十五常吃，软糯有馅，煮在清水里浮起来
- 玩家B: 团圆寓意，外皮用糯米面搓圆，甜馅居多
- 玩家C: 北方冬至也吃，和元宵长得像但包法不同
× 错误：五个人都说「糯米粉包裹馅料搓圆煮熟」——高度重复，禁止"""

    def _parse_description_response(self, content: str) -> tuple[str, str]:
        """解析模型返回的思考与描述"""
        thinking = ""
        description = ""
        summary_match = None
        desc_match = None

        for line in content.split("\n"):
            line_stripped = line.strip()
            if line_stripped.lower().startswith("总结:"):
                summary_match = line_stripped[3:].strip()
            elif line_stripped.startswith("总结："):
                summary_match = line_stripped[3:].strip()
            elif line_stripped.lower().startswith("描述:"):
                desc_match = line_stripped[3:].strip()
            elif line_stripped.startswith("描述："):
                desc_match = line_stripped[3:].strip()

        if summary_match and desc_match:
            thinking = summary_match
            description = desc_match
        elif "######" in content:
            parts = content.split("######", 1)
            thinking = parts[0].strip()
            description = parts[1].strip()
        else:
            lines = content.split("\n")
            if len(lines) >= 2:
                thinking = lines[0].strip()
                description = " ".join(lines[1:]).strip()
            else:
                thinking = "思考过程未正确格式化"
                description = content

        return thinking, description

    def make_description(
        self,
        game: "Game",
        alive_players: List[int],
        current_turn: Optional[TurnRecord] = None,
    ) -> str:
        """
        生成词语描述
        
        Args:
            game: 游戏实例（用于读取完整公开历史）
            alive_players: 存活玩家列表
            current_turn: 本轮进行中记录（含本轮已发言玩家）
            
        Returns:
            生成的描述文本
        """
        if not self.client:
            return "[系统错误：无法连接到AI]"
        
        if not self.is_alive:
            return "[已出局]"
        
        public_history = game.format_public_history_for_prompt(current_turn)
        recent_others = game.collect_recent_descriptions(
            current_turn, exclude_player=self.index
        )
        other_texts = [desc for _, desc in recent_others]

        system_prompt = self._build_description_prompt(
            public_history, alive_players, game.current_round, recent_others
        )
        user_msg = "请根据公开历史分析并生成你的描述，务必与他人已有描述明显不同。"

        for attempt in range(1, DESCRIPTION_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.75 if attempt == 1 else 0.9,
                    max_tokens=200,
                )

                content = response.choices[0].message.content.strip()
                thinking, description = self._parse_description_response(content)

                if other_texts and self._is_description_too_similar(description, other_texts):
                    print(
                        f"    ⚠️ 玩家{self.index} 描述与已有发言过于相似(尝试{attempt}/"
                        f"{DESCRIPTION_MAX_RETRIES})，要求重写"
                    )
                    user_msg = (
                        "你的描述与他人重复度太高。请换一个完全不同的角度重新描述，"
                        "禁止复述「糯米粉/搓圆/煮熟/馅料」等同一句式。"
                        "仍按「总结:」「描述:」格式输出。"
                    )
                    continue

                self.think_history.append(thinking)
                self.description_history.append(description)
                print(f"    💭 思考：{thinking}")
                return description

            except Exception as e:
                print(f"    ❌ 生成描述时出错(尝试{attempt}): {e}")
                if attempt >= DESCRIPTION_MAX_RETRIES:
                    break

        return "这是一种有特定食用或文化场景的传统食品，别称和做法各地不太一样"

    def _build_vote_prompt(
        self,
        public_history: str,
        alive_players: List[int],
        current_round: int,
    ) -> str:
        """
        构建投票阶段的系统提示词
        
        Args:
            public_history: 格式化的各轮公开历史
            alive_players: 存活玩家列表
            current_round: 当前轮次
            
        Returns:
            系统提示词字符串
        """
        cfg = self.config
        return f"""你是"谁是卧底"游戏中的玩家{self.index}号，现在需要投票淘汰一名玩家。

【游戏规则】
- 共{cfg.player_num}名玩家：{cfg.civilian_num}名平民、{cfg.undercover_num}名卧底
- 投票淘汰得票最多的玩家
- 所有卧底出局则平民胜；存活卧底多于存活平民则卧底胜

【当前状态】
- 你的词："{self.word}"
- 存活玩家：{alive_players}
- 你是玩家：{self.index}号（不能投给自己）
- 当前轮次：第{current_round}轮

【各轮公开历史 — 务必参考所有人发言】
{public_history}

【你过往的思考】
{self.think_history[-3:] if self.think_history else "暂无"}

【投票策略】
1. 分析其他玩家的描述，找出与你词义不符的人
2. 注意：卧底会尽量模仿平民的描述，但可能有细微差别
3. 如果多人可疑，选择最可疑的一个
4. 绝对不能投给自己！

【投票流程 — 分两步，不可跳过】
第一步（思考）：只输出一行「总结: …」，分析可疑玩家与理由，禁止调用工具
第二步（投票）：根据你的总结，调用 vote 工具完成投票

【硬性要求】
- 总结不得超过 {VOTE_SUMMARY_MAX_LEN} 字，禁止 Markdown 标题与长篇推演
- 可投票编号：存活玩家 {alive_players} 中除你自己（{self.index}）以外
- 以【当前状态】中的存活列表为准，勿纠结上一轮是否已淘汰

【错误示例】
- 未写总结就直接投票
- 只写分析不调用 vote
- 超过 100 字的长篇推理"""

    @staticmethod
    def _extract_vote_summary(raw_content: str) -> str:
        """从投票回复中提取简短总结"""
        if not raw_content:
            return ""
        for line in raw_content.split("\n"):
            s = line.strip()
            if s.lower().startswith("总结:"):
                return s[3:].strip()[:VOTE_SUMMARY_MAX_LEN]
            if s.startswith("总结："):
                return s[3:].strip()[:VOTE_SUMMARY_MAX_LEN]
        compact = re.sub(r"\s+", " ", raw_content.strip())
        return compact[:VOTE_SUMMARY_MAX_LEN] + ("…" if len(compact) > VOTE_SUMMARY_MAX_LEN else "")

    @staticmethod
    def _is_valid_vote_summary(summary: str) -> bool:
        """判断思考总结是否有效（非空且有一定信息量）"""
        s = summary.strip()
        if not s:
            return False
        if s.startswith("（无") or s == "思考过程未正确格式化":
            return False
        # 至少 4 个字符，避免「好的」之类敷衍
        return len(s) >= 4

    @staticmethod
    def _parse_vote_target_from_text(
        text: str, voter_index: int, alive_players: List[int]
    ) -> Optional[int]:
        """从文本中解析投票目标（兜底）"""
        candidates = [i for i in alive_players if i != voter_index]
        if not candidates:
            return None
        patterns = [
            r"投票给\s*玩家?\s*(\d+)",
            r"投给\s*玩家?\s*(\d+)",
            r"淘汰\s*玩家?\s*(\d+)",
            r"怀疑\s*玩家?\s*(\d+)",
            r"target_index[\"']?\s*[:=]\s*(\d+)",
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.I):
                target = int(m.group(1))
                if target in candidates:
                    return target
        return None

    def _execute_vote_tool_call(
        self,
        tool_call: Any,
        tools_map: Dict[str, Any],
        game: "Game",
    ) -> Optional[int]:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments or "{}")
        if tool_name not in tools_map:
            print(f"    ⚠️ 无效的工具名: {tool_name}")
            return None
        tool_args["game"] = game
        tool_args["voter_index"] = self.index
        if tools_map[tool_name](**tool_args):
            return tool_args.get("target_index")
        return None

    def _apply_fallback_vote(
        self,
        game: "Game",
        alive_players: List[int],
        tools_map: Dict[str, Any],
        hint_text: str = "",
    ) -> Optional[int]:
        """未调用工具时的强制兜底投票"""
        candidates = [i for i in alive_players if i != self.index]
        if not candidates:
            return None
        target = self._parse_vote_target_from_text(hint_text, self.index, alive_players)
        if target is None:
            target = candidates[0]
            print(f"    ⚠️ 玩家{self.index} 未调用 vote，系统随机兜底 → 玩家{target}")
        else:
            print(f"    ⚠️ 玩家{self.index} 未调用 vote，从文本解析兜底 → 玩家{target}")
        vote(game=game, voter_index=self.index, target_index=target)
        return target

    def _run_vote_think_phase(self, system_prompt: str, candidates: List[int]) -> str:
        """第一步：仅输出思考总结，不调用工具"""
        think_user = (
            "【第一步：思考】\n"
            "请分析各玩家发言，找出最可疑的人。\n"
            "只输出一行，格式必须是：总结: <你的分析（100字以内）>\n"
            "禁止调用任何工具；禁止输出第二行或 Markdown 标题。"
        )
        last_raw = ""
        for attempt in range(1, VOTE_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": think_user},
                    ],
                    temperature=0.6,
                    max_tokens=180,
                )
                last_raw = response.choices[0].message.content or ""
                summary = self._extract_vote_summary(last_raw)
                if self._is_valid_vote_summary(summary):
                    return summary
                print(f"    ⚠️ 玩家{self.index} 第{attempt}次思考格式无效，重试")
                think_user = (
                    "你的回复不合格。请严格只输出一行：\n"
                    "总结: （一句话说明怀疑谁及理由，100字内）\n"
                    "不要调用工具，不要写标题或分点。"
                )
            except Exception as e:
                print(f"    ❌ 玩家{self.index}思考阶段出错(尝试{attempt}): {e}")
                if attempt >= VOTE_MAX_RETRIES:
                    break
        # 思考阶段兜底：用截断原文或默认句
        fallback = self._extract_vote_summary(last_raw)
        if not self._is_valid_vote_summary(fallback):
            fallback = f"综合发言，怀疑玩家{candidates[0]}描述与我的词不符"
        return fallback[:VOTE_SUMMARY_MAX_LEN]

    def _run_vote_action_phase(
        self,
        system_prompt: str,
        summary: str,
        candidates: List[int],
        tools_schema: List[Dict],
        tools_map: Dict[str, Any],
        game: "Game",
    ) -> Optional[int]:
        """第二步：根据总结调用 vote 工具"""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "【第一步：思考】请先分析并写总结（已完成则见下条助手消息）。"
                ),
            },
            {"role": "assistant", "content": f"总结: {summary}"},
            {
                "role": "user",
                "content": (
                    "【第二步：投票】\n"
                    "你已写完总结。现在不要重复分析，"
                    f"必须调用 vote 工具，target_index 从 {candidates} 中选择一个。"
                ),
            },
        ]
        for attempt in range(1, VOTE_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice={"type": "function", "function": {"name": "vote"}},
                    temperature=0.3,
                    max_tokens=128,
                )
                message = response.choices[0].message
                tool_calls = message.tool_calls
                if tool_calls:
                    for tool_call in tool_calls:
                        target = self._execute_vote_tool_call(tool_call, tools_map, game)
                        if target is not None:
                            return target
                print(f"    ⚠️ 玩家{self.index} 第{attempt}次未返回有效 vote")
                messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (tool_calls or [])
                    ],
                })
                messages.append({
                    "role": "user",
                    "content": (
                        f"请立即调用 vote，target_index 只能是 {candidates} 之一，禁止再写分析。"
                    ),
                })
            except Exception as e:
                print(f"    ❌ 玩家{self.index}投票阶段出错(尝试{attempt}): {e}")
        return None

    def make_vote(
        self,
        game: "Game",
        alive_players: List[int],
        tools_schema: List[Dict],
        tools_map: Dict[str, Any],
        current_turn: Optional[TurnRecord] = None,
    ) -> Optional[int]:
        """
        进行投票：先思考总结，再调用 vote 工具（必须产生有效投票）
        
        Returns:
            投票目标的玩家编号
        """
        if not self.client:
            print(f"    ❌ 玩家{self.index}: API 未配置，跳过投票")
            return None
        
        if not self.is_alive:
            return None

        candidates = [i for i in alive_players if i != self.index]
        if not candidates:
            print(f"    ⚠️ 玩家{self.index} 无合法投票对象")
            return None

        public_history = game.format_public_history_for_prompt(current_turn)
        system_prompt = self._build_vote_prompt(
            public_history, alive_players, game.current_round
        )

        # 第一步：思考（无工具）
        summary = self._run_vote_think_phase(system_prompt, candidates)
        print(f"    💭 思考：{summary}")
        self.think_history.append(summary)

        # 第二步：投票（强制工具）
        target = self._run_vote_action_phase(
            system_prompt, summary, candidates, tools_schema, tools_map, game
        )
        if target is not None:
            return target

        hint = f"总结: {summary}"
        return self._apply_fallback_vote(game, alive_players, tools_map, hint)


# =====================================================
# 第二部分：裁判类
# =====================================================

class Judger:
    """游戏裁判，负责生成词语"""
    
    def __init__(self, client: Optional[openai.OpenAI]):
        """
        初始化裁判
        
        Args:
            client: OpenAI 客户端实例
        """
        self.client = client
        self.true_word: Optional[str] = None
        self.fake_word: Optional[str] = None
        self.word_pair_history: List[tuple] = []  # 记录用过的词对，避免重复
    
    def _build_word_prompt(self) -> str:
        """构建生成词语的系统提示词"""
        used_pairs = json.dumps(self.word_pair_history, ensure_ascii=False)
        
        return f"""你是"谁是卧底"游戏的裁判，需要生成一对相似但有区别的词语。

【核心要求】
1. 两个词必须是同一类别（如都是水果、都是动物）
2. 词义相近但**必须不同**（绝对不能相同！）
3. 让卧底有发挥空间，但平民能识别差异
4. 不要使用以下已用过的词对：{used_pairs if self.word_pair_history else "无"}

【重要提醒】
- 生成的两个词如果相同，游戏将无法进行
- 请仔细检查你生成的两个词是否不同

【优秀示例】
- 苹果,香蕉（都是常见水果，词不同）
- 猫,狗（都是宠物，词不同）
- 火车,地铁（都是交通工具，词不同）
- 衬衫,T恤（都是上衣，词不同）

【错误示例】
- 裁判,裁判（× 两个词相同！）
- 苹果,苹果（× 两个词相同！）

【输出格式】
直接输出：真词,假词
例如：苹果,香蕉"""

    def generate_word(self, config: GameConfig) -> List[str]:
        """
        生成游戏词语并分配给玩家
        
        Args:
            config: 游戏配置（人数、卧底数）
            
        Returns:
            分配给每个玩家的词列表
        """
        if not self.client:
            print("⚠️ 警告: 使用默认词对（API未配置）")
            return self._get_default_words(config)
        
        # 尝试3次生成合适的词
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self._build_word_prompt()},
                        {"role": "user", "content": "生成词对"}
                    ],
                    temperature=0.9,
                    max_tokens=50
                )
                
                content = response.choices[0].message.content.strip()
                
                # 解析词对
                # 处理可能的多种格式："苹果,香蕉" 或 "苹果，香蕉" 或 "真词:苹果 假词:香蕉"
                words = self._parse_word_pair(content)
                
                if len(words) >= 2:
                    word1, word2 = words[0], words[1]
                    
                    # 再次检查两个词是否不同
                    if word1 == word2:
                        print(f"⚠️ 生成的两个词相同(尝试{attempt+1}/3): '{word1}'，重新生成")
                        continue
                    
                    self.true_word = word1
                    self.fake_word = word2
                    
                    # 记录词对
                    self.word_pair_history.append((self.true_word, self.fake_word))
                    
                    player_words = self._build_player_word_list(config)
                    shuffle(player_words)
                    
                    print(
                        f"✅ 成功生成词对：平民词'{self.true_word}'，卧底词'{self.fake_word}' "
                        f"（{config.undercover_num}卧底 / {config.civilian_num}平民）"
                    )
                    return player_words
                else:
                    print(f"⚠️ 生成的词格式不正确(尝试{attempt+1}/3): {content}")
                    
            except Exception as e:
                print(f"⚠️ 生成词语时出错(尝试{attempt+1}/3): {e}")
        
        # 如果3次都失败，使用默认词
        print("使用默认词对")
        return self._get_default_words(config)

    def _build_player_word_list(self, config: GameConfig) -> List[str]:
        """按配置分配卧底词与平民词"""
        return (
            [self.fake_word] * config.undercover_num
            + [self.true_word] * config.civilian_num
        )
    
    def _parse_word_pair(self, content: str) -> List[str]:
        """
        解析词对，处理多种可能的格式
        
        Args:
            content: AI 返回的文本
            
        Returns:
            词列表（两个不同的词）
        """
        # 移除可能的引号和说明文字
        content = content.strip().strip('"').strip("'")
        
        # 尝试多种分隔符
        for separator in [',', '，', '|', '/', ' ']:
            if separator in content:
                words = [w.strip() for w in content.split(separator)]
                # 过滤掉空字符串和太长的解释文字
                words = [w for w in words if w and len(w) <= 10]
                if len(words) >= 2:
                    # 检查两个词是否不同
                    word1, word2 = words[0], words[1]
                    if word1 != word2:
                        return [word1, word2]
                    else:
                        print(f"    ⚠️ 生成的两个词相同: '{word1}'，尝试其他分隔符")
        
        # 尝试正则提取
        import re
        # 匹配 "真词:xxx 假词:yyy" 或 "卧底词:xxx 平民词:yyy" 等格式
        patterns = [
            r'[:：]\s*([^,，\s]+)',
            r'([^,，\s]+)\s*和\s*([^,，\s]+)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, content)
            if matches and len(matches) >= 2:
                word1, word2 = matches[0], matches[1]
                if word1 != word2:
                    return [word1, word2]
                else:
                    print(f"    ⚠️ 正则提取的两个词相同: '{word1}'")
        
        return []
    
    def _get_default_words(self, config: GameConfig) -> List[str]:
        """
        获取默认词对
        
        Args:
            config: 游戏配置
            
        Returns:
            默认词列表
        """
        default_pairs = [
            ("苹果", "香蕉"),
            ("猫", "狗"),
            ("火车", "地铁"),
            ("衬衫", "T恤"),
            ("咖啡", "奶茶"),
            ("篮球", "足球"),
        ]
        
        # 选择一个没用过的词对
        for pair in default_pairs:
            if pair not in self.word_pair_history:
                self.true_word, self.fake_word = pair
                self.word_pair_history.append(pair)
                break
        else:
            # 都用过了，重置历史
            self.word_pair_history = []
            self.true_word, self.fake_word = default_pairs[0]
            self.word_pair_history.append(default_pairs[0])
        
        player_words = self._build_player_word_list(config)
        shuffle(player_words)
        return player_words


# =====================================================
# 第三部分：游戏主类
# =====================================================

class Game:
    """游戏主控类"""
    
    def __init__(self, config: GameConfig | None = None):
        """
        初始化游戏
        
        Args:
            config: 游戏配置；默认使用 DEFAULT_GAME_CONFIG
        """
        self.config = config or DEFAULT_GAME_CONFIG
        self.player_num = self.config.player_num
        self.undercover_num = self.config.undercover_num
        self.client = create_client()
        self.players: List[Player] = [
            Player(i, self.client, self.config) for i in range(self.player_num)
        ]
        self.judger = Judger(self.client)
        self.game_history: List[Dict] = []
        self.turn_records: List[TurnRecord] = []  # 结构化回合记录
        self.current_round = 0
        self.votes: Dict[int, int] = {i: 0 for i in range(self.player_num)}
        self.ended = False
        self.start_time: Optional[datetime] = None
        self.result: Optional[GameResult] = None
    
    def game_start(self) -> bool:
        """
        初始化游戏，分配词语
        
        Returns:
            是否成功启动
        """
        print("\n" + "="*60)
        print("🎮 谁是卧底 - AI 对战版")
        print("="*60)
        
        if not self.client:
            print("\n⚠️ 警告: API 未配置，将使用默认词对运行")
        
        self.start_time = datetime.now()
        
        # 生成并分配词语
        words = self.judger.generate_word(self.config)
        for i, player in enumerate(self.players):
            player.set_word(words[i])
        
        # 打印游戏信息
        print(f"\n📋 游戏信息:")
        print(f"   玩家人数: {self.player_num}（平民 {self.config.civilian_num} / 卧底 {self.undercover_num}）")
        print(f"   平民词: {self.judger.true_word}")
        print(f"   卧底词: {self.judger.fake_word}")
        print(f"   最大轮数: {self.config.max_rounds}")
        
        # 打印每个玩家的词（调试用，实际游戏应该保密）
        print(f"\n🎲 词语分配:")
        for i, player in enumerate(self.players):
            word_type = "平民" if player.word == self.judger.true_word else "卧底"
            print(f"   玩家{i}: {player.word} [{word_type}]")
        
        return True
    
    def get_alive_players(self) -> List[int]:
        """
        获取存活玩家索引列表
        
        Returns:
            存活玩家的索引列表
        """
        return [i for i, p in enumerate(self.players) if p.is_alive]
    
    def is_undercover(self, player: Player) -> bool:
        """
        判断玩家是否是卧底
        
        Args:
            player: 玩家实例
            
        Returns:
            是否是卧底
        """
        return player.word == self.judger.fake_word
    
    def get_undercover_indices(self) -> List[int]:
        """获取所有卧底玩家的索引（含已出局）"""
        return [i for i, p in enumerate(self.players) if self.is_undercover(p)]

    def count_alive_by_role(self) -> tuple[int, int]:
        """返回 (存活卧底数, 存活平民数)"""
        undercover_alive = 0
        civilian_alive = 0
        for i in self.get_alive_players():
            if self.is_undercover(self.players[i]):
                undercover_alive += 1
            else:
                civilian_alive += 1
        return undercover_alive, civilian_alive
    
    def record_history(self, action: str, detail: str) -> None:
        """
        记录游戏历史
        
        Args:
            action: 动作类型
            detail: 详细内容
        """
        self.game_history.append({
            "round": self.current_round,
            "action": action,
            "detail": detail,
            "timestamp": datetime.now().isoformat()
        })
    
    def reset_votes(self) -> None:
        """重置投票计数"""
        self.votes = {i: 0 for i in range(self.player_num)}

    def _format_turn_record(self, tr: TurnRecord, reveal_eliminated_role: bool = True) -> str:
        """将单轮记录格式化为可读文本"""
        lines = [f"【第{tr.round_num}轮】"]
        if tr.descriptions:
            lines.append("发言：")
            for pid in sorted(tr.descriptions.keys()):
                lines.append(f"  - 玩家{pid}: 「{tr.descriptions[pid]}」")
        if tr.votes:
            lines.append("投票：")
            for voter in sorted(tr.votes.keys()):
                lines.append(f"  - 玩家{voter} → 玩家{tr.votes[voter]}")
        if tr.eliminated is not None:
            elim = self.players[tr.eliminated]
            if reveal_eliminated_role:
                role = "卧底" if self.is_undercover(elim) else "平民"
                lines.append(f"淘汰：玩家{tr.eliminated}（{role}，词：{elim.word}）")
            else:
                lines.append(f"淘汰：玩家{tr.eliminated}")
        return "\n".join(lines)

    def format_public_history_for_prompt(
        self, current_turn: Optional[TurnRecord] = None
    ) -> str:
        """
        生成注入 LLM 的完整公开历史（各轮发言 + 投票 + 淘汰）。
        包含已结束轮次；current_turn 可带上本轮已进行中的发言/投票。
        """
        blocks: List[str] = []
        for tr in self.turn_records:
            blocks.append(self._format_turn_record(tr))

        if current_turn is not None:
            # 当前轮：可能已有发言，投票阶段则描述与投票都齐全
            has_content = bool(current_turn.descriptions or current_turn.votes)
            if has_content:
                # 若该轮已在 turn_records 中（不应重复），跳过
                if not self.turn_records or self.turn_records[-1] is not current_turn:
                    blocks.append(self._format_turn_record(current_turn))

        if not blocks:
            return "（暂无历史，本轮为首次发言）"
        return "\n\n".join(blocks)

    def collect_recent_descriptions(
        self,
        current_turn: Optional[TurnRecord] = None,
        exclude_player: Optional[int] = None,
        max_count: int = 16,
    ) -> List[tuple[int, str]]:
        """
        收集近期他人描述，用于防重复提示与相似度检测。
        返回 (玩家编号, 描述) 列表，按时间顺序。
        """
        items: List[tuple[int, str]] = []
        seen: set[str] = set()

        def add(pid: int, desc: str) -> None:
            if exclude_player is not None and pid == exclude_player:
                return
            norm = Player._normalize_description_text(desc)
            if norm in seen:
                return
            seen.add(norm)
            items.append((pid, desc))

        for tr in self.turn_records:
            for pid in sorted(tr.descriptions.keys()):
                add(pid, tr.descriptions[pid])

        if current_turn is not None:
            for pid in sorted(current_turn.descriptions.keys()):
                add(pid, current_turn.descriptions[pid])

        return items[-max_count:]


# =====================================================
# 第四部分：工具函数定义
# =====================================================

def vote(game: Game, voter_index: int, target_index: int) -> bool:
    """
    投票工具函数
    
    Args:
        game: 游戏实例
        voter_index: 投票玩家编号
        target_index: 被投票玩家编号
        
    Returns:
        投票是否成功
    """
    # 检查玩家索引范围
    if not (0 <= voter_index < game.player_num and 0 <= target_index < game.player_num):
        print(f"    ❌ 无效的玩家编号")
        return False
    
    # 检查投票玩家存活状态
    if not game.players[voter_index].is_alive:
        print(f"    ❌ 玩家{voter_index}已出局，不能投票")
        return False
    
    # 检查目标玩家存活状态
    if not game.players[target_index].is_alive:
        print(f"    ❌ 玩家{target_index}已出局，不能被投票")
        return False
    
    # 检查不能投给自己
    if voter_index == target_index:
        print(f"    ❌ 不能投给自己")
        return False
    
    # 记录投票
    game.votes[target_index] += 1
    game.players[target_index].suspicion_level += 1
    
    game.record_history("vote", f"玩家{voter_index}投票给玩家{target_index}")
    print(f"    ✅ 玩家{voter_index} → 玩家{target_index}")
    
    return True


def build_tools_schema(player_num: int) -> List[Dict]:
    """根据玩家人数生成投票工具 schema"""
    last_index = player_num - 1
    return [
        {
            "type": "function",
            "function": {
                "name": "vote",
                "description": (
                    "【必须调用】投票淘汰一名玩家。"
                    "本轮每位存活玩家都必须调用一次，不可只输出文字分析。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_index": {
                            "type": "integer",
                            "description": (
                                f"被投票的玩家编号（0-{last_index}），"
                                "不能是自己或已出局的玩家"
                            ),
                        }
                    },
                    "required": ["target_index"],
                },
            },
        }
    ]


TOOLS_MAP = {"vote": vote}


def _make_game_result(game: Game, winner: str) -> GameResult:
    return GameResult(
        winner=winner,
        undercover_indices=game.get_undercover_indices(),
        true_word=game.judger.true_word or "",
        fake_word=game.judger.fake_word or "",
        rounds=game.current_round,
        duration=(datetime.now() - game.start_time).total_seconds()
        if game.start_time
        else 0,
    )


def check_win_conditions(game: Game) -> bool:
    """
    检查是否满足终局条件。
    
    Returns:
        若游戏已结束返回 True
    """
    undercover_alive, civilian_alive = game.count_alive_by_role()

    if undercover_alive == 0:
        print(f"\n  🎉 所有卧底已出局！平民获胜！")
        game.record_history("civilian_win", "all undercovers eliminated")
        game.ended = True
        game.result = _make_game_result(game, "civilian")
        return True

    if undercover_alive > civilian_alive:
        print(
            f"\n  🎭 卧底人数占优（卧底 {undercover_alive} : 平民 {civilian_alive}），卧底获胜！"
        )
        game.record_history(
            "undercover_win",
            f"undercover {undercover_alive} > civilian {civilian_alive}",
        )
        game.ended = True
        game.result = _make_game_result(game, "undercover")
        return True

    return False


# =====================================================
# 第五部分：游戏流程控制
# =====================================================

def run_description_phase(game: Game) -> TurnRecord:
    """
    运行描述阶段
    
    Args:
        game: 游戏实例
        
    Returns:
        本回合记录
    """
    turn_record = TurnRecord(round_num=game.current_round)
    alive_players = game.get_alive_players()
    
    print(f"\n{'='*60}")
    print(f"🎤 第{game.current_round}轮 - 发言阶段")
    print(f"{'='*60}")
    
    for i in range(game.player_num):
        player = game.players[i]
        if not player.is_alive:
            continue
        
        # 显示玩家信息
        word_type = "平民" if player.word == game.judger.true_word else "卧底"
        print(f"\n  玩家 {i} [{word_type}]:")
        
        # 生成描述
        description = player.make_description(game, alive_players, current_turn=turn_record)
        turn_record.descriptions[i] = description
        
        # 显示描述
        print(f'    🗣️  "{description}"')
        
        # 记录到游戏历史
        game.record_history(f"player_{i}_description", description)
    
    return turn_record


def run_vote_phase(game: Game, turn_record: TurnRecord) -> Optional[int]:
    """
    运行投票阶段
    
    Args:
        game: 游戏实例
        turn_record: 本回合记录
        
    Returns:
        被淘汰玩家的编号，如果平票返回 None
    """
    alive_players = game.get_alive_players()
    
    print(f"\n{'='*60}")
    print(f"🗳️ 第{game.current_round}轮 - 投票阶段")
    print(f"{'='*60}")
    
    # 重置投票
    game.reset_votes()
    
    for i in range(game.player_num):
        player = game.players[i]
        if not player.is_alive:
            continue
        
        word_type = "平民" if player.word == game.judger.true_word else "卧底"
        print(f"\n  玩家 {i} [{word_type}] 投票:")
        
        # 进行投票
        target = player.make_vote(
            game,
            alive_players,
            build_tools_schema(game.player_num),
            TOOLS_MAP,
            current_turn=turn_record,
        )
        
        if target is not None:
            turn_record.votes[i] = target
    
    # 统计票数
    print(f"\n{'='*60}")
    print("📊 投票结果")
    print(f"{'='*60}")
    
    max_votes = 0
    top_players = []
    
    for i in range(game.player_num):
        player = game.players[i]
        if not player.is_alive:
            continue
        
        vote_count = game.votes.get(i, 0)
        print(f"  玩家{i}: {vote_count} 票")
        
        if vote_count > max_votes:
            top_players = [i]
            max_votes = vote_count
        elif vote_count == max_votes and vote_count > 0:
            top_players.append(i)
    
    # 判断结果
    if len(top_players) == 1:
        eliminated = top_players[0]
        print(f"\n  ☠️ 玩家{eliminated} 被淘汰（{max_votes}票）")
        turn_record.eliminated = eliminated
        return eliminated
    else:
        print(f"\n  ⚖️ 平票！最高票玩家: {top_players}（各{max_votes}票）")
        print(f"     本轮无人淘汰，游戏继续")
        return None


def eliminate_player(game: Game, player_index: int) -> bool:
    """
    淘汰玩家并判断游戏是否结束
    
    Args:
        game: 游戏实例
        player_index: 被淘汰玩家编号
        
    Returns:
        游戏是否结束
    """
    eliminated = game.players[player_index]
    eliminated.is_alive = False
    
    was_undercover = game.is_undercover(eliminated)
    undercover_alive, civilian_alive = game.count_alive_by_role()
    
    print(f"\n{'='*60}")
    print("💀 淘汰公告")
    print(f"{'='*60}")
    print(f"  玩家{player_index} 被淘汰")
    print(f"  身份: {'卧底' if was_undercover else '平民'}")
    print(f"  词语: {eliminated.word}")
    print(f"  场上剩余: 卧底 {undercover_alive} 人，平民 {civilian_alive} 人")
    
    if was_undercover:
        game.record_history("undercover_eliminated", f"player {player_index}")
        if undercover_alive > 0:
            print(f"  仍有 {undercover_alive} 名卧底存活，游戏继续")
        else:
            print(f"  最后一名卧底已出局")
    else:
        print(f"  💔 平民被淘汰")
        game.record_history("civilian_eliminated", f"player {player_index}")
    
    return check_win_conditions(game)


def run_game_loop(game: Game) -> GameResult:
    """
    运行完整游戏循环
    
    Args:
        game: 游戏实例
        
    Returns:
        游戏结果
    """
    print(f"\n{'='*60}")
    print("🎮 游戏开始！")
    print(f"{'='*60}")
    
    while not game.ended:
        game.current_round += 1
        
        # 描述阶段
        turn_record = run_description_phase(game)
        game.turn_records.append(turn_record)
        
        # 投票阶段
        eliminated = run_vote_phase(game, turn_record)
        
        if eliminated is not None:
            # 有人被淘汰
            if eliminate_player(game, eliminated):
                break
        else:
            # 平票，继续
            print(f"\n  ⏳ 本轮平票，无人淘汰")
            game.record_history("tie", "no elimination")
        
        if check_win_conditions(game):
            break
        
        # 检查最大轮数限制（防止无限循环）
        if game.current_round >= game.config.max_rounds:
            print(f"\n  ⏰ 达到最大轮数限制（{game.config.max_rounds}），游戏结束")
            game.ended = True
            break
    
    return game.result or GameResult(
        winner="unknown",
        undercover_indices=[],
        true_word="",
        fake_word="",
        rounds=game.current_round,
        duration=0,
    )


def print_game_summary(game: Game) -> None:
    """
    打印游戏总结
    
    Args:
        game: 游戏实例
    """
    print(f"\n{'='*60}")
    print("📋 游戏总结")
    print(f"{'='*60}")
    
    if game.result:
        print(f"\n  获胜方: {'🎉 平民' if game.result.winner == 'civilian' else '🎭 卧底'}")
        undercover_str = ", ".join(f"玩家{i}" for i in game.result.undercover_indices)
        print(f"  卧底玩家: {undercover_str or '无'}")
        print(f"  平民词: {game.result.true_word}")
        print(f"  卧底词: {game.result.fake_word}")
        print(f"  总轮数: {game.result.rounds}")
        print(f"  游戏时长: {game.result.duration:.2f}秒")
    
    print(f"\n  详细记录:")
    print(f"  {'轮次':<6} {'阶段':<12} {'详情'}")
    print(f"  {'-'*50}")
    
    for record in game.game_history:
        round_num = record.get('round', '-')
        action = record.get('action', '')[:10]
        detail = record.get('detail', '')[:40]
        print(f"  {round_num:<6} {action:<12} {detail}")
    
    print(f"\n  玩家表现:")
    for i, player in enumerate(game.players):
        status = "存活" if player.is_alive else "淘汰"
        identity = "卧底" if game.is_undercover(player) else "平民"
        print(f"    玩家{i}: {identity} | {status} | 被投{player.suspicion_level}票")


# =====================================================
# 第六部分：主程序入口
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="谁是卧底 - AI 对战")
    parser.add_argument(
        "--players",
        type=int,
        default=int(os.getenv("UNDERCOVER_PLAYER_NUM", "8")),
        help="玩家人数（默认 8，可用环境变量 UNDERCOVER_PLAYER_NUM）",
    )
    parser.add_argument(
        "--undercovers",
        type=int,
        default=int(os.getenv("UNDERCOVER_UNDERCOVER_NUM", "3")),
        help="卧底人数（默认 3，可用环境变量 UNDERCOVER_UNDERCOVER_NUM）",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=int(os.getenv("UNDERCOVER_MAX_ROUNDS", "20")),
        help="最大轮数（默认 20，可用环境变量 UNDERCOVER_MAX_ROUNDS）",
    )
    args = parser.parse_args()

    try:
        config = GameConfig(
            player_num=args.players,
            undercover_num=args.undercovers,
            max_rounds=args.max_rounds,
        )
    except ValueError as e:
        _ORIGINAL_PRINT(f"❌ 配置错误: {e}")
        raise SystemExit(1) from e

    # 设置日志系统
    log_path = setup_game_logging()
    _ORIGINAL_PRINT(f"📝 日志文件: {log_path}")

    game = Game(config=config)

    if game.game_start():
        run_game_loop(game)
        print_game_summary(game)
    else:
        print("❌ 游戏启动失败")

    close_game_logging()
