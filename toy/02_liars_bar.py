#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import atexit
import builtins
import openai
import random
import json
from datetime import datetime
from typing import Dict, List, Any, TextIO

_LOG_FILE: TextIO | None = None
_LOG_PATH: str | None = None
_ORIGINAL_PRINT = builtins.print

ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def setup_game_logging(log_dir: str | None = None) -> str:
    """复写 print，将输出写入带时间戳的日志文件（同时保留控制台）"""
    global _LOG_FILE, _LOG_PATH

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOG_PATH = os.path.join(log_dir, f"liars_bar_{ts}.log")
    _LOG_FILE = open(_LOG_PATH, "w", encoding="utf-8")
    _LOG_FILE.write(f"=== Liar's Bar {datetime.now().isoformat(timespec='seconds')} ===\n")
    _LOG_FILE.flush()

    builtins.print = game_print
    atexit.register(close_game_logging)
    return _LOG_PATH


def game_print(*args, **kwargs):
    """替代内置 print：写入日志；默认仍输出到控制台

    log_plain: 可选，写入日志的纯文本（无 ANSI）；控制台仍用 args 拼接结果。
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
    global _LOG_FILE
    if _LOG_FILE and not _LOG_FILE.closed:
        _LOG_FILE.write(f"\n=== END {datetime.now().isoformat(timespec='seconds')} ===\n")
        _LOG_FILE.close()
        _LOG_FILE = None

MAX_THOUGHTS = 20
MAX_OPPONENT_NOTES = 15

CARD_TOKEN_RE = re.compile(r"Joker|J|Q|K|X", re.IGNORECASE)


def _normalize_card(token: str) -> str:
    t = token.strip()
    if t.upper() in ("X", "JOKER"):
        return "Joker"
    return t.upper()


def _parse_card_list(text: str) -> List[str]:
    """从「打出 Q, K」或「打出 [Q,K]」等文本解析实际牌面"""
    if not text:
        return []
    return [_normalize_card(m) for m in CARD_TOKEN_RE.findall(text)]


SUMMARY_LINE_RE = re.compile(
    r"总结\s*[：:]\s*(.+?)(?=\n\s*(?:唬人|PLAY|CHALLENGE|PASS|表演|行为)\s*[:：]|\Z)",
    re.I | re.DOTALL,
)

THOUGHT_MAX_LEN = 100


def _extract_thought(response: str) -> str:
    """提取「总结」行作为思考记录，限制 100 字以内"""
    m = SUMMARY_LINE_RE.search(response)
    if m:
        return m.group(1).strip()[:THOUGHT_MAX_LEN]
    kept = []
    for line in response.strip().splitlines():
        u = line.strip().upper()
        if re.match(r"^(PLAY|CHALLENGE|PASS|唬人|表演|行为|总结)\s*[:：]", u):
            continue
        if u in ("PASS", "CHALLENGE", "PLAY"):
            continue
        kept.append(line)
    text = "\n".join(kept).strip()
    return (text[:THOUGHT_MAX_LEN] if text else response[:THOUGHT_MAX_LEN])


BLUFF_LINE_RE = re.compile(
    r"(?:唬人|表演|行为)\s*[：:]\s*(.+?)"
    r"(?=\n\s*(?:PLAY|CHALLENGE|PASS|唬人|表演|行为)\s*[:：]|\Z)",
    re.I | re.DOTALL,
)


def _parse_bluff_behavior(response: str) -> str:
    m = BLUFF_LINE_RE.search(response)
    if m:
        return m.group(1).strip().strip('"\'""')
    return ""


BLUFF_PERSONAS = [
    "老江湖赌徒，懒洋洋、爱嘲讽",
    "紧张的新手，越想装镇定越露馅",
    "傲慢贵族，用鼻子看人",
    "嬉皮笑脸的混混，满嘴跑火车",
    "沉默寡言的冷面杀手，只出手少说话",
    "话痨销售，夸张吹捧自己的牌",
    "醉汉，含糊不清但莫名自信",
    "学院派分析师，像在做学术报告",
]

BLUFF_ACTIONS = [
    "把牌像筹码一样摞齐，手腕一抖滑到桌心",
    "单指按住牌角，慢条斯理推过去",
    "双手盖住牌堆，停顿半秒才松开",
    "把牌甩成扇形又合拢，故意弄出声响",
    "用指节敲了敲桌面，再把牌丢下",
    "像发扑克一样啪地铺开",
    "捏着牌边在桌沿上蹭了一下，才推出",
    "把牌举到下巴前瞄一眼，随手丢下",
]

BLUFF_POSTURES = [
    "后背靠住椅背，二郎腿一晃",
    "身体前倾，手肘撑桌",
    "耸肩摊手，一脸无所谓",
    "歪头咬唇，假装犹豫后出手",
    "眯眼打量全场，嘴角带笑",
    "转着手里的酒杯，眼皮也不抬",
    "双手抱胸，下巴微扬",
    "刻意深呼吸，装出很轻松的样子",
]

BLUFF_TONES = [
    "轻飘飘地",
    "压低嗓子",
    "拉长尾音",
    "冷笑着说",
    "故作惊讶地",
    "漫不经心地",
    "一字一顿地",
    "嘟囔着",
]

BLUFF_LINES = [
    "「{count} 张 {card}，有问题？」",
    "「瞧好了，{count} 张 {card}。」",
    "「就这？{count} 张 {card} 而已。」",
    "「{count} 张 {card}，爱信不信。」",
    "「哼，{count} 张 {card}，还能有假？」",
    "「{count} 张 {card}——稳的。」",
    "「{count} 张 {card}，别眨眼。」",
    "「{count} 张 {card}，跟不跟？」",
]

BLUFF_QUIRKS = [
    "顺带吐槽上家的表情",
    "故意停顿等别人紧张",
    "假装打哈欠掩饰心虚",
    "用玩笑话岔开话题",
    "敲桌两下当节拍",
    "朝左右扫一眼再出牌",
    "自言自语似的报牌",
    "夸张地吹口气再说话",
]


def _default_bluff_behavior(claimed_count: int, claimed: str) -> str:
    """随机拼装唬人表演（兜底，模型未写时使用）"""
    line = random.choice(BLUFF_LINES).format(count=claimed_count, card=claimed)
    parts = [
        random.choice(BLUFF_ACTIONS),
        random.choice(BLUFF_POSTURES),
        f"{random.choice(BLUFF_TONES)}{line}",
    ]
    if random.random() < 0.4:
        parts.append(random.choice(BLUFF_QUIRKS))
    random.shuffle(parts)
    return "，".join(parts)


def _bluff_style_hint(player_index: int, persona: str) -> str:
    """每手随机风格提示，注入 LLM prompt"""
    quirk = random.choice(BLUFF_QUIRKS)
    action = random.choice(BLUFF_ACTIONS)
    return (
        f"人设：{persona}。"
        f"本手建议动作：{action}；语气细节：{quirk}。"
        f"勿复读常见套话（如「后背靠椅」「轻飘飘」），换新鲜说法。"
    )


def _format_cards_display(cards: List[str]) -> str:
    """日志/观战用手牌展示（Joker → X）"""
    return " ".join("X" if c == "Joker" else c for c in cards)


def _bullet_slots_display(player: "Player") -> tuple[str, str]:
    """返回 (日志用纯文本, 控制台用含 ANSI 着色)"""
    plain_parts: List[str] = []
    color_parts: List[str] = []
    for b in range(BULLET_NUM):
        if b == player.bullet:
            plain_char, color_char = "★", "\033[91m●\033[0m"
        elif b < player.current:
            plain_char = color_char = "○"
        else:
            plain_char = color_char = "●"
        suffix = "▲" if b == player.current and player.current < BULLET_NUM else " "
        plain_parts.append(f"{plain_char}{suffix}")
        color_parts.append(f"{color_char}{suffix}")
    return " ".join(plain_parts), " ".join(color_parts)


def _format_log_text(text: str, max_len: int = 500) -> str:
    """日志展示：超长截断并标注，避免误以为越界错误"""
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}…（共 {len(text)} 字，已截断）"


def _find_final_command(response: str) -> str | None:
    """取回复中最后一条决策行，避免正文里的 CHALLENGE/PLAY 误匹配"""
    for line in reversed(response.strip().splitlines()):
        s = line.strip()
        if not s:
            continue
        u = s.upper()
        if re.match(r"^(PLAY|CHALLENGE|PASS)\s*[:：]", u):
            return s
        # 与 prompt 一致：总结后单独一行 CHALLENGE / PASS / PLAY
        if u in ("PASS", "CHALLENGE", "PLAY"):
            return s
    return None


class PlayerMemory:
    """每位玩家私有的记忆库：自己的思考 + 对其他玩家的观察"""

    def __init__(self, owner_number: int):
        self.owner_number = owner_number
        self.thoughts: List[Dict[str, Any]] = []
        self.opponents: Dict[int, Dict[str, Any]] = {}

    def _profile(self, player_number: int) -> Dict[str, Any]:
        if player_number not in self.opponents:
            self.opponents[player_number] = {
                "player_number": player_number,
                "notes": [],
                "stats": {
                    "plays": 0,
                    "challenged": 0,
                    "lie_caught": 0,
                    "bluff_success": 0,
                    "false_challenges": 0,
                },
            }
        return self.opponents[player_number]

    def add_thought(
        self,
        round_num: int,
        situation: str,
        reasoning: str,
        decision: dict,
    ):
        self.thoughts.append({
            "round": round_num,
            "situation": situation[:300],
            "reasoning": reasoning[:THOUGHT_MAX_LEN],
            "decision": decision,
        })
        if len(self.thoughts) > MAX_THOUGHTS:
            self.thoughts = self.thoughts[-MAX_THOUGHTS:]

    def observe_opponent(self, player_number: int, note: str):
        if player_number == self.owner_number:
            return
        profile = self._profile(player_number)
        profile["notes"].append(note)
        if len(profile["notes"]) > MAX_OPPONENT_NOTES:
            profile["notes"] = profile["notes"][-MAX_OPPONENT_NOTES:]

    def bump_stat(self, player_number: int, key: str, delta: int = 1):
        if player_number == self.owner_number:
            return
        profile = self._profile(player_number)
        profile["stats"][key] = profile["stats"].get(key, 0) + delta

    def format_for_prompt(self) -> str:
        lines = []
        if self.thoughts:
            lines.append("【近期自我思考】")
            for t in self.thoughts[-3:]:
                d = t["decision"]
                act = d.get("action", "?")
                summary = act
                if act == "play":
                    summary = (
                        f"出牌 实际{d.get('cards')} 声称{d.get('claimed_count')}张{d.get('claimed')}"
                    )
                    if d.get("bluff"):
                        b = d["bluff"]
                        summary += f" 唬人:{b[:40]}{'…' if len(b) > 40 else ''}"
                lines.append(
                    f"- 第{t['round']}轮 [{summary}] "
                    f"{t['reasoning'][:THOUGHT_MAX_LEN]}"
                )
        if self.opponents:
            lines.append("【对手风格档案】")
            for num in sorted(self.opponents):
                p = self.opponents[num]
                s = p["stats"]
                style = (
                    f"出牌{s['plays']}次, 被质疑{s['challenged']}次, "
                    f"说谎被抓{s['lie_caught']}次, 唬住质疑{s['bluff_success']}次"
                )
                lines.append(f"- 玩家{num}: {style}")
                for note in p["notes"][-3:]:
                    lines.append(f"    · {note}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "owner": self.owner_number,
            "thoughts": self.thoughts,
            "opponents": self.opponents,
        }


# ===================================================================
# 第一部分： 预定义
# ===================================================================


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "revealed_card",
            "description": "获取本轮因质疑而翻开的牌；未质疑仍面朝下的牌不在此列",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "round_history",
            "description": "获取本轮出牌历史（仅声称与表演，不含牌面朝下的实际牌）",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "history",
            "description": "历史轮次公开记录（不含未质疑牌的真实牌型）",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "card_status",
            "description": (
                "获取指定玩家的手牌（只能查自己）。含本轮目标牌张数、Joker 张数及万能牌说明。"
                "player_number 与界面一致：玩家1→1，玩家2→2"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_number": {
                        "type": "integer",
                        "description": "玩家编号，1-4，与状态栏「玩家1」等显示一致"
                    }
                },
                "required": ["player_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "bullet_status",
            "description": "查看子弹槽。只能查自己；仅可知已开几枪，不可知实弹位置",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_number": {
                        "type": "integer",
                        "description": "玩家编号，1-4，与状态栏「玩家1」等显示一致"
                    }
                },
                "required": ["player_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "my_knowledge",
            "description": "读取自己的记忆库：近期思考过程 + 各对手风格习惯观察",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]

# ===================================================================
# 第二部分：主循环
# ===================================================================

PLAYER_NUM = 4

BULLET_NUM = 6

class ReActAgent:
    def __init__(self, player_index: int = -1, game=None):
        self.client = openai.Client(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
        
        self.player_index = player_index
        self.game = game  # LiarsBar 实例，用于注册工具
        self.tools = {}   # 工具函数映射
        
        # 注册游戏工具
        if game:
            self._register_tools()

        self.system_prompt = f"""
你是骗子酒馆(Liar's Bar)游戏的AI玩家。你的目标是通过虚张声势和识破对手来存活到最后。

【游戏规则】
1. 牌堆：J、Q、K 各6张，Joker(万能牌) 2张，共20张，每人发5张
2. 【Joker 万能牌】本轮目标为 J/Q/K 时，Joker **视同该目标牌**，不是假牌、更不是「冒充」某张牌：
   - 打出 Joker 并声称为本轮目标 → **诚实出牌**，质疑翻牌后你败（上家未说谎）
   - **说谎**仅指：实际打出非目标且非 Joker 的牌（如目标 J 却打出 Q 或 K）
   - 勿写「用 Joker 冒充 J」；应理解成「Joker 即当作目标牌打出」
3. 每一「轮」发一次牌，指定目标牌（J/Q/K）；轮内循环：上家出牌 → 下家质疑或放行
4. 下家 PASS：不质疑，由该下家继续出牌（对上家的下家再决定是否质疑）
5. 下家 CHALLENGE：质疑成功则上家开枪；质疑失败则下家自己开枪
6. 开枪：轮盘逐发，不知实弹位置；仅知自己已开几枪
7. 本轮结束：一旦发生质疑并开枪，本轮立即结束；若有人死亡则游戏继续到剩一人，否则重新发牌开始下一轮
8. 终局：场上仅你一人有手牌且你是下家时，必须质疑，不得 PASS

【你的工具】
- revealed_card(): 仅「被质疑后翻开」的牌，未质疑的出牌不会出现在此
- round_history(): 本轮公开记录（张数、声称、是否被质疑/放行；无未质疑牌的真实牌型）
- history(): 历史轮次公开记录（同上）
- card_status(player_number): 查看手牌，只能看自己；编号 1-4 与界面「玩家1」一致
- bullet_status(player_number): 查看子弹槽；编号 1-4 与界面一致
- my_knowledge(): 读取你的私人记忆库（自我思考 + 对手风格档案）

【你的身份】你是玩家{self.player_index + 1}。查自己信息时传 player_number={self.player_index + 1}
决策前可用 my_knowledge() 回顾历史；每次决策会写入你的思考记录

【思考策略】
1. 出牌前：分析手牌、场上已出的牌、对手剩余牌数
2. 选择：出真目标或 Joker（均算诚实）积累信誉 / 出 Q、K 等非目标假牌唬人清牌
3. 质疑时：权衡风险——质疑失败是你开枪；round_history 无牌面下的 actual，勿透视臆测
4. 强疑点可 CHALLENGE：对手有说谎前科、上家声称 2-3 张、表演浮夸；勿因「可能出了 Joker」而质疑（Joker 合法）
5. 宜 PASS：你目标+Joker 很多、上家只出 1 张且镇定、你已开过多次枪余弹不多

【输出格式 — 严格遵守】
工具调用阶段可自行分析；最终回复禁止长篇堆砌，按顺序输出：

总结: [分析后用一句话说明理由，100字以内，不要重复工具原文]

出牌时再接：
唬人: [表演，一两句]
PLAY: 打出 [实际牌], 声称: "X张[目标牌]"

质疑时：总结后一行 CHALLENGE 或 PASS（不要废话）

【重要】必须先用工具获取信息，再写总结与决策！
"""

        self.max_iterations = 10

    def _register_tools(self):
        """注册游戏工具函数"""
        self.tools = {
            "revealed_card": lambda: self.game.revealed_card if self.game else [],
            "round_history": lambda: (
                self.game.get_public_round_history() if self.game else []
            ),
            "history": lambda: (
                self.game.get_public_history() if self.game else []
            ),
            "card_status": lambda player_number: self._get_card_status(player_number),
            "bullet_status": lambda player_number: self._get_bullet_status(player_number),
            "my_knowledge": lambda: self._get_my_knowledge(),
        }

    def _get_my_knowledge(self) -> dict:
        if not self.game or self.player_index < 0:
            return {"error": "游戏未初始化"}
        memory = self.game.players[self.player_index].memory
        return {
            "summary": memory.format_for_prompt(),
            "detail": memory.to_dict(),
        }

    def _resolve_player_number(self, player_number: int) -> tuple[int | None, str | None]:
        """界面编号(1-4) → 内部索引(0-3)。兼容 LLM 误传 0-based。"""
        if 1 <= player_number <= PLAYER_NUM:
            return player_number - 1, None
        if 0 <= player_number < PLAYER_NUM:
            return player_number, (
                f"player_number 应为 1-{PLAYER_NUM}（玩家1→1），"
                f"你传了 {player_number}，已按内部索引处理"
            )
        return None, f"无效的玩家编号 {player_number}，应为 1-{PLAYER_NUM}"

    def _get_card_status(self, player_number: int):
        """获取手牌信息（只能看自己）"""
        if not self.game:
            return {"error": "游戏未初始化"}
        idx, hint = self._resolve_player_number(player_number)
        if idx is None:
            return {"error": hint}
        player = self.game.players[idx]
        if idx != self.player_index:
            return {
                "error": "不能查看其他玩家的手牌",
                "player_number": player_number,
                "your_player_number": self.player_index + 1,
            }
        target = self.game.current_target if self.game else None
        result = {
            "player_number": idx + 1,
            "cards": player.cards,
            "card_count": len(player.cards),
            "alive": player.alive,
        }
        if target:
            n_target = sum(1 for c in player.cards if c == target)
            n_joker = sum(1 for c in player.cards if c == "Joker")
            result["current_target"] = target
            result["count_as_target"] = n_target + n_joker
            result["true_target_count"] = n_target
            result["joker_count"] = n_joker
            result["joker_rule"] = (
                f"Joker 视同本轮目标 {target}；打出 Joker 并声称 {target} 为诚实出牌，"
                f"质疑翻牌不会判上家说谎。"
            )
        if hint:
            result["hint"] = hint
        return result

    def _get_bullet_status(self, player_number: int):
        """子弹槽：仅本人可查；只暴露已开枪次数，不暴露实弹位置"""
        if not self.game:
            return {"error": "游戏未初始化"}
        idx, hint = self._resolve_player_number(player_number)
        if idx is None:
            return {"error": hint}
        player = self.game.players[idx]

        if idx != self.player_index:
            return {
                "error": "只能查看自己的子弹槽",
                "player_number": player_number,
                "your_player_number": self.player_index + 1,
            }

        slots = []
        for b in range(BULLET_NUM):
            if b < player.current:
                slots.append("已击发(○)")
            else:
                slots.append("未击发(●)")

        result = {
            "player_number": idx + 1,
            "slots_visual": slots,
            "shots_fired": player.current,
            "remaining_chambers": BULLET_NUM - player.current,
            "alive": player.alive,
            "description": (
                f"已开枪 {player.current} 次，"
                f"剩余 {BULLET_NUM - player.current} 格未击发（实弹位置未知）"
            ),
        }
        if hint:
            result["hint"] = hint
        return result

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行工具并返回 JSON 字符串"""
        if tool_name not in self.tools:
            return json.dumps({"error": f"工具 {tool_name} 不存在"}, ensure_ascii=False)

        try:
            fn = self.tools[tool_name]
            result = fn(**tool_args) if tool_args else fn()
            if not isinstance(result, (dict, list)):
                result = {"result": result}
            return json.dumps(result, ensure_ascii=False)
        except TypeError as e:
            return json.dumps({"error": f"参数错误: {e}"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def run(self, user_input: str) -> str:
        """Function Calling 循环，直到模型给出最终回答"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        for iteration in range(1, self.max_iterations + 1):
            print(f"🔄 第 {iteration} 轮对话")

            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 无工具调用 → 直接返回文本
            if not message.tool_calls:
                content = message.content or ""
                print("✅ LLM 生成最终回答")
                return content

            print(f"🔧 LLM 请求调用 {len(message.tool_calls)} 个工具")

            # 助手消息（含 tool_calls，需序列化为 dict）
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    tool_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"   调用 {tool_name}({tool_args})")
                result_str = self._execute_tool(tool_name, tool_args)
                print(f"   结果：{_format_log_text(result_str)}")

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": tool_name,
                    "content": result_str,
                })

        return "达到最大迭代次数，无法完成请求。"


class Player:
    def __init__(self, index: int, name: str, game=None):
        self.index = index
        self.name = name
        self.cards = []
        self.game = game  # LiarsBar 实例
        self.agent = ReActAgent(player_index=index, game=game)
        self.memory = PlayerMemory(owner_number=index + 1)
        self.bluff_persona = BLUFF_PERSONAS[index % len(BLUFF_PERSONAS)]
        self.bullet = random.randint(0, BULLET_NUM - 1)  # 0-5，对应6个子弹槽位置
        self.current = 0
        self.alive = True

    def get_decision(
        self,
        current_target_card: str,
        last_action=None,
        phase: str = "play",
        must_challenge: bool = False,
    ):
        """获取玩家决策。phase: play=须出牌, react=质疑或跳过"""
        if phase == "play":
            phase_desc = "出牌"
        elif must_challenge:
            phase_desc = "下家质疑（终局：你必须 CHALLENGE 上家，禁止 PASS）"
        else:
            phase_desc = "下家质疑（你是上家的下家，可 CHALLENGE 或 PASS）"

        context = f"""
当前游戏状态：
- 你是玩家{self.index + 1} ({self.name})，工具参数 player_number 填 {self.index + 1}
- 本轮目标牌：{current_target_card}
- 当前阶段：{phase_desc}
- 手牌张数：{len(self.cards)}
- 可当目标出的牌：真{current_target_card} {sum(1 for c in self.cards if c == current_target_card)} 张 + Joker {sum(1 for c in self.cards if c == "Joker")} 张（Joker=目标，非冒充）
- 子弹槽：已开枪 {self.current} 次（实弹位置未知，共 {BULLET_NUM} 格）
"""
        if last_action:
            context += f"\n待回应行动：{last_action}\n"

        memory_text = self.memory.format_for_prompt()
        if memory_text:
            context += f"\n【你的记忆库摘要】\n{memory_text}\n"

        output_fmt = (
            "\n【回复格式】先写「总结:」（≤100字，一句话理由），再写决策行；"
            "勿输出长段分析过程。\n"
        )

        if phase == "play":
            style_hint = _bluff_style_hint(self.index, self.bluff_persona)
            context += f"\n【本手唬人风格提示】{style_hint}\n"
            actions = (
                "请出牌：总结 → 唬人 → PLAY。\n"
                "PLAY: 打出 [实际牌], 声称: \"X张目标牌\""
            )
        elif must_challenge:
            actions = "必须质疑：总结（≤100字）→ CHALLENGE"
        else:
            if self.game and self.game.pending_play:
                ch_hint = self.game.build_challenge_hint(
                    self.index, self.game.pending_play
                )
                context += f"\n{ch_hint}\n"
            actions = "请质疑：总结（≤100字）→ CHALLENGE 或 PASS"

        prompt = context + output_fmt + f"{actions}\n"
        response = self.agent.run(prompt)
        decision = self._parse_decision(
            response, current_target_card, phase=phase
        )
        summary = _extract_thought(response)
        if summary:
            print(f"   💭 {summary}")
        self._record_thought(prompt, decision, response)
        return decision

    def _record_thought(self, situation: str, decision: dict, response: str):
        round_num = self.game.round_num if self.game else 0
        self.memory.add_thought(
            round_num, situation, _extract_thought(response), decision
        )

    def pick_cards_for_play(self, decision: dict) -> List[str]:
        """从决策中取出可出的手牌（校验、去重、最多 3 张）"""
        hand = list(self.cards)
        picked: List[str] = []
        for c in decision.get("cards", []):
            if len(picked) >= 3:
                break
            if c in hand:
                picked.append(c)
                hand.remove(c)
        if picked:
            return picked
        if self.cards:
            return [self.cards[0]]
        return []
    
    def _parse_decision(
        self, response: str, target_card: str = None, phase: str = "play"
    ) -> dict:
        """解析 AI 决策；react 阶段仅允许 challenge / pass"""
        cmd = _find_final_command(response) or ""

        if phase == "react":
            if re.match(r"^CHALLENGE(?:\s*[:：]|$)", cmd, re.I):
                target_m = re.search(r"CHALLENGE\s*:\s*玩家\s*(\d+)", cmd, re.I)
                result = {"action": "challenge"}
                if target_m:
                    result["target_player_number"] = int(target_m.group(1))
                return result
            if re.match(r"^PLAY\s*[:：]", cmd, re.I) or "打出" in cmd:
                print("  ⚠️ 质疑阶段不应出牌，已视为 PASS")
            return {"action": "pass"}

        if re.match(r"^CHALLENGE(?:\s*[:：]|$)", cmd, re.I):
            target_m = re.search(r"CHALLENGE\s*:\s*玩家\s*(\d+)", cmd, re.I)
            result = {"action": "challenge"}
            if target_m:
                result["target_player_number"] = int(target_m.group(1))
            return result

        play_text = cmd if re.match(r"^PLAY\s*[:：]", cmd, re.I) else ""
        if play_text or "打出" in response:
            claimed_count = None
            claimed = target_card

            claimed_m = re.search(
                r"声称\s*[：:\"]?\s*(\d+)\s*张?\s*([JQK])",
                play_text or response,
                re.I,
            )
            if claimed_m:
                claimed_count = int(claimed_m.group(1))
                claimed = claimed_m.group(2).upper()

            cards: List[str] = []
            play_m = re.search(
                r"打出\s*(\[[^\]]+\]|[^,，声称唬人表演]+?)"
                r"(?=\s*[,，]?\s*声称|声称|唬人|表演|行为|$)",
                play_text or response,
                re.I,
            )
            if play_m:
                cards = _parse_card_list(play_m.group(1))[:3]

            if claimed_count is None:
                claimed_count = len(cards) if cards else 1
            claimed_count = min(max(claimed_count, 1), 3)

            claimed = claimed or target_card or "?"
            bluff = _parse_bluff_behavior(response)
            if not bluff:
                bluff = _default_bluff_behavior(claimed_count, claimed)

            return {
                "action": "play",
                "cards": cards,
                "claimed": claimed,
                "claimed_count": claimed_count,
                "bluff": bluff,
            }

        return {"action": "pass"}

    def shoot(self):
        if self.current == self.bullet:
            self.alive = False
        else:
            self.current += 1

class Deck:
    def __init__(self):
        self.cards = []
        self.cards.extend(["J"] * 6)
        self.cards.extend(["Q"] * 6)
        self.cards.extend(["K"] * 6)
        self.cards.extend(["Joker"] * 2)
        random.shuffle(self.cards)
    
    def distribute(self, players: List[Player]):
        random.shuffle(self.cards)
        alive = [pl for pl in players if pl.alive]
        need = len(alive) * 5
        if need > len(self.cards):
            raise ValueError(
                f"牌堆不足：需 {need} 张，牌堆仅 {len(self.cards)} 张"
            )
        card_index = 0
        for player in alive:
            player.cards = self.cards[card_index : card_index + 5]
            card_index += 5

class LiarsBar:
    TARGET_CARDS = ["J", "Q", "K"]
    
    def __init__(self):
        self.players = []  # 延迟初始化，等self准备好
        self.deck = Deck()
        self.history = []  # 所有轮次历史
        self.round_history = []  # 当前轮次出牌记录
        self.revealed_card = []  # 当前轮次已翻开的牌
        self.current_target = None  # 当前轮目标牌
        self.current_player_idx = 0  # 当前行动玩家
        self.round_num = 0  # 轮次数
        self.pending_play = None  # 待质疑的出牌
        self.round_active = False  # 本轮是否进行中
        self._lap_played: set[int] = set()  # 本轮当前一圈已出过牌的玩家
        self._mid_round_status_shown = False  # 本轮是否已打过「全员出过牌」快照

    def init_players(self):
        """初始化玩家（在self准备好后调用）"""
        self.players = [Player(i, f"玩家{i + 1}", game=self) for i in range(PLAYER_NUM)]

    def start(self):
        """开始游戏"""
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise RuntimeError("请先设置环境变量 DEEPSEEK_API_KEY")
        self.init_players()
        print("🎮 骗子酒馆游戏开始！")
        print(f"玩家人数：{PLAYER_NUM}")
        print(f"目标牌序列：{' → '.join(self.TARGET_CARDS)} 循环")
        print("一轮 = 发牌后循环「出牌→下家质疑/放行」，直至发生质疑开枪；然后重新发牌")

    def new_round(self):
        """新一轮：重新发牌，重置轮内状态"""
        self.round_num += 1
        self.current_target = self.TARGET_CARDS[(self.round_num - 1) % 3]
        self.round_history = []
        self.revealed_card = []
        self.pending_play = None
        self.round_active = True
        self._lap_played = set()
        self._mid_round_status_shown = False

        alive_players = [p for p in self.players if p.alive]
        if len(alive_players) <= 1:
            return False

        self.deck = Deck()
        self.deck.distribute(self.players)
        self.current_player_idx = next(i for i, p in enumerate(self.players) if p.alive)
        print(f"\n🎯 第 {self.round_num} 轮开始！目标牌：{self.current_target}")
        print(f"   先手：玩家{self.current_player_idx + 1}")
        return True

    def _finish_round(self, reason: str):
        """质疑开枪后结束本轮，存档"""
        self.round_active = False
        self.pending_play = None
        self.history.append({
            "round": self.round_num,
            "target": self.current_target,
            "reason": reason,
            "events": self.get_public_round_history(),
        })
        print(f"\n📋 第 {self.round_num} 轮结束：{reason}")

    def get_alive_players(self):
        """获取存活玩家列表"""
        return [p for p in self.players if p.alive]

    def _notify_observers(
        self,
        about_player_number: int,
        note: str,
        exclude_idx: int | None = None,
        stat_updates: dict | None = None,
    ):
        """向其他存活玩家写入对某人的公开观察"""
        stat_updates = stat_updates or {}
        for i, p in enumerate(self.players):
            if not p.alive or i == exclude_idx:
                continue
            p.memory.observe_opponent(about_player_number, note)
            for key, delta in stat_updates.items():
                p.memory.bump_stat(about_player_number, key, delta)

    def get_next_player_idx(self, start_idx=None):
        """获取下一个存活玩家的索引"""
        if start_idx is None:
            start_idx = self.current_player_idx
        
        idx = (start_idx + 1) % PLAYER_NUM
        while not self.players[idx].alive and idx != start_idx:
            idx = (idx + 1) % PLAYER_NUM
        return idx if self.players[idx].alive else None

    @staticmethod
    def _public_play_view(action: dict) -> dict:
        """其他玩家可见的出牌信息（不含未质疑时的真实牌型）"""
        if action.get("action") != "play":
            return dict(action)

        view = {
            "player": action["player"],
            "player_number": action["player"] + 1,
            "action": "play",
            "card_count": action.get("claimed_count", action.get("count")),
            "claimed": action["claimed"],
            "bluff": action.get("bluff"),
        }
        if action.get("challenged"):
            view["status"] = "被质疑已翻开"
            if action.get("revealed_actual") is not None:
                view["revealed_actual"] = action["revealed_actual"]
        elif action.get("passed_unchallenged"):
            view["status"] = "下家放行未质疑"
            view["note"] = "真实牌型未知（仅出牌者自知）"
        else:
            view["status"] = "待下家回应"
            view["note"] = "真实牌型未知"
        return view

    def _mark_last_play_passed(self):
        for action in reversed(self.round_history):
            if action.get("action") == "play":
                action["passed_unchallenged"] = True
                return action
        return None

    def _mark_last_play_challenged(self, actual_cards: List[str]):
        for action in reversed(self.round_history):
            if action.get("action") == "play":
                action["challenged"] = True
                action["revealed_actual"] = list(actual_cards)
                action["passed_unchallenged"] = False
                return action
        return None

    def get_public_round_history(self) -> list:
        return [self._public_play_view(a) for a in self.round_history]

    def get_public_history(self) -> list:
        return [
            {
                "round": rec["round"],
                "target": rec.get("target"),
                "reason": rec.get("reason"),
                "events": [
                    self._public_play_view(e)
                    if e.get("action") == "play"
                    else e
                    for e in rec.get("events", [])
                ],
            }
            for rec in self.history
        ]

    def build_challenge_hint(self, challenger_idx: int, pending_play: dict) -> str:
        """给下家的质疑权衡提示（不含 actual，避免无脑质疑）"""
        who = pending_play["player"]
        pn = who + 1
        challenger = self.players[challenger_idx]
        target = self.current_target
        good = sum(
            1 for c in challenger.cards if c == target or c == "Joker"
        )
        remaining = BULLET_NUM - challenger.current
        cc = pending_play.get("claimed_count", 1)

        lines = [
            "【质疑权衡】质疑失败由你开枪。牌面朝下无法确知 actual，勿凭空断言。",
            "强疑点再 CHALLENGE；证据一般或你余弹不多时宜 PASS。",
        ]

        if remaining <= 2:
            lines.append(
                f"- 你仅剩 {remaining} 格未击发，质疑失败极易暴毙，非铁证建议 PASS。"
            )
        elif challenger.current >= 2:
            lines.append(
                f"- 你已开枪 {challenger.current} 次，质疑须更谨慎。"
            )

        opp = challenger.memory.opponents.get(pn, {})
        stats = opp.get("stats", {})
        suspicion = 0

        if stats.get("lie_caught", 0) > 0:
            suspicion += 2
            lines.append(f"- 玩家{pn} 有说谎前科，疑点+。")
        if stats.get("plays", 0) >= 3 and stats.get("challenged", 0) == 0:
            suspicion += 1
            lines.append(f"- 玩家{pn} 多次出牌少被质疑，略可疑。")

        if cc >= 3:
            suspicion += 2
            lines.append(f"- 上家声称 {cc} 张 {target}，偏激进，疑点+。")
        elif cc == 2:
            suspicion += 1

        if good <= 1:
            suspicion += 1
            lines.append(
                f"- 你手中可当 {target} 的牌很少（真{target}+Joker 共 {good} 张），"
                f"上家大额声称时牌池余量偏紧（注：上家出 Joker 声称 {target} 仍算诚实）。"
            )
        elif good >= 4 and cc == 1:
            lines.append(
                f"- 你手握较多 {target}（{good} 张）且上家只出 1 张，可先 PASS。"
            )

        if suspicion >= 3:
            lines.append("- 综合：疑点偏多，可考虑 CHALLENGE。")
        elif suspicion <= 0 and good >= 3:
            lines.append("- 综合：疑点较少，倾向 PASS。")
        else:
            lines.append("- 综合：疑点中等，CHALLENGE / PASS 均可，慎判。")

        return "\n".join(lines)

    def get_last_play_internal(self):
        """内部用：含 actual，仅质疑结算"""
        if self.pending_play:
            return self.pending_play
        return None

    def format_play_action(self, action: dict) -> str:
        """待质疑提示：仅公开信息，不含真实牌型"""
        pn = action["player"] + 1
        cc = action.get("claimed_count", action.get("count"))
        line = f"玩家{pn} 打出 {cc} 张（声称均为 {action['claimed']}），牌面朝下"
        bluff = action.get("bluff")
        if bluff:
            line += f"。表演：{bluff}"
        return line

    def players_holding_cards(self) -> List[int]:
        """仍有手牌的存活玩家索引"""
        return [i for i, p in enumerate(self.players) if p.alive and p.cards]

    def get_challenger_idx(self) -> int | None:
        """下家：出牌者的下一位存活玩家（唯一可质疑者）"""
        if not self.pending_play:
            return None
        return self.get_next_player_idx(self.pending_play["player"])

    def is_challenger(self, player_idx: int) -> bool:
        return player_idx == self.get_challenger_idx()

    def must_challenge(self, player_idx: int) -> bool:
        """终局：仅下家且场上只剩该玩家有手牌时，必须质疑"""
        if not self.is_challenger(player_idx):
            return False
        holders = self.players_holding_cards()
        return len(holders) == 1 and holders[0] == player_idx

    def get_winner(self):
        alive = [p for p in self.players if p.alive]
        if len(alive) == 1:
            return alive[0]
        return None

    def _set_turn(self, idx: int):
        self.current_player_idx = idx

    def _record_play_completed(self, player_idx: int):
        """存活玩家各出过一轮牌后打印一次状态（不含质疑/放行）"""
        if not self.round_active or self._mid_round_status_shown:
            return
        alive = {i for i, p in enumerate(self.players) if p.alive}
        if not alive:
            return
        self._lap_played.add(player_idx)
        if self._lap_played >= alive:
            print("\n📊 当前状态（全员已出过牌）")
            self.player_status()
            self._mid_round_status_shown = True
            self._lap_played = set()

    def _advance_turn(self, from_idx: int):
        nxt = self.get_next_player_idx(from_idx)
        if nxt is not None:
            self._set_turn(nxt)

    def _clear_pending(self):
        self.pending_play = None

    def apply_decision(self, player_idx: int, decision: dict, phase: str) -> bool:
        """执行决策。返回 False 表示整局游戏结束"""
        player = self.players[player_idx]
        pn = player_idx + 1

        if phase == "react":
            if not self.is_challenger(player_idx):
                print(f"  ⚠️ 仅下家可质疑，玩家{pn} 无权表态")
                return True

            if self.must_challenge(player_idx) and decision.get("action") != "challenge":
                print(f"  ⚠️ 玩家{pn} 场上仅剩你尚有手牌，规则要求必须质疑上家")
                decision = {"action": "challenge"}

            if decision.get("action") == "challenge":
                print(f"\n⚔️  玩家{pn} 质疑！")
                success, loser_idx = self.challenge(player_idx, -1)
                target_p = (
                    self.pending_play["player"] if self.pending_play else -1
                )
                self.round_history.append({
                    "action": "challenge",
                    "challenger": player_idx,
                    "target": target_p,
                    "success": success,
                })
                died = self.shoot(loser_idx)
                if self.get_winner():
                    self._finish_round(
                        f"玩家{loser_idx + 1} 死亡，胜者已产生"
                    )
                    return False
                if died:
                    self._finish_round(
                        f"玩家{loser_idx + 1} 中弹死亡，重新发牌"
                    )
                else:
                    self._finish_round("质疑结算（空弹），重新发牌")
                return True

            who_played = self.pending_play["player"]
            passed = self._mark_last_play_passed()
            print(f"  ⏭️  下家玩家{pn} 选择不质疑")
            print(f"  ✓ 玩家{who_played + 1} 的出牌成立（真实牌型仍保密），轮到玩家{pn} 出牌")
            if passed:
                cc = passed.get("claimed_count", passed.get("count"))
                self._notify_observers(
                    who_played + 1,
                    f"第{self.round_num}轮：玩家{who_played + 1} 出 {cc} 张（声称{passed['claimed']}），"
                    f"玩家{pn} 放行未质疑，真实牌型未知",
                )
            self._clear_pending()
            self._set_turn(player_idx)
            return True

        # phase == play
        if decision.get("action") != "play":
            print(f"  ⚠️ 玩家{pn} 未出牌，强制出 1 张")
            decision = {"action": "play", "cards": [player.cards[0]], "claimed": self.current_target}

        cards = player.pick_cards_for_play(decision)
        if not cards:
            print(f"  ⚠️ 玩家{pn} 无牌可出，跳过")
            self._advance_turn(player_idx)
            return True

        claimed = decision.get("claimed", self.current_target)
        claimed_count = decision.get("claimed_count", len(cards))
        if claimed_count != len(cards):
            claimed_count = len(cards)

        bluff = decision.get("bluff") or _default_bluff_behavior(claimed_count, claimed)
        action = self.play_card(
            player_idx, cards, claimed, claimed_count, bluff_behavior=bluff
        )
        self.pending_play = action
        challenger = self.get_challenger_idx()
        if challenger is not None:
            self._set_turn(challenger)
        self._record_play_completed(player_idx)
        return True

    def process_turn(self) -> bool:
        """处理当前玩家一回合。返回 False 表示游戏结束"""
        if self.get_winner():
            return False

        idx = self.current_player_idx
        player = self.players[idx]

        if not player.alive:
            self._advance_turn(idx)
            return True

        if self.pending_play:
            challenger = self.get_challenger_idx()
            if idx == self.pending_play["player"]:
                if challenger is not None:
                    self._set_turn(challenger)
                return True
            if idx != challenger:
                self._advance_turn(idx)
                return True
            phase = "react"
            last_action = self.format_play_action(self.pending_play)
        else:
            if not player.cards:
                self._advance_turn(idx)
                return True
            phase = "play"
            last_action = None

        forced = self.must_challenge(idx)
        tag = f"{phase}, 必须质疑" if forced else phase
        print(f"\n🤖 玩家{idx + 1} 思考中... ({tag})")
        decision = player.get_decision(
            self.current_target,
            last_action=last_action,
            phase=phase,
            must_challenge=forced,
        )
        print(f"   决策：{decision}")
        return self.apply_decision(idx, decision, phase)

    def run_round(self) -> bool:
        """本轮：循环「出牌→下家质疑/放行」，直至质疑开枪结束。返回 False 表示整局结束"""
        guard = 0
        while self.round_active and guard < 500:
            guard += 1
            if self.get_winner():
                return False
            if not self.process_turn():
                return False
        if guard >= 500:
            print("⚠️ 本轮步数超限，强制结束")
            self._finish_round("步数超限")
        return True

    def run_game(self, max_rounds: int = 20):
        """运行完整对局"""
        while self.round_num < max_rounds:
            if not self.new_round():
                break
            self.player_status()
            if not self.run_round():
                break
            self.player_status()

        winner = self.get_winner()
        if winner:
            print(f"\n🏆 游戏结束！胜者：{winner.name}（玩家{winner.index + 1}）")
        else:
            alive = [p for p in self.players if p.alive]
            print(f"\n⏹️ 游戏停止。存活 {len(alive)} 人")

    def play_card(
        self,
        player_idx: int,
        cards: List[str],
        claimed: str,
        claimed_count: int | None = None,
        bluff_behavior: str | None = None,
    ):
        """玩家出牌

        Args:
            player_idx: 玩家索引
            cards: 实际打出的牌（牌面朝下，质疑前不公开）
            claimed: 对外声称的牌面（通常为当前目标牌，可与实际不同）
            claimed_count: 声称张数，默认等于实际打出张数
            bluff_behavior: 唬人表演（动作、语气，公开可见）
        """
        player = self.players[player_idx]
        if not cards or len(cards) > 3:
            raise ValueError("每次须打出 1-3 张牌")
        for c in cards:
            if c not in player.cards:
                raise ValueError(f"手牌中没有 {c}")

        claimed_count = claimed_count if claimed_count is not None else len(cards)

        for card in cards:
            if card in player.cards:
                player.cards.remove(card)

        bluff = (bluff_behavior or "").strip() or _default_bluff_behavior(
            claimed_count, claimed
        )

        public_record = {
            "player": player_idx,
            "action": "play",
            "count": len(cards),
            "claimed_count": claimed_count,
            "claimed": claimed,
            "bluff": bluff,
            "passed_unchallenged": False,
            "challenged": False,
        }
        self.round_history.append(public_record)
        action = {**public_record, "actual": list(cards)}

        pn = player_idx + 1
        print(f"  🎭 玩家{pn} {bluff}")
        actual_str = _format_cards_display(cards)
        print(
            f"  🎴 玩家{pn} 声称 {claimed_count} 张 {claimed} | "
            f"实际 [{actual_str}]（{len(cards)}张，牌面朝下）[日志可见]"
        )
        self._notify_observers(
            pn,
            f"第{self.round_num}轮：玩家{pn} 声称 {claimed_count} 张 {claimed}；"
            f"表演：{bluff}",
            exclude_idx=player_idx,
            stat_updates={"plays": 1},
        )
        return action

    def challenge(self, challenger_idx: int, target_idx: int):
        """质疑
        
        Returns:
            (挑战结果, 输家索引) - 结果True表示质疑成功（目标说谎）
        """
        last_play = self.pending_play or self.get_last_play_internal()
        if not last_play:
            return False, challenger_idx

        target_idx = last_play["player"]
        actual_cards = last_play.get("actual")
        if actual_cards is None:
            return False, challenger_idx
        claimed = last_play["claimed"]

        self._mark_last_play_challenged(actual_cards)
        
        # 判断是否说谎（实际牌不是声称的牌，且不是Joker）
        is_lie = any(c != claimed and c != "Joker" for c in actual_cards)
        
        target_pn = target_idx + 1
        challenger_pn = challenger_idx + 1
        self.revealed_card.extend(actual_cards)

        if is_lie:
            print(f"  ✅ 质疑成功！玩家{target_pn} 实际打出：{actual_cards}")
            self._notify_observers(
                target_pn,
                f"第{self.round_num}轮：玩家{target_pn} 被玩家{challenger_pn} 质疑成功，"
                f"实际 {actual_cards}（说谎）",
                stat_updates={"challenged": 1, "lie_caught": 1},
            )
            return True, target_idx

        print(f"  ❌ 质疑失败！玩家{target_pn} 实际打出：{actual_cards}")
        self._notify_observers(
            target_pn,
            f"第{self.round_num}轮：玩家{target_pn} 被玩家{challenger_pn} 质疑失败，"
            f"翻开 {actual_cards}（确为真牌）",
            stat_updates={"challenged": 1, "bluff_success": 1},
        )
        for i, p in enumerate(self.players):
            if p.alive and i != target_idx:
                p.memory.bump_stat(challenger_pn, "false_challenges", 1)
        return False, challenger_idx

    def shoot(self, player_idx: int):
        """玩家开枪"""
        player = self.players[player_idx]
        result = player.shoot()
        
        if player.alive:
            print(
                f"  🔫 玩家{player_idx + 1} 开枪：咔！（空弹）"
                f"已开 {player.current}/{BULLET_NUM} 枪"
                f"（实弹在格 {player.bullet + 1}）"
            )
        else:
            print(
                f"  💀 玩家{player_idx + 1} 开枪：砰！中弹死亡"
                f"（第 {player.current + 1} 发，实弹在格 {player.bullet + 1}）"
            )
        
        return not player.alive  # 返回是否死亡


    def player_status(self):
        # 启用 Windows ANSI 颜色支持
        os.system("")
        
        print("=" * 65)
        print(
            f"{'玩家':<6} {'手牌':<30} "
            f"{'子弹槽 (○已开 ●未开 红/★=实弹 ▲下一发) [日志可见]'}"
        )
        print("=" * 65)
        
        for idx, player in enumerate(self.players):
            if not player.alive:
                print(f"玩家{idx + 1}     已死亡 💀")
            else:
                hand_display = []
                for card in player.cards:
                    if card == "Joker":
                        hand_display.append("X")
                    else:
                        hand_display.append(card)
                hand_str = " ".join(hand_display)

                bullets_plain, bullets_color = _bullet_slots_display(player)
                tail = f"  已开{player.current}枪 实弹@格{player.bullet + 1}"
                print(
                    f"玩家{idx + 1}     {hand_str:<30} {bullets_color}{tail}",
                    log_plain=(
                        f"玩家{idx + 1}     {hand_str:<30} {bullets_plain}{tail}"
                    ),
                )
        
        print("=" * 65)

liars_bar = LiarsBar()
# ===================================================================
# 第三部分： 工具函数
# ===================================================================




# ===================================================================
# 第四部分： 调试
# ===================================================================
if __name__ == "__main__":
    log_path = setup_game_logging()
    _ORIGINAL_PRINT(f"📝 日志文件: {log_path}")
    game = LiarsBar()
    game.start()
    game.run_game(max_rounds=10)
    close_game_logging()