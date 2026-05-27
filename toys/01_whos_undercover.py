#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
谁是卧底 - AI 对战版
====================

4名 AI 玩家参与的"谁是卧底"游戏，使用 DeepSeek API 进行推理和决策。

游戏规则：
- 4名玩家中1人是卧底（拿到假词），3人是平民（拿到真词）
- 每轮玩家轮流描述自己的词（不能直接说出词）
- 描述后投票淘汰一名玩家
- 卧底被淘汰则平民获胜，否则卧底获胜

特点：
- 完整的 AI 思考过程展示
- Function Calling 投票机制
- 结构化游戏历史记录
"""

from random import shuffle
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import openai
import os
import json


# =====================================================
# 配置常量
# =====================================================

PLAYER_NUM = 4
API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"


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
    undercover_index: int
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
    
    def __init__(self, index: int, client: Optional[openai.OpenAI]):
        """
        初始化玩家
        
        Args:
            index: 玩家编号（0-3）
            client: OpenAI 客户端实例
        """
        self.index = index
        self.word: Optional[str] = None
        self.is_alive = True
        self.think_history: List[str] = []  # 思考历史
        self.description_history: List[str] = []  # 发言历史
        self.client = client
        self.suspicion_level = 0  # 怀疑度（被投票次数）
    
    def set_word(self, word: str) -> None:
        """设置玩家的词"""
        self.word = word
    
    def _build_description_prompt(self, game_history: List[Dict], alive_players: List[int]) -> str:
        """
        构建描述阶段的系统提示词
        
        Args:
            game_history: 游戏历史记录
            alive_players: 存活玩家列表
            
        Returns:
            系统提示词字符串
        """
        history_str = json.dumps(game_history[-5:], ensure_ascii=False, indent=2) if game_history else "游戏刚开始"
        
        return f"""你是"谁是卧底"游戏中的玩家{self.index}号。

【游戏规则】
- 4名玩家中有3名平民（拿到相同的真词），1名卧底（拿到相似的假词）
- 每轮玩家轮流描述自己的词，不能直接说出词本身
- 描述后投票淘汰一名玩家
- 卧底被淘汰则平民获胜，否则卧底获胜

【当前状态】
- 你的词："{self.word}"
- 存活玩家：{alive_players}
- 你是玩家：{self.index}号
- 游戏历史：{history_str}
- 你过往的思考：{self.think_history[-3:] if self.think_history else "暂无"}

【任务要求】
1. 先分析：根据历史发言判断自己是不是卧底
2. 再描述：用一句话描述这个词的特征（不能直接说出词）
3. 卧底策略：如果怀疑自己是卧底，要顺着平民的话说，但不能完全相同
4. 平民策略：准确描述，帮助其他平民识别卧底

【输出格式】
严格按照以下格式输出：
<思考推理过程>######<描述>

【格式说明】
- 用"######"作为分隔符
- 思考部分：简洁总结你的推理逻辑（50字以内）
- 描述部分：一句话描述词的特征，不要直接说出词

【正确示例】
前面三人都说圆形水果，我可能卧底######这是一种常见的水果，红色或绿色，口感脆甜多汁

【错误示例】
我的词是苹果######我喜欢吃苹果（错误：不能直接说出词）
大家说的都差不多######这是一种水果（错误：描述太模糊）"""

    def make_description(self, game_history: List[Dict], alive_players: List[int]) -> str:
        """
        生成词语描述
        
        Args:
            game_history: 游戏历史记录
            alive_players: 存活玩家列表
            
        Returns:
            生成的描述文本
        """
        if not self.client:
            return "[系统错误：无法连接到AI]"
        
        if not self.is_alive:
            return "[已出局]"
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self._build_description_prompt(game_history, alive_players)},
                    {"role": "user", "content": "请生成你的描述"}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析思考和描述
            if "######" in content:
                parts = content.split("######", 1)
                thinking = parts[0].strip()
                description = parts[1].strip()
            else:
                # 如果没有分隔符，尝试智能分割
                lines = content.split('\n')
                if len(lines) >= 2:
                    thinking = lines[0].strip()
                    description = ' '.join(lines[1:]).strip()
                else:
                    thinking = "思考过程未正确格式化"
                    description = content
            
            # 保存思考历史
            self.think_history.append(thinking)
            self.description_history.append(description)
            
            # 显示思考过程（调试用）
            print(f"    💭 思考：{thinking[:80]}{'...' if len(thinking) > 80 else ''}")
            
            return description
            
        except Exception as e:
            print(f"    ❌ 生成描述时出错: {e}")
            # 返回一个通用的安全描述
            return "这是一种常见的物品，人们经常使用"

    def _build_vote_prompt(self, game_history: List[Dict], alive_players: List[int]) -> str:
        """
        构建投票阶段的系统提示词
        
        Args:
            game_history: 游戏历史记录
            alive_players: 存活玩家列表
            
        Returns:
            系统提示词字符串
        """
        history_str = json.dumps(game_history[-5:], ensure_ascii=False, indent=2)
        
        return f"""你是"谁是卧底"游戏中的玩家{self.index}号，现在需要投票淘汰一名玩家。

【游戏规则】
- 4名玩家中有3名平民，1名卧底
- 投票淘汰得票最多的玩家
- 卧底被淘汰则平民获胜

【当前状态】
- 你的词："{self.word}"
- 存活玩家：{alive_players}
- 你是玩家：{self.index}号（不能投给自己）
- 游戏历史：{history_str}
- 你过往的思考：{self.think_history[-3:] if self.think_history else "暂无"}

【投票策略】
1. 分析其他玩家的描述，找出与你词义不符的人
2. 注意：卧底会尽量模仿平民的描述，但可能有细微差别
3. 如果多人可疑，选择最可疑的一个
4. 绝对不能投给自己！

【可用工具】
你可以使用 vote 工具进行投票。

【输出要求】
先说明你怀疑谁以及理由，然后调用 vote 工具。"""

    def make_vote(self, game_history: List[Dict], alive_players: List[int], 
                  tools_schema: List[Dict], tools_map: Dict[str, Any], game: 'Game') -> Optional[int]:
        """
        进行投票
        
        Args:
            game_history: 游戏历史记录
            alive_players: 存活玩家列表
            tools_schema: 工具 schema
            tools_map: 工具函数映射
            game: 游戏实例
            
        Returns:
            投票目标的玩家编号，失败则返回 None
        """
        if not self.client:
            print(f"    ❌ 玩家{self.index}: API 未配置，跳过投票")
            return None
        
        if not self.is_alive:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": self._build_vote_prompt(game_history, alive_players)},
                    {"role": "user", "content": "请分析并投票"}
                ],
                tools=tools_schema,
                temperature=0.7,
                max_tokens=500
            )
            
            # 获取思考过程
            think_process = response.choices[0].message.content or "未输出思考过程"
            print(f"    💭 思考：{think_process[:80]}{'...' if len(think_process) > 80 else ''}")
            
            self.think_history.append(think_process)
            
            # 处理工具调用
            tool_calls = response.choices[0].message.tool_calls
            
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    if tool_name not in tools_map:
                        print(f"    ⚠️ 无效的工具名: {tool_name}")
                        continue
                    
                    # 注入必要的上下文参数
                    tool_args['game'] = game
                    tool_args['voter_index'] = self.index
                    
                    # 执行投票
                    result = tools_map[tool_name](**tool_args)
                    
                    if result and 'target' in tool_args:
                        return tool_args['target']
            else:
                print(f"    ⚠️ 玩家{self.index}没有使用投票工具")
                
        except Exception as e:
            print(f"    ❌ 玩家{self.index}投票时出错: {e}")
        
        return None


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

【要求】
1. 两个词必须是同一类别（如都是水果、都是动物）
2. 词义相近但不同，让卧底有发挥空间，但平民能识别差异
3. 不要使用以下已用过的词对：{used_pairs if self.word_pair_history else "无"}

【优秀示例】
- 苹果,香蕉（都是常见水果）
- 猫,狗（都是宠物）
- 火车,地铁（都是交通工具）
- 衬衫,T恤（都是上衣）

【输出格式】
直接输出：真词,假词
例如：苹果,香蕉"""

    def generate_word(self, player_num: int) -> List[str]:
        """
        生成游戏词语并分配给玩家
        
        Args:
            player_num: 玩家数量
            
        Returns:
            分配给每个玩家的词列表
        """
        if not self.client:
            print("⚠️ 警告: 使用默认词对（API未配置）")
            return self._get_default_words(player_num)
        
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
                    self.true_word = words[0]
                    self.fake_word = words[1]
                    
                    # 记录词对
                    self.word_pair_history.append((self.true_word, self.fake_word))
                    
                    # 构建玩家词列表：1个卧底词，其余平民词
                    player_words = [self.fake_word] + [self.true_word] * (player_num - 1)
                    shuffle(player_words)
                    
                    return player_words
                else:
                    print(f"⚠️ 生成的词格式不正确(尝试{attempt+1}/3): {content}")
                    
            except Exception as e:
                print(f"⚠️ 生成词语时出错(尝试{attempt+1}/3): {e}")
        
        # 如果3次都失败，使用默认词
        print("使用默认词对")
        return self._get_default_words(player_num)
    
    def _parse_word_pair(self, content: str) -> List[str]:
        """
        解析词对，处理多种可能的格式
        
        Args:
            content: AI 返回的文本
            
        Returns:
            词列表
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
                    return words[:2]
        
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
                return matches[:2]
        
        return []
    
    def _get_default_words(self, player_num: int) -> List[str]:
        """
        获取默认词对
        
        Args:
            player_num: 玩家数量
            
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
        
        player_words = [self.fake_word] + [self.true_word] * (player_num - 1)
        shuffle(player_words)
        return player_words


# =====================================================
# 第三部分：游戏主类
# =====================================================

class Game:
    """游戏主控类"""
    
    def __init__(self, player_num: int = PLAYER_NUM):
        """
        初始化游戏
        
        Args:
            player_num: 玩家数量，默认4人
        """
        self.player_num = player_num
        self.client = create_client()
        self.players: List[Player] = [Player(i, self.client) for i in range(player_num)]
        self.judger = Judger(self.client)
        self.game_history: List[Dict] = []
        self.turn_records: List[TurnRecord] = []  # 结构化回合记录
        self.current_round = 0
        self.votes: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(player_num)})
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
        words = self.judger.generate_word(self.player_num)
        for i, player in enumerate(self.players):
            player.set_word(words[i])
        
        # 打印游戏信息
        print(f"\n📋 游戏信息:")
        print(f"   玩家人数: {self.player_num}")
        print(f"   平民词: {self.judger.true_word}")
        print(f"   卧底词: {self.judger.fake_word}")
        
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
    
    def get_undercover_index(self) -> Optional[int]:
        """
        获取卧底玩家的索引
        
        Returns:
            卧底玩家的索引，如果没找到返回 None
        """
        for i, player in enumerate(self.players):
            if self.is_undercover(player):
                return i
        return None
    
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


# 工具 Schema 定义
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "vote",
            "description": "投票给一名玩家，淘汰得票最多的玩家",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_index": {
                        "type": "integer",
                        "description": "被投票的玩家编号（0-3），不能是自己或已出局的玩家"
                    }
                },
                "required": ["target_index"]
            }
        }
    }
]

TOOLS_MAP = {"vote": vote}


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
        description = player.make_description(game.game_history, alive_players)
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
            game.game_history, 
            alive_players, 
            TOOLS_SCHEMA, 
            TOOLS_MAP, 
            game
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
    
    is_undercover = game.is_undercover(eliminated)
    
    print(f"\n{'='*60}")
    print("💀 淘汰公告")
    print(f"{'='*60}")
    print(f"  玩家{player_index} 被淘汰")
    print(f"  身份: {'卧底' if is_undercover else '平民'}")
    print(f"  词语: {eliminated.word}")
    
    if is_undercover:
        # 卧底被淘汰，平民获胜
        print(f"\n  🎉 卧底被淘汰！平民获胜！")
        game.record_history("undercover_eliminated", f"player {player_index}")
        game.ended = True
        game.result = GameResult(
            winner="civilian",
            undercover_index=game.get_undercover_index() or -1,
            true_word=game.judger.true_word or "",
            fake_word=game.judger.fake_word or "",
            rounds=game.current_round,
            duration=(datetime.now() - game.start_time).total_seconds() if game.start_time else 0
        )
        return True
    else:
        # 平民被淘汰，检查卧底是否获胜
        print(f"  💔 平民被淘汰，游戏继续")
        game.record_history("civilian_eliminated", f"player {player_index}")
        
        alive_players = game.get_alive_players()
        alive_count = len(alive_players)
        undercover_alive = sum(1 for i in alive_players if game.is_undercover(game.players[i]))
        
        # 卧底获胜条件：存活玩家只剩2人且卧底在其中
        if alive_count == 2 and undercover_alive == 1:
            undercover_idx = next(i for i in alive_players if game.is_undercover(game.players[i]))
            print(f"\n  🎭 卧底获胜！卧底是玩家{undercover_idx}")
            game.record_history("undercover_win", f"player {undercover_idx}")
            game.ended = True
            game.result = GameResult(
                winner="undercover",
                undercover_index=undercover_idx,
                true_word=game.judger.true_word or "",
                fake_word=game.judger.fake_word or "",
                rounds=game.current_round,
                duration=(datetime.now() - game.start_time).total_seconds() if game.start_time else 0
            )
            return True
    
    return False


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
        
        # 检查游戏是否结束
        if game.ended:
            break
        
        # 检查最大轮数限制（防止无限循环）
        if game.current_round >= 10:
            print(f"\n  ⏰ 达到最大轮数限制，游戏结束")
            game.ended = True
            break
    
    return game.result or GameResult(
        winner="unknown",
        undercover_index=-1,
        true_word="",
        fake_word="",
        rounds=game.current_round,
        duration=0
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
        print(f"  卧底玩家: 玩家{game.result.undercover_index}")
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
    # 创建并运行游戏
    game = Game(player_num=PLAYER_NUM)
    
    if game.game_start():
        result = run_game_loop(game)
        print_game_summary(game)
    else:
        print("❌ 游戏启动失败")
