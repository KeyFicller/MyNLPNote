# toy/common — 共享模块
#
# 日志系统、API 客户端、游戏引擎框架——被多个 toy 游戏复用。

from common.logger import setup_game_logging, game_print, close_game_logging, original_print
from common.client import create_client, API_KEY, BASE_URL, MODEL_NAME
from common.engine import BasePlayer, BaseGame, GameResult
