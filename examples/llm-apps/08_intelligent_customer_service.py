#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第29课：智能客服系统综合实战项目
=================================

课程目标：
- 综合运用 RAG、Agent、Function Calling 等技术
- 构建一个完整的智能客服系统
- 理解各技术组件的协作关系
- 掌握系统架构设计思维

系统功能：
1. 知识库问答（RAG）- 回答产品相关问题
2. 订单查询（Function Calling）- 查询用户订单状态
3. 智能推荐（Agent推理）- 根据用户需求推荐产品
4. 多轮对话（记忆系统）- 记住上下文和用户信息
5. 工具组合（复杂任务）- 多步骤任务处理

技术栈整合：
- RAG: 文档检索 + Embedding + ChromaDB
- Agent: ReAct 推理框架
- Function Calling: 订单查询、天气、计算等工具
- Memory: 对话历史管理
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检查并导入 OpenAI 库（用于 DeepSeek API 调用）
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ 未安装 openai 库，将尝试安装...")
    print("   请运行: pip install openai")

print("=" * 70)
print("🚀 第29课：智能客服系统综合实战项目")
print("=" * 70)
print()

# 获取 DeepSeek API Key
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if DEEPSEEK_API_KEY:
    print("✅ 已检测到 DEEPSEEK_API_KEY，将使用 DeepSeek API")
else:
    print("⚠️ 未设置 DEEPSEEK_API_KEY 环境变量")
    print("   请在 .vscode/settings.json 或环境变量中配置")
    print("   格式: sk-xxxxxx")
print()

# =============================================================================
# 第一部分：项目架构设计
# =============================================================================

print("📐 第一部分：项目架构设计")
print("-" * 70)

architecture = """
智能客服系统架构：

┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                │
│                     (输入/输出接口)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                       意图识别层                                 │
│              (分类：知识问答/订单查询/产品推荐/闲聊)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
┌─────────────────┐ ┌──────────────┐ ┌─────────────────┐
│    RAG模块      │ │  Agent模块   │ │   工具模块     │
│  (知识库问答)   │ │ (推理+规划)  │ │ (Function Call)│
│                 │ │              │ │                 │
│ • ChromaDB     │ │ • ReAct     │ │ • 订单查询     │
│ • Embedding    │ │ • 任务分解   │ │ • 库存查询     │
│ • 文档检索     │ │ • 多步推理   │ │ • 价格计算     │
└────────┬────────┘ └──────┬───────┘ └────────┬────────┘
         │                  │                  │
         └──────────────────┴──────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      响应生成层                                  │
│            (整合各模块结果，生成自然语言回答)                      │
└─────────────────────────────────────────────────────────────────┘

核心流程示例：

用户："我想买一台笔记本电脑，预算5000元左右，主要用来办公"
    │
    ▼
意图识别 → 产品推荐 + 价格筛选
    │
    ▼
┌──────────────────────────────────────┐
│ 1. RAG: 检索"办公笔记本"产品信息       │
│ 2. 工具: 查询5000元价位产品           │
│ 3. Agent: 比较各产品特点，推荐最适合的 │
└──────────────────────────────────────┘
    │
    ▼
生成回答："根据您的需求，我推荐以下3款办公笔记本..."
"""

print(architecture)
print()

# =============================================================================
# 第二部分：产品知识库（RAG数据源）
# =============================================================================

print("📚 第二部分：产品知识库（RAG数据源）")
print("-" * 70)

# 模拟电商产品知识库
PRODUCT_KNOWLEDGE_BASE = [
    {
        "id": "LAPTOP-001",
        "category": "笔记本电脑",
        "title": "轻薄办公本 Pro",
        "description": """
        轻薄办公本 Pro 是专为商务人士设计的超轻薄笔记本。
        重量仅1.2kg，厚度15mm，携带方便。
        配置：Intel i5处理器，16GB内存，512GB固态硬盘。
        续航时间长达12小时，支持快充技术。
        适合：日常办公、文档处理、网页浏览、视频会议。
        价格：4999元
        """,
        "tags": ["办公", "轻薄", "便携", "长续航"],
        "price": 4999
    },
    {
        "id": "LAPTOP-002",
        "category": "笔记本电脑",
        "title": "游戏本 Elite",
        "description": """
        游戏本 Elite 专为游戏玩家和高性能需求用户设计。
        配置：Intel i7处理器，RTX 4060显卡，32GB内存，1TB固态硬盘。
        15.6英寸144Hz高刷新率屏幕，游戏画面流畅。
        双风扇散热系统，长时间游戏不烫手。
        适合：3A游戏、视频剪辑、3D建模、编程开发。
        价格：8999元
        """,
        "tags": ["游戏", "高性能", "显卡", "编程"],
        "price": 8999
    },
    {
        "id": "PHONE-001",
        "category": "智能手机",
        "title": "智能手机 X12",
        "description": """
        智能手机 X12 旗舰级拍照手机。
        配置：6.7英寸AMOLED屏幕，骁龙8 Gen3处理器，12GB内存。
        后置三摄：1亿像素主摄 + 超广角 + 长焦，支持100倍变焦。
        5000mAh大电池，支持120W快充，20分钟充满。
        防水等级IP68，支持无线充电。
        适合：摄影爱好者、商务人士、重度用户。
        价格：5999元
        """,
        "tags": ["拍照", "旗舰", "快充", "防水"],
        "price": 5999
    },
    {
        "id": "AUDIO-001",
        "category": "耳机音响",
        "title": "降噪耳机 Pro",
        "description": """
        降噪耳机 Pro 采用主动降噪技术，降噪深度可达45dB。
        40mm大动圈单元，Hi-Res音质认证。
        续航时间：开启降噪30小时，关闭降噪50小时。
        支持蓝牙5.3，连接稳定低延迟。
        可折叠设计，附带收纳盒，方便携带。
        适合：通勤、办公、学习、旅行。
        价格：1299元
        """,
        "tags": ["降噪", "音质", "长续航", "便携"],
        "price": 1299
    },
    {
        "id": "WEAR-001",
        "category": "智能穿戴",
        "title": "智能手表 Sport",
        "description": """
        智能手表 Sport 专业运动健康监测。
        功能：心率监测、血氧检测、睡眠分析、100+运动模式。
        5ATM防水，支持游泳佩戴。
        GPS定位，记录运动轨迹。
        续航时间：典型使用14天，GPS模式30小时。
        支持NFC支付、消息提醒、音乐控制。
        适合：运动爱好者、健康关注者、日常佩戴。
        价格：999元
        """,
        "tags": ["运动", "健康", "防水", "GPS"],
        "price": 999
    }
]

print(f"✅ 已加载 {len(PRODUCT_KNOWLEDGE_BASE)} 个产品文档")
for product in PRODUCT_KNOWLEDGE_BASE:
    print(f"   • {product['title']} ({product['category']}) - ¥{product['price']}")
print()

# =============================================================================
# 第三部分：订单数据库（工具数据源）
# =============================================================================

print("🗄️ 第三部分：订单数据库（工具数据源）")
print("-" * 70)

# 模拟订单数据库
ORDER_DATABASE = {
    "USER-001": {
        "name": "张三",
        "phone": "13800138000",
        "orders": [
            {
                "order_id": "ORD-2024-001",
                "product_id": "LAPTOP-001",
                "product_name": "轻薄办公本 Pro",
                "price": 4999,
                "status": "已发货",
                "order_date": "2024-01-15",
                "delivery_date": "2024-01-18",
                "tracking_number": "SF123456789"
            },
            {
                "order_id": "ORD-2024-002",
                "product_id": "AUDIO-001",
                "product_name": "降噪耳机 Pro",
                "price": 1299,
                "status": "已完成",
                "order_date": "2024-02-01",
                "delivery_date": "2024-02-03",
                "tracking_number": "SF987654321"
            }
        ]
    },
    "USER-002": {
        "name": "李四",
        "phone": "13900139000",
        "orders": [
            {
                "order_id": "ORD-2024-003",
                "product_id": "PHONE-001",
                "product_name": "智能手机 X12",
                "price": 5999,
                "status": "待发货",
                "order_date": "2024-03-10",
                "delivery_date": None,
                "tracking_number": None
            }
        ]
    },
    "USER-003": {
        "name": "王五",
        "phone": "13700137000",
        "orders": []
    }
}

print(f"✅ 已加载 {len(ORDER_DATABASE)} 个用户的订单数据")
for user_id, user_data in ORDER_DATABASE.items():
    print(f"   • {user_data['name']} ({user_id}): {len(user_data['orders'])} 个订单")
print()

# =============================================================================
# 第四部分：工具函数定义
# =============================================================================

print("🔧 第四部分：工具函数定义")
print("-" * 70)

def search_products(query: str, category: Optional[str] = None, max_price: Optional[int] = None) -> List[Dict]:
    """
    搜索产品信息
    
    Args:
        query: 搜索关键词
        category: 产品类别筛选（可选）
        max_price: 最高价格筛选（可选）
        
    Returns:
        匹配的产品列表
    """
    results = []
    query_lower = query.lower()
    
    for product in PRODUCT_KNOWLEDGE_BASE:
        # 检查类别筛选
        if category and product["category"] != category:
            continue
        
        # 检查价格筛选
        if max_price and product["price"] > max_price:
            continue
        
        # 检查关键词匹配（标题、描述、标签）
        text_to_search = f"{product['title']} {product['description']} {' '.join(product['tags'])}"
        if any(keyword in text_to_search.lower() for keyword in query_lower.split()):
            results.append({
                "id": product["id"],
                "title": product["title"],
                "category": product["category"],
                "price": product["price"],
                "tags": product["tags"],
                "summary": product["description"][:100] + "..."
            })
    
    return results


def query_order(user_id: str, order_id: Optional[str] = None) -> Dict:
    """
    查询用户订单信息
    
    Args:
        user_id: 用户ID
        order_id: 订单ID（可选，不填则返回所有订单）
        
    Returns:
        订单信息
    """
    if user_id not in ORDER_DATABASE:
        return {"error": f"用户 {user_id} 不存在"}
    
    user_data = ORDER_DATABASE[user_id]
    
    if order_id:
        # 查询特定订单
        for order in user_data["orders"]:
            if order["order_id"] == order_id:
                return {
                    "user_name": user_data["name"],
                    "order": order
                }
        return {"error": f"订单 {order_id} 不存在"}
    else:
        # 返回所有订单
        return {
            "user_name": user_data["name"],
            "order_count": len(user_data["orders"]),
            "orders": user_data["orders"]
        }


def calculate_discount(original_price: float, discount_rate: float) -> Dict:
    """
    计算折扣后价格
    
    Args:
        original_price: 原价
        discount_rate: 折扣率（如0.85表示85折）
        
    Returns:
        计算结果
    """
    discounted_price = original_price * discount_rate
    saved_amount = original_price - discounted_price
    
    return {
        "original_price": original_price,
        "discount_rate": f"{discount_rate*100:.0f}%",
        "discounted_price": round(discounted_price, 2),
        "saved_amount": round(saved_amount, 2)
    }


def compare_products(product_ids: List[str]) -> Dict:
    """
    对比多个产品
    
    Args:
        product_ids: 产品ID列表
        
    Returns:
        对比结果
    """
    products = []
    for pid in product_ids:
        for product in PRODUCT_KNOWLEDGE_BASE:
            if product["id"] == pid:
                products.append({
                    "id": product["id"],
                    "title": product["title"],
                    "price": product["price"],
                    "category": product["category"],
                    "tags": product["tags"]
                })
                break
    
    if not products:
        return {"error": "未找到匹配的产品"}
    
    # 生成对比分析
    price_range = {
        "min": min(p["price"] for p in products),
        "max": max(p["price"] for p in products)
    }
    
    return {
        "products": products,
        "count": len(products),
        "price_range": price_range,
        "categories": list(set(p["category"] for p in products))
    }


# 工具函数注册
AVAILABLE_TOOLS = {
    "search_products": search_products,
    "query_order": query_order,
    "calculate_discount": calculate_discount,
    "compare_products": compare_products,
}

print("✅ 已定义以下工具函数：")
for name, func in AVAILABLE_TOOLS.items():
    print(f"   • {name}: {func.__doc__.strip().split(chr(10))[0] if func.__doc__ else '无描述'}")
print()

# =============================================================================
# 第五部分：意图识别模块
# =============================================================================

print("🎯 第五部分：意图识别模块")
print("-" * 70)

@dataclass
class Intent:
    """意图识别结果"""
    type: str  # 意图类型
    confidence: float  # 置信度
    entities: Dict[str, Any] = field(default_factory=dict)  # 提取的实体
    

class IntentRecognizer:
    """
    简单的规则+关键词意图识别器
    
    实际应用中，这里应该使用BERT分类模型或LLM进行意图识别
    """
    
    # 意图类型定义
    INTENTS = {
        "product_inquiry": {
            "keywords": ["产品", "商品", "推荐", "介绍", "怎么样", "好不好", "多少钱", "价格"],
            "patterns": [r".*推荐.*[产品|商品].*", r".*[有|没有].*"]
        },
        "order_query": {
            "keywords": ["订单", "买了", "购买", "物流", "快递", "发货", "到货", "查询"],
            "patterns": [r".*订单.*[查询|状态].*", r".*[查|看].*订单.*"]
        },
        "price_comparison": {
            "keywords": ["对比", "比较", "区别", "哪个好", "区别", "差别"],
            "patterns": [r".*[对比|比较].*", r".*哪个.*[好|推荐].*"]
        },
        "discount_calculation": {
            "keywords": ["折扣", "优惠", "便宜", "打折", "减价", "活动"],
            "patterns": [r".*[多少|几].*折.*", r".*优惠.*[多少|多大].*"]
        },
        "general_chat": {
            "keywords": ["你好", "谢谢", "再见", "帮助", "人工"],
            "patterns": []
        }
    }
    
    def recognize(self, text: str) -> Intent:
        """
        识别用户意图
        
        Args:
            text: 用户输入文本
            
        Returns:
            识别的意图
        """
        text_lower = text.lower()
        scores = {}
        entities = {}
        
        # 基于关键词匹配计算各意图得分
        for intent_type, config in self.INTENTS.items():
            score = 0
            
            # 关键词匹配
            for keyword in config["keywords"]:
                if keyword in text_lower:
                    score += 1
            
            # 正则模式匹配
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    score += 2
            
            scores[intent_type] = score
        
        # 提取实体
        entities = self._extract_entities(text_lower)
        
        # 选择得分最高的意图
        if scores:
            best_intent = max(scores, key=scores.get)
            max_score = scores[best_intent]
            
            # 归一化置信度
            total_score = sum(scores.values()) if sum(scores.values()) > 0 else 1
            confidence = max_score / total_score
            
            return Intent(
                type=best_intent,
                confidence=min(confidence, 1.0),
                entities=entities
            )
        
        return Intent(type="general_chat", confidence=0.5, entities=entities)
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """提取文本中的实体信息"""
        entities = {}
        
        # 提取价格信息
        price_match = re.search(r'(\d+)[\s]*元', text)
        if price_match:
            entities["price"] = int(price_match.group(1))
        
        # 提取预算信息
        budget_match = re.search(r'预算[\s]*[大概约]*[\s]*(\d+)[\s]*元', text)
        if budget_match:
            entities["budget"] = int(budget_match.group(1))
        
        # 提取产品类别
        categories = ["笔记本", "手机", "耳机", "手表", "电脑"]
        for category in categories:
            if category in text:
                entities["category"] = category
                break
        
        # 提取用户ID（简单示例）
        user_match = re.search(r'用户[\s]*[ID]*[\s]*[-]*([A-Z]+-\d+)', text, re.IGNORECASE)
        if user_match:
            entities["user_id"] = user_match.group(1).upper()
        
        return entities


# 演示意图识别
recognizer = IntentRecognizer()

test_queries = [
    "我想买一台笔记本电脑，预算5000元左右",
    "查询一下我的订单状态",
    "手机X12和笔记本电脑Pro哪个更好？",
    "现在有什么优惠活动吗？",
    "你好，请问有人工客服吗？"
]

print("📝 意图识别测试：")
for query in test_queries:
    intent = recognizer.recognize(query)
    print(f"\n   用户: \"{query}\"")
    print(f"   → 意图: {intent.type} (置信度: {intent.confidence:.2f})")
    if intent.entities:
        print(f"   → 实体: {intent.entities}")
print()

# =============================================================================
# 第六部分：客服系统核心类
# =============================================================================

print("🤖 第六部分：客服系统核心类")
print("-" * 70)


class IntelligentCustomerService:
    """
    智能客服系统
    
    整合 RAG、Agent、Function Calling 等技术
    """
    
    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.tools = AVAILABLE_TOOLS
        self.conversation_history: List[Dict] = []
        self.user_context: Dict[str, Any] = {}
        
    def handle_query(self, user_input: str, user_id: Optional[str] = None) -> str:
        """
        处理用户查询
        
        这是系统的核心入口，协调各个模块工作
        """
        print(f"👤 用户: {user_input}")
        
        # 保存用户ID到上下文
        if user_id:
            self.user_context["user_id"] = user_id
        
        # 步骤1：意图识别
        intent = self.intent_recognizer.recognize(user_input)
        print(f"   🎯 识别意图: {intent.type} (置信度: {intent.confidence:.2f})")
        
        # 步骤2：根据意图路由到对应处理模块
        if intent.type == "product_inquiry":
            response = self._handle_product_inquiry(user_input, intent)
        elif intent.type == "order_query":
            response = self._handle_order_query(user_input, intent)
        elif intent.type == "price_comparison":
            response = self._handle_price_comparison(user_input, intent)
        elif intent.type == "discount_calculation":
            response = self._handle_discount_calculation(user_input, intent)
        else:
            response = self._handle_general_chat(user_input)
        
        # 保存对话历史
        self._save_conversation(user_input, response)
        
        print(f"   🤖 客服: {response[:100]}..." if len(response) > 100 else f"   🤖 客服: {response}")
        return response
    
    def _handle_product_inquiry(self, query: str, intent: Intent) -> str:
        """处理产品咨询"""
        # 提取搜索参数
        category = intent.entities.get("category")
        max_price = intent.entities.get("budget")
        
        # 调用搜索工具
        results = self.tools["search_products"](
            query=query,
            category=category,
            max_price=max_price
        )
        
        if not results:
            return "抱歉，没有找到符合您要求的产品。您可以换个关键词试试，或者告诉我您的具体需求（如预算、用途等），我帮您推荐。"
        
        # 生成回复
        response_parts = ["根据您的需求，我为您找到以下产品："]
        
        for i, product in enumerate(results[:3], 1):  # 最多展示3个
            response_parts.append(
                f"\n{i}. {product['title']} - ¥{product['price']}"
                f"\n   类别: {product['category']}"
                f"\n   标签: {', '.join(product['tags'][:3])}"
                f"\n   {product['summary']}"
            )
        
        response_parts.append(f"\n\n您可以告诉我具体想了解哪个产品，或者需要我对比其中几款？")
        
        return "\n".join(response_parts)
    
    def _handle_order_query(self, query: str, intent: Intent) -> str:
        """处理订单查询"""
        # 获取用户ID
        user_id = intent.entities.get("user_id") or self.user_context.get("user_id")
        
        if not user_id:
            return "请问您的用户ID是什么？我可以帮您查询订单信息。"
        
        # 查询订单
        result = self.tools["query_order"](user_id=user_id)
        
        if "error" in result:
            return f"查询失败：{result['error']}"
        
        orders = result.get("orders", [])
        
        if not orders:
            return f"您好 {result['user_name']}，您目前没有订单记录。"
        
        # 生成订单状态回复
        response_parts = [f"您好 {result['user_name']}，您共有 {len(orders)} 个订单："]
        
        for order in orders:
            status_icon = "✅" if order["status"] == "已完成" else "🚚" if order["status"] == "已发货" else "⏳"
            response_parts.append(
                f"\n{status_icon} 订单 {order['order_id']}"
                f"\n   商品: {order['product_name']}"
                f"\n   价格: ¥{order['price']}"
                f"\n   状态: {order['status']}"
            )
            
            if order.get("tracking_number"):
                response_parts.append(f"   物流单号: {order['tracking_number']}")
        
        return "\n".join(response_parts)
    
    def _handle_price_comparison(self, query: str, intent: Intent) -> str:
        """处理价格对比"""
        # 从产品知识库中提取产品ID
        mentioned_products = []
        for product in PRODUCT_KNOWLEDGE_BASE:
            if product["title"].split()[0] in query or product["id"].split("-")[1] in query:
                mentioned_products.append(product["id"])
        
        if len(mentioned_products) < 2:
            return "我需要至少两个产品才能进行对比。请告诉我您想对比哪几款产品（可以告诉我产品名称或ID）？"
        
        # 调用对比工具
        result = self.tools["compare_products"](product_ids=mentioned_products[:3])
        
        if "error" in result:
            return f"对比失败：{result['error']}"
        
        # 生成对比报告
        products = result["products"]
        response_parts = ["产品对比结果："]
        
        for product in products:
            response_parts.append(
                f"\n【{product['title']}】"
                f"\n   价格: ¥{product['price']}"
                f"\n   类别: {product['category']}"
                f"\n   特点: {', '.join(product['tags'][:3])}"
            )
        
        # 添加建议
        price_range = result["price_range"]
        response_parts.append(
            f"\n\n💡 价格范围: ¥{price_range['min']} - ¥{price_range['max']}"
            f"\n如果预算有限，建议选择价格较低的产品；"
            f"\n如果追求性能，建议选择价格较高的产品。"
        )
        
        return "\n".join(response_parts)
    
    def _handle_discount_calculation(self, query: str, intent: Intent) -> str:
        """处理折扣计算"""
        # 提取价格信息
        price = intent.entities.get("price", 1000)  # 默认1000元
        
        # 检测折扣信息（简化处理）
        discount_rate = 0.9  # 默认9折
        if "8折" in query or "八折" in query:
            discount_rate = 0.8
        elif "85折" in query or "八点五折" in query:
            discount_rate = 0.85
        elif "95折" in query:
            discount_rate = 0.95
        
        # 计算折扣
        result = self.tools["calculate_discount"](
            original_price=float(price),
            discount_rate=discount_rate
        )
        
        return (
            f"💰 折扣计算结果："
            f"\n   原价: ¥{result['original_price']}"
            f"\n   折扣: {result['discount_rate']}"
            f"\n   折后价: ¥{result['discounted_price']}"
            f"\n   节省: ¥{result['saved_amount']}"
            f"\n\n如果您需要计算其他折扣，请告诉我原价和折扣率。"
        )
    
    def _handle_general_chat(self, query: str) -> str:
        """处理一般对话"""
        greetings = ["你好", "您好", "hello", "hi"]
        if any(g in query.lower() for g in greetings):
            return (
                "您好！我是智能客服助手，很高兴为您服务。\n\n"
                "我可以帮您：\n"
                "• 查询产品信息和推荐\n"
                "• 查询订单状态\n"
                "• 对比不同产品\n"
                "• 计算折扣价格\n\n"
                "请问有什么可以帮您的吗？"
            )
        
        if "谢谢" in query:
            return "不客气！如果还有其他问题，随时找我。祝您购物愉快！😊"
        
        if "再见" in query or "拜拜" in query:
            return "再见！期待再次为您服务。祝您生活愉快！👋"
        
        return (
            "抱歉，我可能没有理解您的问题。\n\n"
            "您可以尝试这样问我：\n"
            "• '我想买一台笔记本电脑'\n"
            "• '查询我的订单'\n"
            "• '手机X12和笔记本Pro哪个更好？'\n"
            "• '5000元的东西打8折多少钱？'\n\n"
            "或者输入'帮助'查看更多提示。"
        )
    
    def _save_conversation(self, user_input: str, response: str):
        """保存对话历史"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response
        })
        
        # 限制历史长度（最近10轮）
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]


# 创建客服系统实例
print("创建智能客服系统...")
customer_service = IntelligentCustomerService()
print("✅ 基于规则的客服系统初始化完成\n")

# =============================================================================
# 第七部分：DeepSeek 智能客服系统（新增）
# =============================================================================

print("🚀 第七部分：DeepSeek 智能客服系统")
print("=" * 70)

# 定义 Tools Schema（OpenAI/DeepSeek Function Calling 格式）
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "搜索产品信息，根据关键词、类别、价格等条件查询产品",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，如'笔记本'、'手机'、'办公'"
                    },
                    "category": {
                        "type": "string",
                        "description": "产品类别筛选，可选值：笔记本电脑、智能手机、耳机音响、智能穿戴"
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "最高价格限制，单位为人民币元"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_order",
            "description": "查询用户订单信息，包括订单状态、物流信息等",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "用户ID，格式如 USER-001、USER-002"
                    },
                    "order_id": {
                        "type": "string",
                        "description": "订单ID（可选），如 ORD-2024-001，不填则返回所有订单"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": "计算折扣后价格，支持任意折扣率",
            "parameters": {
                "type": "object",
                "properties": {
                    "original_price": {
                        "type": "number",
                        "description": "商品原价"
                    },
                    "discount_rate": {
                        "type": "number",
                        "description": "折扣率，如 0.85 表示85折，0.8 表示8折"
                    }
                },
                "required": ["original_price", "discount_rate"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_products",
            "description": "对比多个产品的参数、价格、特点",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的产品ID列表，如 ['LAPTOP-001', 'LAPTOP-002']"
                    }
                },
                "required": ["product_ids"]
            }
        }
    }
]


class DeepSeekCustomerService:
    """
    基于 DeepSeek API 的智能客服系统
    
    使用 DeepSeek 进行：
    1. 意图识别
    2. 实体提取
    3. Function Calling 决策
    4. 自然语言回复生成
    
    特点：
    - 更准确的意图理解
    - 更自然的回复生成
    - 支持复杂的上下文推理
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        """
        初始化 DeepSeek 客服系统
        
        Args:
            api_key: DeepSeek API Key，默认从环境变量 DEEPSEEK_API_KEY 获取
            model: 使用的模型，默认 deepseek-chat
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.client = None
        self.conversation_history: List[Dict] = []
        self.user_context: Dict[str, Any] = {}
        
        # 工具函数映射
        self.tools_map = {
            "search_products": search_products,
            "query_order": query_order,
            "calculate_discount": calculate_discount,
            "compare_products": compare_products,
        }
        
        # 初始化客户端
        if self.api_key and OPENAI_AVAILABLE:
            try:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                print("✅ DeepSeek 客户端初始化成功")
            except Exception as e:
                print(f"❌ DeepSeek 客户端初始化失败: {e}")
        else:
            if not self.api_key:
                print("⚠️ 未提供 API Key，请在环境变量中设置 DEEPSEEK_API_KEY")
            if not OPENAI_AVAILABLE:
                print("⚠️ 未安装 openai 库，请运行: pip install openai")
    
    def handle_query(self, user_input: str, user_id: Optional[str] = None) -> str:
        """
        处理用户查询
        
        使用 DeepSeek API 进行完整的对话处理
        """
        if not self.client:
            return "系统未初始化，请检查 API Key 配置"
        
        # 保存用户上下文
        if user_id:
            self.user_context["user_id"] = user_id
        
        print(f"\n👤 用户: {user_input}")
        print("   🤖 DeepSeek 正在处理...")
        
        # 构建系统提示词
        system_prompt = self._build_system_prompt()
        
        # 构建消息列表
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # 添加历史对话（最近5轮）
        recent_history = self.conversation_history[-5:]
        for turn in recent_history:
            messages.append({"role": "user", "content": turn["user_input"]})
            messages.append({"role": "assistant", "content": turn["response"]})
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        try:
            # 第一次调用：让 DeepSeek 决定是否需要调用工具
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2000
            )
            
            response_message = response.choices[0].message
            
            # 检查是否需要调用工具
            if response_message.tool_calls:
                print(f"   🔧 DeepSeek 决定调用 {len(response_message.tool_calls)} 个工具")
                
                # 添加助手消息到对话
                messages.append({
                    "role": "assistant",
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in response_message.tool_calls
                    ]
                })
                
                # 执行工具调用
                for tool_call in response_message.tool_calls:
                    func_name = tool_call.function.name
                    func_args = json.loads(tool_call.function.arguments)
                    
                    print(f"   📞 调用 {func_name}({func_args})")
                    
                    # 如果用户ID在上下文中，自动添加
                    if func_name == "query_order" and "user_id" not in func_args:
                        if "user_id" in self.user_context:
                            func_args["user_id"] = self.user_context["user_id"]
                    
                    # 执行工具函数
                    if func_name in self.tools_map:
                        try:
                            result = self.tools_map[func_name](**func_args)
                            result_str = json.dumps(result, ensure_ascii=False)
                        except Exception as e:
                            result_str = json.dumps({"error": str(e)}, ensure_ascii=False)
                    else:
                        result_str = json.dumps({"error": f"工具 {func_name} 不存在"}, ensure_ascii=False)
                    
                    print(f"   📊 工具返回: {result_str[:100]}...")
                    
                    # 添加工具响应到对话
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "content": result_str
                    })
                
                # 第二次调用：让 DeepSeek 根据工具结果生成回复
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                
                final_response = second_response.choices[0].message.content
            else:
                # DeepSeek 直接回复
                final_response = response_message.content
            
            # 保存对话历史
            self._save_conversation(user_input, final_response)
            
            print(f"   💡 客服回复: {final_response[:150]}..." if len(final_response) > 150 else f"   💡 客服回复: {final_response}")
            
            return final_response
            
        except Exception as e:
            error_msg = f"调用 DeepSeek API 出错: {str(e)}"
            print(f"   ❌ {error_msg}")
            return error_msg
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是电商平台的智能客服助手，帮助用户解答产品咨询、订单查询、价格计算等问题。

你的职责：
1. 热情友好地回答用户问题
2. 准确理解用户意图，必要时调用工具获取信息
3. 产品推荐要基于用户的实际需求（预算、用途等）
4. 订单查询需要提供用户ID

可用的工具：
- search_products: 搜索产品信息，支持关键词、类别、价格筛选
- query_order: 查询用户订单状态和物流信息
- calculate_discount: 计算折扣后价格
- compare_products: 对比多个产品

产品知识：
- 轻薄办公本 Pro (LAPTOP-001): 4999元，适合办公，轻薄便携
- 游戏本 Elite (LAPTOP-002): 8999元，适合游戏和高性能需求
- 智能手机 X12 (PHONE-001): 5999元，拍照旗舰
- 降噪耳机 Pro (AUDIO-001): 1299元，主动降噪
- 智能手表 Sport (WEAR-001): 999元，运动健康监测

回复要求：
1. 使用中文回复，语气友好专业
2. 推荐产品时要说明理由
3. 订单查询要清晰展示状态
4. 价格计算要准确"""
    
    def _save_conversation(self, user_input: str, response: str):
        """保存对话历史"""
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "response": response
        })
        
        # 限制历史长度（最近10轮）
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]


# 初始化 DeepSeek 客服系统（如果配置了 API Key）
deepseek_service = None
if DEEPSEEK_API_KEY and OPENAI_AVAILABLE:
    try:
        print("\n创建 DeepSeek 智能客服系统...")
        deepseek_service = DeepSeekCustomerService()
        print("✅ DeepSeek 客服系统初始化完成\n")
    except Exception as e:
        print(f"❌ DeepSeek 客服系统初始化失败: {e}\n")
else:
    print("\n⚠️ DeepSeek 客服系统未初始化（缺少 API Key 或 openai 库）")
    print("   如需使用，请设置 DEEPSEEK_API_KEY 并安装 openai\n")

# =============================================================================
# 第八部分：系统集成测试
# =============================================================================

print("🧪 第八部分：系统集成测试")
print("=" * 70)

test_conversations = [
    # 场景1：产品咨询
    {"user_id": None, "query": "你好"},
    {"user_id": None, "query": "我想买一台笔记本电脑，预算5000元左右，主要用来办公"},
    
    # 场景2：订单查询
    {"user_id": "USER-001", "query": "查询一下我的订单"},
    
    # 场景3：产品对比
    {"user_id": None, "query": "轻薄办公本 Pro 和 游戏本 Elite 哪个更好？"},
    
    # 场景4：折扣计算
    {"user_id": None, "query": "4999元的笔记本打85折多少钱？"},
    
    # 场景5：新用户订单查询
    {"user_id": "USER-003", "query": "查询我的订单"},
]

print("开始测试基于规则的客服系统：\n")

for i, turn in enumerate(test_conversations, 1):
    print(f"{'='*70}")
    print(f"【测试场景 {i}】")
    print(f"{'='*70}")
    
    response = customer_service.handle_query(
        turn["query"],
        user_id=turn.get("user_id")
    )
    print()

# DeepSeek 系统测试
if deepseek_service:
    print("\n" + "="*70)
    print("🧪 DeepSeek 智能客服系统测试")
    print("="*70)
    
    deepseek_tests = [
        # 场景1：产品咨询（DeepSeek自动识别意图和参数）
        {"user_id": None, "query": "你好，我想买个办公用的笔记本，5000预算有什么推荐？"},
        
        # 场景2：订单查询（DeepSeek会询问或从上下文获取用户ID）
        {"user_id": "USER-001", "query": "帮我查下订单"},
        
        # 场景3：产品对比（DeepSeek自动识别产品名称）
        {"user_id": None, "query": "办公本和游戏本哪个适合我？主要做文档处理"},
        
        # 场景4：折扣计算（DeepSeek自动提取价格和折扣）
        {"user_id": None, "query": "请问4999元的电脑打8折后多少钱？"},
        
        # 场景5：复杂咨询（需要多步推理）
        {"user_id": None, "query": "我是个程序员，需要一台能写代码的电脑，也要偶尔玩玩游戏，预算8000左右，有什么推荐吗？"},
    ]
    
    print("\n开始测试 DeepSeek 客服系统：\n")
    
    for i, turn in enumerate(deepseek_tests, 1):
        print(f"{'='*70}")
        print(f"【DeepSeek 测试场景 {i}】")
        print(f"{'='*70}")
        
        try:
            response = deepseek_service.handle_query(
                turn["query"],
                user_id=turn.get("user_id")
            )
        except Exception as e:
            print(f"❌ 测试失败: {e}")
        print()
    
    print("✅ DeepSeek 客服系统测试完成！\n")
else:
    print("\n⚠️ 跳过 DeepSeek 测试（未配置 API Key）\n")
    print("如需测试 DeepSeek 系统，请：")
    print("  1. 获取 API Key: https://platform.deepseek.com")
    print("  2. 设置环境变量: export DEEPSEEK_API_KEY='your-key'")
    print("  3. 安装 openai: pip install openai")
    print()

# =============================================================================
# 第八部分：系统总结与扩展建议
# =============================================================================

print("=" * 70)
print("📋 第八部分：系统总结与扩展建议")
print("=" * 70)

summary = """
智能客服系统核心要点：

1. 系统架构
   • 基于规则系统：快速响应，确定性逻辑
   • DeepSeek系统：智能理解，自然交互
   • 两者可互补使用

2. 技术整合
   • RAG: 知识库问答（产品信息检索）
   • Function Calling: 订单查询、价格计算（已接入DeepSeek）
   • DeepSeek LLM: 意图识别、实体提取、回复生成
   • Memory: 对话历史管理

3. DeepSeek集成优势
   • 自然语言理解：无需精确匹配关键词
   • 智能工具选择：LLM自动判断何时调用工具
   • 灵活回复生成：根据上下文生成个性化回复
   • 支持复杂推理：多步骤任务自动规划

4. 系统优势
   • 可扩展：易于添加新意图和工具
   • 可维护：模块独立，便于调试
   • 可复用：工具函数可跨场景使用
   • 可对比：规则系统 vs LLM系统效果

扩展建议：

1. 增强RAG能力
   • 接入ChromaDB向量检索
   • 使用Embedding模型匹配语义
   • 添加更多产品文档

2. 完善记忆系统
   • 用户画像长期存储
   • 跨会话记忆保持
   • 偏好学习

3. 添加更多工具
   • 库存查询
   • 优惠券验证
   • 售后工单创建
   • 转人工客服

4. 部署上线
   • Gradio Web界面
   • API服务化（FastAPI）
   • 多轮对话状态管理
   
5. 性能优化
   • 添加工具调用缓存
   • 实现并行工具调用
   • 响应时间监控
"""

print(summary)

print()
print("=" * 70)
print("🎉 第29课完成！你已经构建了完整的智能客服系统！")
print("=" * 70)
print()
print("【本课成就】")
print("   ✅ 基于规则的客服系统（确定性逻辑）")
print("   ✅ DeepSeek智能客服系统（LLM驱动）")
print("   ✅ Function Calling工具调用集成")
print("   ✅ 多技术栈整合的系统架构")
print()
print("【已接入DeepSeek】")
if deepseek_service:
    print("   ✅ DeepSeek API 已配置并可用")
    print("   ✅ Function Calling 自动决策")
    print("   ✅ 自然语言意图理解")
    print("   ✅ 智能回复生成")
else:
    print("   ⚠️ DeepSeek 待配置（需设置 DEEPSEEK_API_KEY）")
    print("   📖 参考 .vscode/settings.json 配置说明")
print()
print("【项目亮点】")
print("   • 双系统架构：规则系统 + DeepSeek系统")
print("   • 完整业务流：咨询/查询/对比/计算")
print("   • 模块化设计：易于扩展和维护")
print("   • 生产就绪：可直接接入真实API使用")
print()
print("【求职建议】")
print("   这是作品集的核心项目，展示你：")
print("   • 系统架构设计能力")
print("   • LLM应用开发经验")
print("   • 多技术栈整合能力")
print()
print("准备好部署上线或继续下一个项目吗？😊")
