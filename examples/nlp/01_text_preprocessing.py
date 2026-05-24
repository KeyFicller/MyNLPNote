#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第15课：文本预处理与词嵌入（NLP第一课！）
============================================
本课程学习NLP的第一步：
- 文本预处理：分词、清洗、规范化
- 中文 vs 英文分词差异
- 词嵌入：从One-hot到Word2Vec
- 词向量运算：有趣的词向量算术

需要安装：pip install jieba numpy
"""

import re
import numpy as np
from collections import Counter
import torch
import torch.nn as nn

print("=" * 70)
print("第15课：文本预处理与词嵌入 🚀 NLP第一课！")
print("=" * 70)

# ============================================================
# 第一部分：文本预处理流程
# ============================================================
print("\n" + "=" * 70)
print("第一部分：文本预处理 - 让机器理解文本的第一步")
print("=" * 70)

print("""
【为什么需要预处理？】
原始文本: "Hello!! NLP is AMAZING... 🤩  有＃很￥多噪声！！"
机器看到: 各种符号、大小写、特殊字符、多余空格

预处理后: "hello nlp is amazing 有 很 多 噪声"
机器看到: 干净的、规范的、统一的格式

【标准预处理流程】
1. 去除噪声（HTML标签、URL、特殊符号）
2. 规范化（统一大小写、全角转半角）
3. 分词（将句子切成词）
4. 去除停用词（的、是、了等无意义词）
5. 可选：词干提取/词形还原（英文）
""")

# 示例文本
raw_text = """
【重磅消息】🎉 人工智能NLP技术突破！！！
访问 https://example.com 了解更多。
模型准确率达到99.9%，效果 AMAZING...
Email: contact@nlp.com  #AI #深度学习
"""

print("\n【示例1：原始文本】")
print(f"{raw_text}")

# 预处理函数
def clean_text(text):
    """文本清洗函数"""
    # 1. 去除HTML标签（如果有）
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 去除URL
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # 3. 去除email
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. 去除特殊符号和表情，但保留中英文和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    
    # 5. 去除多余空格
    text = ' '.join(text.split())
    
    # 6. 统一小写（英文）
    text = text.lower()
    
    return text

cleaned = clean_text(raw_text)
print("【清洗后】")
print(f"{cleaned}")

# ============================================================
# 第二部分：中文分词
# ============================================================
print("\n" + "=" * 70)
print("第二部分：中文分词 - 比英文更复杂！")
print("=" * 70)

print("""
【中文分词的难点】
英文: "I love natural language processing"
     ↓ 按空格分
     ["I", "love", "natural", "language", "processing"]

中文: "我爱自然语言处理"
     ↓ ???
     ["我", "爱", "自然语言", "处理"]  ✓
     还是 ["我", "爱", "自然", "语言", "处理"] ?
     还是 ["我", "爱", "自", "然", "语", "言", "处", "理"] ?

【常用中文分词工具】
- jieba: 最流行，支持自定义词典
- HanLP: 功能丰富，学术效果好
- pkuseg: 北大出品，专业领域分词好
- THULAC: 清华出品，速度快
""")

# 简单模拟分词（不依赖外部库）
def simple_chinese_segment(text):
    """
    简化版中文分词模拟
    实际应使用 jieba.cut()
    """
    # 模拟常见词典
    dictionary = {
        '自然语言', '处理', '人工智能', '深度学习', '机器学习',
        '神经网络', ' Transformer', '模型', '训练', '数据',
        '文本', '分词', '词向量', '嵌入', '预处理'
    }
    
    words = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        # 尝试匹配最长词
        matched = False
        for length in range(min(6, text_len - i), 0, -1):  # 最大6个字
            substr = text[i:i+length]
            if substr in dictionary or length == 1:
                words.append(substr)
                i += length
                matched = True
                break
        
        if not matched:
            words.append(text[i])
            i += 1
    
    return words

chinese_text = "我爱自然语言处理"
print(f"\n【示例2：中文分词】")
print(f"原文: {chinese_text}")
print(f"分词: {simple_chinese_segment(chinese_text)}")

# 更多示例
examples = [
    "深度学习模型训练需要大量数据",
    "人工智能正在改变我们的生活",
    "文本预处理是NLP的第一步",
]

print("\n【更多分词示例】")
for text in examples:
    words = simple_chinese_segment(text)
    print(f"  {text}")
    print(f"    → {' | '.join(words)}")

# ============================================================
# 第三部分：英文分词
# ============================================================
print("\n" + "=" * 70)
print("第三部分：英文分词 - 相对简单但有细节")
print("=" * 70)

def simple_english_tokenize(text):
    """
    简化版英文分词
    实际应使用 nltk.word_tokenize() 或 spacy
    """
    # 1. 转小写
    text = text.lower()
    
    # 2. 按空格和标点分词
    tokens = re.findall(r"\b\w+\b", text)
    
    return tokens

english_text = "Natural Language Processing (NLP) is a subfield of AI."
print(f"\n【示例3：英文分词】")
print(f"原文: {english_text}")
print(f"分词: {simple_english_tokenize(english_text)}")

print("""
【英文特殊处理】
1. 词干提取（Stemming）: running → run, better → good
2. 词形还原（Lemmatization）: 更智能的词干提取
3. 停用词: 去除 the, is, at, which 等高频无意义词
4. N-gram: 将连续N个词作为一个单元
""")

# N-gram示例
def generate_ngrams(tokens, n=2):
    """生成N-gram"""
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

tokens = ["natural", "language", "processing", "is", "amazing"]
print(f"\n【N-gram示例】")
print(f"Tokens: {tokens}")
print(f"Bigram (2-gram): {generate_ngrams(tokens, 2)}")
print(f"Trigram (3-gram): {generate_ngrams(tokens, 3)}")

# ============================================================
# 第四部分：词嵌入 - 从符号到向量
# ============================================================
print("\n" + "=" * 70)
print("第四部分：词嵌入 - 让词变成机器能计算的向量")
print("=" * 70)

print("""
【词表示的演进】

1. One-hot 编码（最简单但无效）
   "猫" = [1, 0, 0, 0, ...]
   "狗" = [0, 1, 0, 0, ...]
   "鱼" = [0, 0, 1, 0, ...]
   
   ❌ 问题：维度灾难（10万词=10万维）、无法计算相似度

2. 词嵌入 Word Embedding（革命性！）
   "猫" = [0.2, -0.5, 0.8, ...]  # 300维稠密向量
   "狗" = [0.3, -0.4, 0.9, ...]  # 语义相近，向量相似！
   "鱼" = [-0.8, 0.2, 0.1, ...]  # 语义不同，向量不同
   
   ✅ 优点：低维度、语义相似度可计算、支持向量运算
""")

# One-hot示例
vocab = ["猫", "狗", "鱼", "鸟", "人"]
vocab_size = len(vocab)

print(f"\n【示例4：One-hot编码（词典大小={vocab_size}）】")
for i, word in enumerate(vocab):
    one_hot = np.zeros(vocab_size)
    one_hot[i] = 1
    print(f"  {word}: {one_hot}")

print(f"\n❌ 如果词典有10万词，每个词需要10万维！")

# 模拟词嵌入
embedding_dim = 5
np.random.seed(42)

print(f"\n【示例5：词嵌入（维度={embedding_dim}）】")
embeddings = {}
for word in vocab:
    # 模拟训练好的词向量
    embeddings[word] = np.random.randn(embedding_dim) * 0.5
    print(f"  {word}: {embeddings[word].round(3)}")

# 计算相似度
def cosine_similarity(v1, v2):
    """计算余弦相似度"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print(f"\n【语义相似度计算】")
print(f"猫 vs 狗 (动物): {cosine_similarity(embeddings['猫'], embeddings['狗']):.3f}")
print(f"猫 vs 鱼 (动物): {cosine_similarity(embeddings['猫'], embeddings['鱼']):.3f}")
print(f"猫 vs 人 (较远): {cosine_similarity(embeddings['猫'], embeddings['人']):.3f}")

# ============================================================
# 第五部分：Word2Vec原理
# ============================================================
print("\n" + "=" * 70)
print("第五部分：Word2Vec - 经典词嵌入模型")
print("=" * 70)

print("""
【Word2Vec核心思想】
"一个词的含义由它的上下文决定" —— 分布假说 (Distributional Hypothesis)

【两种训练方式】

1. CBOW (Continuous Bag of Words)
   上下文 → 预测中心词
   
   ["我", "喜欢", "___", "处理"] → "自然语言"
    上下文        目标词

2. Skip-gram (更常用)
   中心词 → 预测上下文
   
   "自然语言" → ["我", "喜欢", "处理"]
    中心词          上下文

【训练过程】
1. 初始化：随机初始化所有词的向量
2. 滑动窗口：在语料上滑动，获取(中心词, 上下文)对
3. 预测训练：用神经网络学习词向量
4. 收敛：语义相近的词在向量空间中靠近
""")

# 模拟Word2Vec训练过程
print("\n【Word2Vec训练示意】")
corpus = [
    "自然语言处理是人工智能的重要分支",
    "深度学习在自然语言处理中应用广泛",
    "机器学习是人工智能的核心技术",
    "神经网络是深度学习的基础模型",
    "文本预处理是自然语言处理的第一步",
]

print("语料库示例：")
for i, sentence in enumerate(corpus[:3], 1):
    print(f"  {i}. {sentence}")

print("""
训练过程可视化：

Epoch 1: 随机向量
  猫 = [0.1, -0.2, 0.3]
  狗 = [0.5, 0.8, -0.1]  ← 距离很远

Epoch 100: 向量开始聚集
  猫 = [0.2, -0.5, 0.8]
  狗 = [0.3, -0.4, 0.9]  ← 距离变近！

Epoch 1000: 语义空间形成
  动物类: 猫、狗、鸟聚集在一起
  食物类: 苹果、香蕉、橙子聚集在一起
  地点类: 北京、上海、纽约聚集在一起
""")

# ============================================================
# 第六部分：有趣的词向量算术
# ============================================================
print("\n" + "=" * 70)
print("第六部分：词向量的神奇算术 🧮")
print("=" * 70)

print("""
【经典例子】
King - Man + Woman ≈ Queen
(国王 - 男人 + 女人 = 女王)

Paris - France + Italy ≈ Rome
(巴黎 - 法国 + 意大利 = 罗马)

【原理】
词向量捕捉了语义关系：
- King 和 Queen 的差异 ≈ Man 和 Woman 的差异（性别）
- Paris 和 France 的关系 ≈ Rome 和 Italy 的关系（首都-国家）
""")

# 模拟词向量算术
def word_vector_math(vocab_embeddings):
    """演示词向量算术"""
    v = vocab_embeddings
    
    # 模拟：中国 - 北京 + 东京 ≈ 日本
    result = v['中国'] - v['北京'] + v['东京']
    
    # 在词表中找最接近的词
    best_match = None
    best_score = -float('inf')
    
    for word in v:
        if word not in ['中国', '北京', '东京']:
            score = cosine_similarity(result, v[word])
            if score > best_score:
                best_score = score
                best_match = word
    
    return best_match, best_score

# 创建模拟的词向量空间（手动设置，确保有意义的相似度）
np.random.seed(42)
countries = ["中国", "日本", "美国", "法国", "英国"]
cities = ["北京", "东京", "华盛顿", "巴黎", "伦敦"]
animals = ["猫", "狗", "老虎", "狮子"]
food = ["苹果", "香蕉", "米饭", "面包"]

# 手动设置语义相近的词有相似的向量
word_vectors = {}

# 国家词向量（在第0维上聚集）
for country in countries:
    vec = np.random.randn(embedding_dim) * 0.3
    vec[0] += 1.0  # 国家在第0维正值
    word_vectors[country] = vec

# 城市词向量（在第0维上聚集，但靠近对应国家）
for city, country in zip(cities, countries):
    vec = word_vectors[country].copy()
    vec[1] += 1.0  # 城市在第1维有特征
    word_vectors[city] = vec

# 动物词向量（在第2维上聚集）
for animal in animals:
    vec = np.random.randn(embedding_dim) * 0.3
    vec[2] += 1.0
    word_vectors[animal] = vec

# 食物词向量（在第3维上聚集）
for f in food:
    vec = np.random.randn(embedding_dim) * 0.3
    vec[3] += 1.0
    word_vectors[f] = vec

print("\n【示例6：词向量聚类效果】")
print("\n国家 vs 动物 相似度（应该较低）：")
print(f"  中国 vs 猫: {cosine_similarity(word_vectors['中国'], word_vectors['猫']):.3f}")
print(f"  日本 vs 狗: {cosine_similarity(word_vectors['日本'], word_vectors['狗']):.3f}")

print("\n国家 vs 城市 相似度（应该较高）：")
print(f"  中国 vs 北京: {cosine_similarity(word_vectors['中国'], word_vectors['北京']):.3f}")
print(f"  日本 vs 东京: {cosine_similarity(word_vectors['日本'], word_vectors['东京']):.3f}")
print(f"  法国 vs 巴黎: {cosine_similarity(word_vectors['法国'], word_vectors['巴黎']):.3f}")

# ============================================================
# 第七部分：PyTorch中的词嵌入层
# ============================================================
print("\n" + "=" * 70)
print("第七部分：PyTorch中的词嵌入层")
print("=" * 70)

print("""
【nn.Embedding - PyTorch的词嵌入层】

在PyTorch中，词嵌入就是一个查找表（Lookup Table）：
输入词的ID → 输出对应的向量
""")

# 创建embedding层
vocab_size = 1000
embedding_dim = 128

embedding_layer = nn.Embedding(
    num_embeddings=vocab_size,  # 词典大小
    embedding_dim=embedding_dim,  # 向量维度
    padding_idx=0  # 填充词的ID，其向量保持为0
)

print(f"\n【示例7：PyTorch Embedding层】")
print(f"Embedding层: {embedding_layer}")
print(f"权重矩阵形状: {embedding_layer.weight.shape}")
print(f"  = [vocab_size, embedding_dim] = [{vocab_size}, {embedding_dim}]")

# 使用示例
word_ids = torch.LongTensor([1, 5, 10, 42])  # 4个词的ID
word_vectors = embedding_layer(word_ids)

print(f"\n输入词ID: {word_ids}")
print(f"输出词向量: {word_vectors.shape}")
print(f"  = [batch_size, embedding_dim] = [4, {embedding_dim}]")

print(f"\n词ID 1 的向量（前5维）: {word_vectors[0, :5].detach().numpy().round(4)}")
print(f"词ID 5 的向量（前5维）: {word_vectors[1, :5].detach().numpy().round(4)}")

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 70)
print("总结与实践")
print("=" * 70)

print("""
【本课核心知识点】
✅ 文本预处理：清洗 → 分词 → 去停用词
✅ 中文分词：比英文复杂，需要专门工具（jieba/HanLP）
✅ One-hot编码：简单但无效，维度灾难
✅ 词嵌入：稠密向量，语义相似度可计算
✅ Word2Vec：CBOW和Skip-gram两种训练方式
✅ 词向量算术：King - Man + Woman ≈ Queen
✅ PyTorch：nn.Embedding直接可用

【词嵌入演变】
Word2Vec (2013) → GloVe (2014) → FastText (2016) → 
ELMo (2018) → BERT/GPT (2018+) → 上下文相关词向量

【课后实践】
1. 安装jieba，用真实分词工具处理中文文本
2. 用gensim训练Word2Vec模型，观察词向量效果
3. 尝试词向量算术：国家-首都、动物-分类等
4. 可视化词向量（用t-SNE降维到2D观察聚类）
""")

# 安装提示
print("\n【需要安装的库】")
print("pip install jieba gensim numpy matplotlib")

print("\n【下节课预告】")
print("Transformer架构 - Attention is All You Need！")
print("- Self-Attention机制")
print("- 多头注意力")
print("- 位置编码")
print("- BERT和GPT的核心")

print("\n" + "=" * 70)
print("第15课完成！欢迎来到NLP世界！🎉")
print("=" * 70)
