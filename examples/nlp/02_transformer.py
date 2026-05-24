#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第16课：Transformer架构 - Attention is All You Need!
======================================================
2017年Google论文《Attention is All You Need》彻底改变了NLP
本课程学习Transformer的核心机制：
- Self-Attention：自注意力机制
- Multi-Head Attention：多头注意力
- Position Encoding：位置编码
- Encoder-Decoder结构
- 代码实现与可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

print("=" * 70)
print("第16课：Transformer架构 - Attention is All You Need!")
print("=" * 70)

# 设置随机种子
torch.manual_seed(42)

# ============================================================
# 第一部分：为什么需要Transformer？
# ============================================================
print("\n" + "=" * 70)
print("第一部分：为什么需要Transformer？")
print("=" * 70)

print("""
【RNN/LSTM的局限】
问题1：串行计算，无法并行
  h1 → h2 → h3 → h4 → h5
  必须等h1算完才能算h2，速度慢！

问题2：长距离依赖困难
  "The cat, which was sitting on the mat, ... , was hungry"
  cat 和 was 距离很远，信息传递困难

问题3：梯度消失/爆炸
  反向传播经过多个时间步，梯度衰减或爆炸

【Transformer的革命性创新】
✅ 完全基于Attention，无需RNN
✅ 可以并行计算，速度快10x+
✅ 长距离依赖同样高效
✅ 2017年后成为NLP绝对主流
""")

# ============================================================
# 第二部分：Self-Attention 原理
# ============================================================
print("\n" + "=" * 70)
print("第二部分：Self-Attention 自注意力机制")
print("=" * 70)

print("""
【核心思想】
"一句话中，每个词都应该关注句子中其他相关的词"

示例：
"The animal didn't cross the street because it was too tired"
                                                ↑
                                              "it"指什么？

Self-Attention会让模型学会：
- "it"高度关注"animal"（代词关注名词）
- "it"低度关注"street"（不相关）

【数学原理：Query-Key-Value】
想象一个信息检索系统：
- Query（查询）：我要找什么信息？
- Key（键）：我有什么信息？
- Value（值）：信息的内容是什么？

注意力分数 = Query · Key 的点积（相似度）
""")

# 实现Scaled Dot-Product Attention
print("\n【示例1：Scaled Dot-Product Attention】")

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: [batch, seq_len, d_k] - Query
    K: [batch, seq_len, d_k] - Key  
    V: [batch, seq_len, d_v] - Value
    """
    d_k = Q.size(-1)
    
    # 1. 计算相似度分数: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, seq_len, seq_len]
    
    # 2. 缩放（防止softmax梯度消失）
    scores = scores / math.sqrt(d_k)
    
    # 3. 可选：应用mask（如padding位置）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax得到注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和: attention_weights @ V
    output = torch.matmul(attention_weights, V)
    
    return output, attention_weights

# 演示Attention计算
seq_len = 4
d_k = 8

# 模拟输入（4个词，每个词8维向量）
Q = torch.randn(1, seq_len, d_k)
K = torch.randn(1, seq_len, d_k)
V = torch.randn(1, seq_len, d_k)

print(f"输入形状:")
print(f"  Q (Query): {Q.shape} - 我关注什么？")
print(f"  K (Key):   {K.shape} - 我有什么信息？")
print(f"  V (Value): {V.shape} - 信息内容是什么？")

output, attention_weights = scaled_dot_product_attention(Q, K, V)

print(f"\n输出:")
print(f"  Attention权重: {attention_weights.shape}")
print(f"    = [batch, seq_len, seq_len] = [1, {seq_len}, {seq_len}]")
print(f"    表示每个词对其他词的关注程度")

print(f"\n  Attention权重矩阵示例：")
weights = attention_weights[0].detach().numpy()
for i in range(seq_len):
    print(f"    词{i}对其他词的关注: {weights[i].round(3)}")

print(f"\n  输出: {output.shape} - 融合上下文信息的新表示")

# ============================================================
# 第三部分：Multi-Head Attention
# ============================================================
print("\n" + "=" * 70)
print("第三部分：Multi-Head Attention 多头注意力")
print("=" * 70)

print("""
【为什么需要多头？】
"一句话可以从不同角度理解"

示例：
"Apple is looking at buying U.K. startup for $1 billion"

不同"头"关注不同方面：
- Head 1: 关注实体识别 (Apple, U.K.)
- Head 2: 关注动作 (buying)
- Head 3: 关注金额 ($1 billion)
- Head 4: 关注语法结构

【多头机制】
1. 将向量分成h份（如8个头）
2. 每个头独立做Attention
3. 拼接所有头的结果
4. 线性变换融合
""")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换矩阵
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 1. 线性变换
        Q = self.W_Q(Q)  # [batch, seq, d_model]
        K = self.W_K(K)
        V = self.W_V(V)
        
        # 2. 分成多个头 [batch, seq, d_model] -> [batch, num_heads, seq, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. 每个头做Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)  # [batch, num_heads, seq, d_k]
        
        # 4. 拼接所有头 [batch, num_heads, seq, d_k] -> [batch, seq, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 5. 最终线性变换
        output = self.W_O(output)
        
        return output, attention_weights

print("\n【示例2：Multi-Head Attention】")
d_model = 512
num_heads = 8

mha = MultiHeadAttention(d_model, num_heads)

seq_len = 10
X = torch.randn(1, seq_len, d_model)  # 输入序列

output, weights = mha(X, X, X)

print(f"参数:")
print(f"  d_model: {d_model} (模型维度)")
print(f"  num_heads: {num_heads} (注意力头数)")
print(f"  d_k: {d_model // num_heads} (每个头的维度)")

print(f"\n输入形状: {X.shape}")
print(f"  = [batch_size, seq_len, d_model]")

print(f"\n输出:")
print(f"  输出形状: {output.shape}")
print(f"  Attention权重: {weights.shape}")
print(f"    = [batch, num_heads, seq_len, seq_len]")
print(f"    = [1, {num_heads}, {seq_len}, {seq_len}]")

print(f"\n多头优势:")
print(f"  - 每个头{d_model//num_heads}维，并行计算{num_heads}个不同的注意力模式")
print(f"  - 最后拼接成{d_model}维，融合多视角信息")

# ============================================================
# 第四部分：Position Encoding 位置编码
# ============================================================
print("\n" + "=" * 70)
print("第四部分：Position Encoding 位置编码")
print("=" * 70)

print("""
【问题】
Attention是"位置无关"的：
  "我爱你" 和 "你爱我" 经过Attention后的每个词向量形状相同
  但意思完全不同！模型需要知道每个词的位置信息

【解决方案】
给每个词加上位置信息：
  词向量 = 词嵌入 + 位置编码

【正弦位置编码】
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

- 每个位置有独特的编码
- 可以处理比训练时更长的序列
- 相邻位置编码相似（平滑过渡）
""")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # sin给偶数索引，cos给奇数索引
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为buffer（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # 截取需要的位置编码
        return x + self.pe[:, :x.size(1), :]

print("\n【示例3：位置编码】")

pe_layer = PositionalEncoding(d_model=64, max_len=100)

# 可视化位置编码
seq_len = 20
d_vis = 8

pe_sample = pe_layer.pe[0, :seq_len, :d_vis].numpy()

print(f"位置编码形状: [{seq_len}, {d_vis}]")
print(f"前5个位置的前{d_vis}维：")
for i in range(min(5, seq_len)):
    print(f"  位置{i:2d}: {pe_sample[i].round(3)}")

print(f"\n位置编码特性:")
print(f"  - 每个位置有独特的编码模式")
print(f"  - 相似位置的编码也相似")
print(f"  - 可以外推到更长的序列")

# 可视化位置编码的相似性
print(f"\n位置编码相似度（余弦相似度）：")
for i in [0, 5, 10]:
    if i < seq_len - 1:
        sim = np.dot(pe_sample[i], pe_sample[i+1]) / (np.linalg.norm(pe_sample[i]) * np.linalg.norm(pe_sample[i+1]))
        print(f"  位置{i} vs 位置{i+1}: {sim:.3f}")

# ============================================================
# 第五部分：完整的Transformer Encoder层
# ============================================================
print("\n" + "=" * 70)
print("第五部分：Transformer Encoder层")
print("=" * 70)

class TransformerEncoderLayer(nn.Module):
    """
    标准Transformer Encoder层
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # 1. Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # 2. Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 子层1: Self-Attention with Add & Norm
        # 残差连接: x + Attention(x)
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 子层2: Feed-Forward with Add & Norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

print("\n【示例4：Transformer Encoder层】")

# 小规模的Encoder层演示
d_model = 128
num_heads = 8
d_ff = 256  # Feed-Forward层的中间维度

encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)

# 输入序列
batch_size = 2
seq_len = 5
x = torch.randn(batch_size, seq_len, d_model)

print(f"Encoder层参数:")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_ff: {d_ff} (FFN中间层，通常是4×d_model)")

print(f"\n输入: {x.shape}")
output = encoder_layer(x)
print(f"输出: {output.shape}")

print(f"\n计算流程:")
print(f"  1. Input: {x.shape}")
print(f"  2. ↓ Multi-Head Self-Attention")
print(f"  3. ↓ Add & LayerNorm (残差连接)")
print(f"  4. ↓ Feed-Forward Network")
print(f"  5. ↓ Add & LayerNorm (残差连接)")
print(f"  6. Output: {output.shape}")

# ============================================================
# 第六部分：完整的Transformer模型
# ============================================================
print("\n" + "=" * 70)
print("第六部分：完整Transformer模型架构")
print("=" * 70)

class SimpleTransformer(nn.Module):
    """
    简化的Transformer分类模型
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, 
                 num_layers=3, d_ff=512, num_classes=2, max_len=500):
        super().__init__()
        
        # 1. 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 2. 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 3. Transformer Encoder堆叠
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        # 4. 输出层
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.d_model = d_model
    
    def forward(self, x, mask=None):
        # 1. 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # 缩放
        x = self.pos_encoding(x)
        
        # 2. 通过所有Encoder层
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # 3. 全局平均池化（取序列平均）
        x = x.mean(dim=1)
        
        # 4. 分类
        return self.classifier(x)

print("\n【示例5：完整Transformer模型】")

vocab_size = 10000
d_model = 128
num_heads = 4
num_layers = 2
num_classes = 2  # 二分类

model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, 
                          d_ff=d_model*4, num_classes=num_classes)

print(f"模型配置:")
print(f"  Vocab Size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  Num Heads: {num_heads}")
print(f"  Num Layers: {num_layers}")
print(f"  Num Classes: {num_classes}")

total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数量: {total_params:,}")

# 测试前向传播
batch_size = 4
seq_len = 32
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"\n输入: {input_ids.shape}")
output = model(input_ids)
print(f"输出: {output.shape}")
print(f"  = [batch_size, num_classes] = [{batch_size}, {num_classes}]")

print(f"\n模型结构:")
print(f"  Input IDs → Embedding ({vocab_size}×{d_model} params)")
print(f"           ↓")
print(f"           + Positional Encoding")
print(f"           ↓")
print(f"           ×{num_layers} Transformer Encoder Layers")
print(f"           ↓")
print(f"           Global Average Pooling")
print(f"           ↓")
print(f"           Classifier ({d_model}×{num_classes} params)")
print(f"           ↓")
print(f"           Output (分类分数)")

# ============================================================
# 第七部分：Attention可视化概念
# ============================================================
print("\n" + "=" * 70)
print("第七部分：Attention可视化")
print("=" * 70)

print("""
【Attention权重解读】

示例句子："我爱自然语言处理"

Attention权重矩阵 [6×6]：
         我    爱   自然   语言   处理
      ┌────┬────┬────┬────┬────┬────┐
我    │0.3 │0.4 │0.1 │0.1 │0.05│0.05│  我关注"爱"最多
      ├────┼────┼────┼────┼────┼────┤
爱    │0.3 │0.2 │0.3 │0.1 │0.05│0.05│  "爱"关注"我"和"自然"
      ├────┼────┼────┼────┼────┼────┤
自然  │0.05│0.05│0.3 │0.4 │0.1 │0.1 │  "自然"关注"语言"
      ├────┼────┼────┼────┼────┼────┤
语言  │0.05│0.05│0.2 │0.3 │0.3 │0.1 │  "语言"关注前后词
      ├────┼────┼────┼────┼────┼────┤
处理  │0.05│0.05│0.1 │0.2 │0.3 │0.3 │  "处理"关注"语言"
      └────┴────┴────┴────┴────┴────┘

观察：
- "自然"和"语言"之间Attention分数高（词语关联）
- "我"和"爱"之间Attention分数高（主谓关系）
- 对角线值通常较高（词关注自身）
""")

# 生成一个示例Attention可视化
sentence = ["我", "爱", "自然", "语言", "处理"]
seq_len = len(sentence)

# 模拟一个合理的Attention矩阵
attention_demo = torch.eye(seq_len) * 0.3  # 对角线
attention_demo[0, 1] = 0.4  # "我"关注"爱"
attention_demo[1, 0] = 0.3  # "爱"关注"我"
attention_demo[1, 2] = 0.3  # "爱"关注"自然"
attention_demo[2, 3] = 0.4  # "自然"关注"语言"
attention_demo[3, 2] = 0.2  # "语言"关注"自然"
attention_demo[3, 4] = 0.3  # "语言"关注"处理"
attention_demo[4, 3] = 0.3  # "处理"关注"语言"

# 归一化
attention_demo = F.softmax(attention_demo, dim=-1)

print("\n模拟的Attention权重矩阵：")
print("     ", "  ".join([f"{w:>4s}" for w in sentence]))
for i, word in enumerate(sentence):
    weights = attention_demo[i].numpy()
    print(f"{word:>4s} |", "  ".join([f"{w:.2f}" for w in weights]))

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 70)
print("总结与实践")
print("=" * 70)

print("""
【本课核心知识点】
✅ Self-Attention: Q·K^T计算相似度，加权V
✅ Multi-Head: 并行多个注意力视角
✅ Position Encoding: 正弦/余弦给位置信息
✅ Encoder: Attention + FFN + LayerNorm + 残差
✅ Transformer: Embedding + PE + N×Encoder + Classifier

【Transformer vs RNN对比】

特性          RNN/LSTM              Transformer
──────────────────────────────────────────────────
并行性        ❌ 串行              ✅ 完全并行
长依赖        ❌ 困难              ✅ 等效
速度          ❌ 慢                ✅ 快10x+
位置感知      ✅ 天然              ✅ 需要位置编码
训练稳定性    ⚠️ 梯度问题          ✅ 稳定

【关键超参数】
- d_model: 模型维度（常用512, 768, 1024）
- num_heads: 注意力头数（常用8, 12, 16）
- d_ff: FFN中间维度（常用4×d_model）
- num_layers: Encoder层数（常用6-24层）

【下节课预告】
BERT和GPT - 基于Transformer的预训练模型！
- BERT: 双向编码器，适合理解任务
- GPT: 单向解码器，适合生成任务
- 预训练 + 微调范式
""")

print("\n【课后实践】")
print("1. 修改num_heads，观察对模型和效果的影响")
print("2. 添加Dropout和更好的正则化")
print("3. 实现Transformer Decoder（用于生成任务）")
print("4. 用真实文本数据训练情感分类器")

print("\n【推荐阅读】")
print("- 论文: Attention is All You Need (2017)")
print("- The Illustrated Transformer (博客)")
print("- Hugging Face Transformers库文档")

print("\n" + "=" * 70)
print("第16课完成！Transformer架构掌握！🎉")
print("=" * 70)
