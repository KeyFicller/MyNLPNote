# 第16课：Transformer架构

## 学习目标
- [x] 理解为什么需要Transformer（RNN的局限）
- [x] 掌握Self-Attention机制
- [x] 理解Multi-Head Attention
- [x] 理解Position Encoding
- [x] 掌握Transformer Encoder结构

---

## 1. 为什么需要Transformer？

### 1.1 RNN/LSTM的局限

| 问题 | 描述 | Transformer解决方案 |
|-----|------|-------------------|
| 串行计算 | h1 → h2 → h3，无法并行 | ✅ 完全并行 |
| 长距离依赖 | cat...was 距离远，信息衰减 | ✅ Attention直连 |
| 梯度问题 | 长序列梯度消失/爆炸 | ✅ 残差连接+LayerNorm |

### 1.2 Transformer的革命性

```
2017年 Google 论文《Attention is All You Need》

核心洞察："Attention机制本身就足够强大，不需要RNN！"
```

**效果对比**：
- 训练速度：比LSTM快 **10-100倍**
- 长距离依赖：所有位置距离都是 **O(1)**
- 准确率：翻译任务SOTA

---

## 2. Self-Attention 自注意力

### 2.1 核心思想

"句子中每个词都应该关注其他相关的词"

```
"The animal didn't cross the street because it was too tired"
                                                    ↑
                                                  "it"指什么？

Self-Attention学习：
- "it" 关注 "animal" (高度相关)
- "it" 不关注 "street" (不相关)
```

### 2.2 Query-Key-Value 机制

类比信息检索：

| 角色 | 类比 | 作用 |
|-----|------|------|
| **Query** | 搜索关键词 | 我要找什么信息？ |
| **Key** | 文档索引 | 我有什么信息？ |
| **Value** | 文档内容 | 信息的内容是什么？ |

### 2.3 Scaled Dot-Product Attention

```python
import torch.nn.functional as F
import math

def attention(Q, K, V, mask=None):
    """
    Q, K, V: [batch, seq_len, d_k]
    """
    # 1. 计算相似度: Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    # 2. 缩放（防止softmax梯度消失）
    scores = scores / math.sqrt(d_k)
    
    # 3. Mask（可选）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 4. Softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # 5. 加权求和
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights
```

### 2.4 图示

```
Input: [我, 爱, 自然, 语言, 处理]

        ┌──────────────────────────────────┐
        ↓                                  ↓
Q = [我Q, 爱Q, 自然Q, 语言Q, 处理Q]
K = [我K, 爱K, 自然K, 语言K, 处理K]  ← 互相计算相似度
V = [我V, 爱V, 自然V, 语言V, 处理V]
        ↓                                  ↓
Attention(Q, K, V) = 融合上下文的表示
```

---

## 3. Multi-Head Attention 多头注意力

### 3.1 为什么需要多头？

"一句话可以从不同角度理解"

```
"Apple is looking at buying U.K. startup for $1 billion"

Head 1: 实体识别 → 关注 "Apple", "U.K."
Head 2: 动作识别 → 关注 "buying"  
Head 3: 金额识别 → 关注 "$1 billion"
Head 4: 语法关系 → 关注词间依存
```

### 3.2 多头机制

```python
# 1. 分头: [batch, seq, d_model] → [batch, num_heads, seq, d_k]
Q = Q.view(batch, -1, num_heads, d_k).transpose(1, 2)

# 2. 每头独立做Attention
head_outputs = [attention(Q_h, K_h, V_h) for h in range(num_heads)]

# 3. 拼接: [batch, num_heads, seq, d_k] → [batch, seq, d_model]
output = concat(head_outputs)

# 4. 线性变换融合
output = Linear(output)
```

### 3.3 参数设置

| 模型 | d_model | num_heads | d_k |
|-----|---------|-----------|-----|
| Transformer-base | 512 | 8 | 64 |
| Transformer-large | 1024 | 16 | 64 |
| BERT-base | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |

---

## 4. Position Encoding 位置编码

### 4.1 问题

Attention是**位置无关**的：
```
"我爱你" 和 "你爱我"  →  Attention输出每个词的形状相同
但意思完全不同！
```

### 4.2 正弦位置编码

```python
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :]
```

### 4.3 优点

- ✅ 每个位置有独特的编码
- ✅ 可以外推到更长的序列
- ✅ 相邻位置编码相似（平滑过渡）

---

## 5. Transformer Encoder

### 5.1 单Encoder层结构

```
Input
  ↓
[Multi-Head Self-Attention]
  ↓
Add & Norm (残差连接 + LayerNorm)
  ↓
[Feed-Forward Network]
  ↓
Add & Norm (残差连接 + LayerNorm)
  ↓
Output
```

### 5.2 代码实现

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-Attention with residual
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FF with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x
```

### 5.3 Layer Normalization vs Batch Normalization

| 特性 | BatchNorm | LayerNorm |
|-----|-----------|-----------|
| 归一化维度 | 跨batch，同特征 | 跨特征，同样本 |
| NLP适用性 | ❌ batch大小不一 | ✅ 适合变长序列 |
| 使用位置 | CNN中常用 | Transformer/RNN中常用 |

---

## 6. 完整Transformer模型

### 6.1 结构图

```
Input Tokens
    ↓
[Embedding] + [Positional Encoding]
    ↓
[Encoder Layer] × N
    ↓
[Global Average Pooling]
    ↓
[Classifier]
    ↓
Output (分类分数/生成token)
```

### 6.2 参数规模

| 模型 | 层数 | 参数量 | 用途 |
|-----|------|-------|------|
| Transformer-base | 6 | 65M | 翻译/基础任务 |
| Transformer-large | 6 | 213M | 翻译/SOTA |
| BERT-base | 12 | 110M | 理解任务 |
| BERT-large | 24 | 340M | 理解任务 |
| GPT-3 | 96 | 175B | 生成任务 |

---

## 7. Attention可视化

### 7.1 Attention权重矩阵

```
句子："我爱自然语言处理"

       我    爱   自然   语言   处理
我   [0.3, 0.4, 0.1,  0.1,  0.1 ]
爱   [0.3, 0.2, 0.3,  0.1,  0.1 ]
自然 [0.1, 0.1, 0.3,  0.4,  0.1 ]
语言 [0.1, 0.1, 0.2,  0.3,  0.3 ]
处理 [0.1, 0.1, 0.1,  0.3,  0.4 ]

观察：
- 对角线较高（词关注自身）
- "自然"和"语言"互相关注度高
- "我"和"爱"互相关注度高
```

---

## 8. 今日速查表

### 8.1 Attention计算

```python
# Scaled Dot-Product Attention
def attention(Q, K, V):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn
```

### 8.2 Transformer层

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Self-attention + residual
        x = self.norm1(x + self.attn(x, x, x))
        # FF + residual
        x = self.norm2(x + self.ff(x))
        return x
```

### 8.3 超参数建议

| 参数 | 小模型 | 中等 | 大模型 |
|-----|-------|------|-------|
| d_model | 256-512 | 512-768 | 768-1024+ |
| num_heads | 4-8 | 8-12 | 12-16+ |
| d_ff | 1024-2048 | 2048-3072 | 3072-4096+ |
| num_layers | 2-4 | 4-8 | 8-24+ |

---

## 9. Transformer vs RNN 对比

| 特性 | RNN/LSTM | Transformer |
|-----|---------|-------------|
| 并行性 | ❌ 串行 | ✅ 完全并行 |
| 训练速度 | ❌ 慢 | ✅ 快10x+ |
| 长距离依赖 | ❌ 困难 | ✅ 等效 |
| 显存占用 | ✅ 低 | ❌ 高(Attention O(n²)) |
| 位置感知 | ✅ 天然 | ✅ 需位置编码 |

---

## 10. 下节课预告

**BERT和GPT - 预训练模型**

- BERT: 双向Encoder，适合理解任务
- GPT: 单向Decoder，适合生成任务
- 预训练 + 微调范式
- Masked Language Model

---

*学习日期：2026-05-24*  
*Attention is All You Need!*
