# 第17课：BERT与GPT预训练模型

## 学习目标
- [x] 理解预训练+微调范式
- [x] 掌握BERT的双向编码原理
- [x] 理解GPT的单向生成原理
- [x] 掌握Masked LM和Autoregressive LM的区别
- [x] 了解Hugging Face Transformers生态

---

## 1. 预训练范式 - NLP的游戏规则变革

### 1.1 2018年前的困境

**问题**：
- 每个NLP任务都要从头训练
- 标注数据昂贵（需要人工标注数万条）
- 小数据集上模型效果不好

### 1.2 预训练+微调范式

```
        大量无标注文本（如维基百科、书籍）
                    ↓
              【预训练阶段】
              学习通用的语言表示
                    ↓
              预训练模型（BERT/GPT）
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   【微调阶段】  【微调阶段】  【微调阶段】
      少量标注      少量标注      少量标注
        ↓           ↓           ↓
    情感分类    命名实体识别    问答系统
```

**优势**：
- ✅ 预训练阶段用海量无标注数据学习语言规律
- ✅ 微调阶段只需少量标注数据即可达到SOTA
- ✅ 一个预训练模型可应用到多个下游任务

### 1.3 迁移学习在NLP的成功

| 阶段 | 数据量 | 计算量 | 标注成本 |
|-----|-------|-------|---------|
| 预训练 | 数十亿词 | 大 | 几乎为零 |
| 微调 | 数千条 | 小 | 较小 |
| 从头训练 | 数十万条 | 中等 | 很高 |

---

## 2. BERT - Bidirectional Encoder Representations

### 2.1 核心思想

> "深层的双向表示对NLP任务至关重要"

**双向 vs 单向**：
```
单向（GPT）: 今天 [MASK] 气真好
             ↑ 只能看左边
            预测？

双向（BERT）: 今天 [MASK] 气真好
             ↑ 看左边"今天" + 看右边"气真好"
            预测"天"！
```

### 2.2 预训练任务1：Masked Language Model (MLM)

```
输入：今天 [MASK] 气真好，我想去 [MASK] 园玩
目标：       天              公

训练过程：
1. 随机mask 15%的词
2. 用Transformer编码器预测被mask的词
3. 损失函数：交叉熵（预测词的概率分布）
```

**15%中的80-10-10策略**：
- 80%：用[MASK]替换
- 10%：用随机词替换
- 10%：保持不变

目的：防止模型只在看到[MASK]时才学习，学会真正的双向表示。

### 2.3 预训练任务2：Next Sentence Prediction (NSP)

```
句子A：今天天气真好
句子B：我想去公园玩    → IsNext (是连续的)

句子A：今天天气真好
句子B：机器学习很有趣  → NotNext (不相关)
```

**注意**：RoBERTa等后续模型发现NSP效果有限，已去掉这个任务。

### 2.4 BERT架构

```python
# BERT输入表示 = Token Emb + Segment Emb + Position Emb

embedding = TokenEmbedding(input_ids) + \
            SegmentEmbedding(segment_ids) + \
            PositionEmbedding(positions)

# 通过多层Transformer Encoder
hidden_states = TransformerEncoder(embedding)

# MLM预测
mlm_logits = MLMHead(hidden_states)  # [batch, seq, vocab]

# 分类任务：取[CLS]token的输出
cls_output = hidden_states[:, 0, :]  # 第一个位置是[CLS]
prediction = Classifier(cls_output)  # [batch, num_classes]
```

### 2.5 BERT变体

| 模型 | 参数量 | 特点 |
|-----|-------|------|
| BERT-base | 110M | 12层，768维，12头 |
| BERT-large | 340M | 24层，1024维，16头 |
| RoBERTa | 355M | 优化训练，去掉NSP |
| ALBERT | 18M | 参数共享，轻量级 |
| DistilBERT | 66M | BERT蒸馏版，快60% |

---

## 3. GPT - Generative Pre-trained Transformer

### 3.1 核心思想

> "通过生成式预训练学习语言模型"

**自回归特性**：
```
给定：<sos>
预测："今天"

给定：<sos> 今天
预测："天气"

给定：<sos> 今天 天气
预测："真好"

给定：<sos> 今天 天气 真好
预测：<eos>
```

### 3.2 因果Mask（Causal Masking）

```
因果Mask（下三角）：
        位置1   位置2   位置3   位置4
位置1  [  0     -inf   -inf   -inf  ]  位置1只能看自己
位置2  [  0      0     -inf   -inf  ]  位置2能看1,2
位置3  [  0      0      0    -inf  ]  位置3能看1,2,3
位置4  [  0      0      0      0   ]  位置4能看1,2,3,4
```

**为什么需要因果mask？**
- 生成任务中，只能看到已生成的词
- 防止模型"偷看"未来的词

### 3.3 GPT架构

```python
# GPT使用Decoder-only架构

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers):
        # Token + 位置嵌入
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Transformer Decoder（带因果mask）
        self.transformer = nn.TransformerDecoder(...)
        
        # 输出投影（与token_emb共享权重）
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_emb.weight  # 权重共享
    
    def forward(self, input_ids):
        # 嵌入
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        # 因果mask
        causal_mask = generate_causal_mask(seq_len)
        
        # Transformer（自回归）
        x = self.transformer(x, mask=causal_mask)
        
        # 预测下一个token
        logits = self.lm_head(x)
        return logits
```

### 3.4 GPT演进

| 模型 | 时间 | 参数量 | 特点 |
|-----|------|-------|------|
| GPT-1 | 2018.06 | 117M | 首个生成式预训练 |
| GPT-2 | 2019.02 | 1.5B | 更大，生成质量惊人 |
| GPT-3 | 2020.05 | 175B | 巨大突破，few-shot |
| GPT-3.5 | 2022.03 | 未知 | Instruct优化 |
| ChatGPT | 2022.11 | 未知 | RLHF，对话能力 |
| GPT-4 | 2023.03 | 未知 | 多模态，推理强 |

---

## 4. BERT vs GPT 对比

### 4.1 架构对比

| 特性 | BERT | GPT |
|-----|------|-----|
| 方向 | 双向 | 单向（左到右） |
| 架构 | Encoder-only | Decoder-only |
| Mask | 随机Mask | 因果Mask |
| 预训练 | MLM（填空） | Autoregressive（接龙） |
| 训练效率 | 并行，快 | 串行，慢 |
| 适合任务 | 理解 | 生成 |

### 4.2 应用场景

| 任务 | 推荐模型 | 原因 |
|-----|---------|------|
| 文本分类 | BERT | 理解整体语义 |
| 命名实体识别 | BERT | 双向上下文定位实体 |
| 问答系统 | BERT | 问题+文章双向交互 |
| 语义相似度 | BERT/SBERT | 双向编码句子 |
| 文本生成 | GPT | 自回归生成能力 |
| 对话系统 | GPT | 自然的对话续写 |
| 代码生成 | GPT/Codex | 代码续写能力强 |
| 创意写作 | GPT | 开放生成能力 |

### 4.3 输入输出对比

**BERT（以情感分类为例）**：
```
输入：[CLS] 今天天气真好 [SEP]
         ↓
   BERT Encoder
         ↓
输出：[CLS_vector] → Classifier → 正面情感 (0.92)
      token2_vector
      token3_vector
      ...
```

**GPT（以文本生成为例）**：
```
输入：<bos> 今天天气
         ↓
   GPT Decoder
         ↓
输出：今天天气 真 好 啊 [SEP]
      ↑    ↑ ↑  ↑ ↑
      t1   t2 t3 t4 t5

      每个位置预测下一个token
```

---

## 5. 微调（Fine-tuning）

### 5.1 微调流程

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 加载预训练模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 2. 冻结底层（可选）
for param in model.bert.encoder.layer[:10].parameters():
    param.requires_grad = False

# 3. 准备优化器（学习率要小！）
optimizer = AdamW(model.parameters(), lr=2e-5)

# 4. 微调训练
for epoch in range(epochs):
    for batch in dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### 5.2 微调技巧

| 技巧 | 说明 | 适用场景 |
|-----|------|---------|
| **小学习率** | 2e-5 ~ 5e-5 | 通用 |
| **少轮数** | 2-4 epochs | 通用 |
| **分层学习率** | 顶层lr大，底层lr小 | 大数据集 |
| **冻结底层** | 只训练顶层 | 小数据集 |
| **早停** | 验证集不提升则停止 | 通用 |

### 5.3 特征提取（Feature Extraction）

如果不微调，也可以直接提取BERT的特征：

```python
from transformers import BertTokenizer, BertModel

# 加载预训练BERT
model = BertModel.from_pretrained('bert-base-chinese')
model.eval()

# 提取特征
text = "今天天气真好"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    
# outputs.last_hidden_state: [1, seq_len, 768]
# 可作为下游任务的输入特征
```

---

## 6. 预训练模型演进时间线

```
2018.06  GPT-1       117M    首个生成式预训练
2018.10  BERT        340M    双向编码，理解任务SOTA
2019.02  GPT-2       1.5B    更大，生成质量惊人
2019.10  RoBERTa     355M    BERT优化版
2019.10  ALBERT      18M     轻量化BERT
2020.05  GPT-3       175B    巨大突破，few-shot learning
2022.03  GPT-3.5     ?       InstructGPT，指令微调
2022.11  ChatGPT     ?       RLHF，对话能力质变
2023.02  LLaMA       7B-65B  Meta开源，推动社区发展
2023.03  GPT-4       ?       多模态，推理能力大幅提升
```

---

## 7. Hugging Face Transformers

### 7.1 生态介绍

| 库 | 功能 |
|-----|------|
| **transformers** | 预训练模型库（BERT/GPT/T5等） |
| **datasets** | 数据集库 |
| **tokenizers** | 快速分词器 |
| **accelerate** | 分布式训练 |
| **model hub** | 模型分享平台（50万+模型） |

### 7.2 基本使用流程

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. 加载Tokenizer和模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

# 2. 文本编码
inputs = tokenizer("今天天气真好", return_tensors="pt", padding=True, truncation=True)
# 返回：input_ids, attention_mask, token_type_ids

# 3. 模型推理
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)

print(f"预测结果: {prediction}")

# 4. 批量处理
texts = ["今天天气真好", "这部电影太烂了"]
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
```

### 7.3 常用模型名称

| 任务 | 推荐模型 |
|-----|---------|
| 中文理解 | bert-base-chinese, roberta-wwm-ext |
| 英文理解 | bert-base-uncased, roberta-base |
| 英文生成 | gpt2, gpt2-medium |
| 代码生成 | codeparrot/codeparrot |
| 多语言 | xlm-roberta-base |
| 长文本 | longformer-base-4096 |

---

## 8. 今日速查表

### 8.1 BERT使用模板

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 加载
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 编码
text = "今天天气真好"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

# 推理
outputs = model(**inputs)
logits = outputs.logits
```

### 8.2 GPT使用模板

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 生成
text = "Once upon a time"
inputs = tokenizer(text, return_tensors="pt")

# 自回归生成
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7
)

generated = tokenizer.decode(outputs[0])
```

### 8.3 微调模板

```python
from transformers import AdamW, get_linear_schedule_with_warmup

# 优化器（分层学习率）
optimizer = AdamW([
    {'params': model.bert.encoder.layer[-4:].parameters(), 'lr': 5e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

# 训练
model.train()
for epoch in range(3):  # 少轮数
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
```

---

## 9. 下节课预告

**Hugging Face Transformers实战**

- 完整的文本分类项目
- Tokenizer高级用法
- 数据预处理Pipeline
- 模型微调和评估
- 模型保存和部署

---

*学习日期：2026-05-24*  
*预训练模型时代已来临！*
