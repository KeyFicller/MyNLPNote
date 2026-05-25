# 中文文本摘要实战 - GPT 文本生成

**项目**: 新闻标题生成与文章摘要  
**技术栈**: GPT-2, PyTorch, Transformers  
**任务**: 文本生成（自回归语言建模）

---

## 1. 项目概览

### 1.1 什么是文本生成

与分类/标注任务不同，**文本生成**是"创作"任务：

| 任务 | 输入 | 输出 | 本质 |
|------|------|------|------|
| 分类 | 句子 | 1个标签 | **判断**（这是什么？） |
| 标注 | 句子 | N个标签 | **识别**（有哪些？） |
| 生成 | 句子 | 新句子 | **创作**（生成什么？） |

### 1.2 文本生成的应用场景

- **文章摘要**: 长文 → 简短摘要
- **标题生成**: 内容 → 吸引人的标题
- **对话生成**: 上文 → 下一句回复
- **翻译**: 源语言 → 目标语言
- **代码生成**: 需求 → 代码

---

## 2. 数据格式

### 2.1 文本对格式

```python
{
    "article": "阿里巴巴集团今日宣布...",
    "summary": "阿里云计划三年投资千亿"
}
```

### 2.2 拼接格式（用于GPT自回归）

```
输入序列: [文章] \n摘要： [摘要] <|endoftext|>

示例:
文章: 苹果公司发布iPhone 15...\n摘要：iPhone 15发布：采用USB-C接口

BERT Token IDs:
[CLS] 苹 果 公 司 发 布 ... [SEP] i P h o n e ... <|endoftext|>
  ↓                    ↓
  文章部分              摘要部分
  (mask=-100)          (计算loss)
```

### 2.3 Label Mask 策略

```python
# 完整序列
input_ids = [CLS] + article_tokens + [SEP] + summary_tokens + [EOS]

# labels: 和input_ids相同，但文章部分设为-100（忽略）
labels = [-100, ..., -100, summary_token_1, summary_token_2, ..., EOS]
           ↑                    ↑
        不计算loss            计算loss

# 原理：模型学习 P(摘要|文章)，文章是条件，不是预测目标
```

---

## 3. 核心代码解析

### 3.1 Dataset 处理

```python
class SummarizationDataset(Dataset):
    def __getitem__(self, idx):
        article = self.data[idx]["article"]
        summary = self.data[idx]["summary"]
        
        # 拼接: 文章 + 分隔符 + 摘要 + 结束符
        full_text = article + "\n摘要：" + summary + "<|endoftext|>"
        
        # Tokenize
        encoding = self.tokenizer(full_text, ...)
        input_ids = encoding['input_ids']
        
        # 找到分隔符位置，确定文章长度
        sep_text = article + "\n摘要："
        sep_encoding = self.tokenizer(sep_text, ...)
        article_len = len(sep_encoding['input_ids'])
        
        # 构建labels
        labels = input_ids.clone()
        labels[:article_len] = -100  # 文章部分mask
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
```

### 3.2 模型加载

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# GPT2是自回归语言模型
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 结构
Input → GPT2 Decoder (12层) → Language Model Head → Logits
                                    ↓
                              预测下一个token的概率
```

### 3.3 训练循环

```python
# 注意：GPT2LMHeadModel内部实现了语言建模loss
outputs = model(
    input_ids=input_ids,      # [batch, seq_len]
    attention_mask=attention_mask,
    labels=labels             # [batch, seq_len]，-100被忽略
)

loss = outputs.loss  # 自动计算 CrossEntropy
loss.backward()
```

---

## 4. 文本生成解码策略

### 4.1 Greedy Search（贪心搜索）

```python
# 每步选择概率最高的token
output = model.generate(
    input_ids,
    max_length=100,
    do_sample=False  # 贪心
)

# 特点
- 优点: 简单、确定、速度快
- 缺点: 容易重复、缺乏多样性

示例:
Step 1: [文章] → "今"
Step 2: [文章]今 → "天"
Step 3: [文章]今天 → "天"
Step 4: [文章]今天天 → "气"
...
输出: "今天天气真好"
```

### 4.2 Beam Search（束搜索）

```python
output = model.generate(
    input_ids,
    max_length=100,
    num_beams=4,        # 保留top-4候选
    num_return_sequences=1,
    early_stopping=True
)

# 特点
- 优点: 质量较高，考虑全局最优
- 缺点: 计算量大，可能过于"安全"

示例 (num_beams=2):
Step 1: 候选 ["今"(0.5), "昨"(0.3)]
Step 2: 扩展 ["今天"(0.4), "今明"(0.1), "昨天"(0.3), "昨今"(0.0)]
        保留top-2: ["今天", "昨天"]
Step 3: 继续扩展...
最终选择概率最高的完整序列
```

### 4.3 Temperature Sampling（温度采样）

```python
output = model.generate(
    input_ids,
    max_length=100,
    do_sample=True,
    temperature=0.7,  # 温度参数
    top_k=50,           # 只从top-50采样
    top_p=0.95        # 或累积概率95%内的token
)

# 温度控制随机性
temperature → 0: 接近贪心（确定性）
temperature → 1: 完全按概率分布采样（最随机）
temperature → ∞: 均匀随机

# Top-k: 只考虑概率最高的k个token
# Top-p (Nucleus): 考虑累积概率达到p的最小token集合

示例 (temperature=0.7):
原始分布: P("今")=0.5, P("昨")=0.3, P("明")=0.2
应用温度后: P'("今")=0.55, P'("昨")=0.28, P'("明")=0.17
更"尖锐"，倾向于选高概率词，但仍有随机性
```

### 4.4 策略对比

| 策略 | 适用场景 | 优缺点 |
|------|----------|--------|
| Greedy | 确定性任务（代码、事实） | 快但重复 |
| Beam | 高质量要求（翻译、摘要） | 好但慢 |
| Sampling | 创造性任务（故事、诗歌） | 多样但不稳 |
| Top-k/p | 平衡方案（通用生成） | 推荐默认 |

---

## 5. 生成优化技巧

### 5.1 避免重复

```python
output = model.generate(
    input_ids,
    no_repeat_ngram_size=2  # 禁止重复2-gram
)

# 防止生成 "今天今天天气" 这种重复
```

### 5.2 提前结束

```python
output = model.generate(
    input_ids,
    early_stopping=True  # 遇到EOS提前停止
)
```

### 5.3 长度控制

```python
output = model.generate(
    input_ids,
    min_length=10,   # 最少生成长度
    max_length=50    # 最大生成长度
)
```

---

## 6. 评估指标 - ROUGE

### 6.1 ROUGE 简介

**ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)：
- 基于n-gram重叠的评估方法
- 比较生成摘要和参考摘要的相似度

### 6.2 ROUGE-N

```python
# ROUGE-1: unigram重叠
generated = ["阿里云", "投资", "千亿", "云", "基础设施"]
reference = ["阿里云", "计划", "投资", "千亿", "建设", "云", "基础设施"]

重叠: ["阿里云", "投资", "千亿", "云", "基础设施"] = 5个
ROUGE-1 Recall = 5 / 7 = 0.71
ROUGE-1 Precision = 5 / 5 = 1.0
ROUGE-1 F1 = 2*0.71*1.0 / (0.71+1.0) = 0.83

# ROUGE-2: bigram重叠
generated_bigrams = ["阿里云投资", "投资千亿", "千亿云", "云基础设施"]
reference_bigrams = ["阿里云计划", "计划投资", "投资千亿", "千亿建设", ...]

重叠: ["投资千亿"] = 1个
ROUGE-2 = ...

# ROUGE-L: 最长公共子序列（考虑词序）
```

### 6.3 计算代码

```python
from rouge import Rouge

rouge = Rouge()
scores = rouge.get_scores(
    hyps=generated_summary,  # 生成摘要
    refs=reference_summary   # 参考摘要
)

# 输出示例
{
    'rouge-1': {'r': 0.71, 'p': 1.0, 'f': 0.83},
    'rouge-2': {'r': 0.25, 'p': 0.33, 'f': 0.29},
    'rouge-l': {'r': 0.68, 'p': 0.95, 'f': 0.79}
}
```

---

## 7. 模型选择

### 7.1 GPT 系列

| 模型 | 参数量 | 特点 |
|------|--------|------|
| GPT-2 | 117M-1.5B | 基础自回归模型 |
| GPT-3 | 175B | 强大的少样本学习能力 |
| GPT-4 | 未公开 | 多模态、更强的推理 |

### 7.2 专门的中文生成模型

| 模型 | 架构 | 适用场景 |
|------|------|----------|
| ChatGLM | Decoder + Prefix | 中文对话、生成 |
| Baichuan | Decoder | 中文通用生成 |
| Qwen | Decoder | 多语言、代码 |
| T5/mT5 | Encoder-Decoder | 多语言翻译、摘要 |
| BART | Encoder-Decoder | 英文摘要、去噪 |

### 7.3 Encoder-Decoder vs Decoder-only

```
Encoder-Decoder (T5/BART):
  Encoder: 理解输入（双向注意力）
            ↓
  Decoder: 生成输出（单向注意力）
  
  优点: 编码和解码分离，结构清晰
  适合: 翻译、摘要等Seq2Seq任务

Decoder-only (GPT):
  统一架构，自回归生成
  
  优点: 简单、统一预训练目标
  适合: 通用文本生成、对话
```

---

## 8. 关键区别总结

### 8.1 三种NLP任务对比

```
┌──────────┬─────────────┬──────────────┬──────────────┐
│   任务   │    输入     │     输出     │    模型      │
├──────────┼─────────────┼──────────────┼──────────────┤
│  分类    │    句子     │   1个标签    │  SequenceCls │
│  标注    │    句子     │  N个标签/字  │   TokenCls   │
│  生成    │  句子+提示   │   新句子     │   LM Head    │
└──────────┴─────────────┴──────────────┴──────────────┘
```

### 8.2 Loss 计算对比

```python
# 分类
logits = model(input_ids)  # [batch, num_classes]
loss = CrossEntropy(logits, labels)  # labels: [batch]

# 标注
logits = model(input_ids)  # [batch, seq_len, num_classes]
loss = CrossEntropy(logits.view(-1, num_classes), labels.view(-1))  # labels: [batch, seq_len]

# 生成
# 内部实现: 每个位置预测下一个token
# labels: [batch, seq_len]
# loss: mean over positions where labels != -100
loss = model(input_ids, labels=labels).loss
```

---

## 9. 进阶方向

### 9.1 使用更大的中文模型

```python
# ChatGLM-6B
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

# 更好的中文理解能力
```

### 9.2 使用Seq2Seq架构（T5/BART）

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('t5-base')

# T5的生成方式不同：
# Encoder处理输入 → Decoder逐个生成输出token
# 更适合摘要任务
```

### 9.3 使用真实数据集

- **LCSTS**: 大规模中文短文本摘要数据集
- **NLPCC**: 自然语言处理与中文计算会议数据集
- **CNewSum**: 中文新闻摘要数据集

### 9.4 强化学习优化（RLHF）

```
1. 预训练语言模型
2. 有监督微调（SFT）
3. 奖励模型训练（人类反馈）
4. 强化学习优化（PPO）

结果: 模型生成更符合人类偏好的文本
```

---

## 10. 核心代码模板

```python
# 1. 数据准备
data = [{"article": "...", "summary": "..."}, ...]

# 2. Dataset
class SummarizationDataset(Dataset):
    def __getitem__(self, idx):
        text = article + "\n摘要：" + summary + "<|endoftext|>"
        encoding = tokenizer(text, ...)
        
        # mask文章部分
        labels = encoding['input_ids'].clone()
        labels[:article_len] = -100
        
        return {
            'input_ids': encoding['input_ids'],
            'labels': labels
        }

# 3. 模型
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 4. 训练
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()

# 5. 生成
output = model.generate(
    input_ids,
    max_length=100,
    num_beams=4,
    no_repeat_ngram_size=2,
    early_stopping=True
)
```

---

**实践项目文件**: `examples/nlp/09_text_summarization.py`

**运行命令**:
```bash
python examples/nlp/09_text_summarization.py
```
