# 第18课：Hugging Face Transformers 实战

## 学习目标
- [x] 掌握 Hugging Face Pipeline 的使用
- [x] 理解 Tokenizer 的工作原理
- [x] 学会加载和使用预训练模型
- [x] 掌握模型微调的完整流程
- [x] 能够使用 BERT 进行分类，使用 GPT 进行生成

---

## 1. Hugging Face 生态系统

### 1.1 核心库介绍

| 库名 | 功能 |
|------|------|
| **transformers** | 预训练模型库（BERT/GPT/T5等） |
| **datasets** | 数据集库 |
| **tokenizers** | 快速分词器（Rust实现） |
| **accelerate** | 分布式训练 |
| **evaluate** | 评估指标 |

### 1.2 Model Hub

- **50万+** 预训练模型
- 支持多种框架（PyTorch、TensorFlow、JAX）
- 社区贡献，持续更新

**访问地址**: https://huggingface.co/models

---

## 2. Pipeline - 开箱即用

### 2.1 什么是 Pipeline？

Pipeline 封装了完整的 NLP 流程：

```
原始文本
    ↓
Tokenizer 编码
    ↓
模型推理
    ↓
结果解码
    ↓
格式化输出
```

### 2.2 支持的 Pipeline 类型

| Pipeline 类型 | 用途 | 示例模型 |
|--------------|------|---------|
| `sentiment-analysis` | 情感分析 | `distilbert-base-uncased-finetuned-sst-2-english` |
| `text-classification` | 文本分类 | `bert-base-chinese` |
| `token-classification` | 序列标注（NER） | `dslim/bert-base-NER` |
| `question-answering` | 问答 | `distilbert-base-cased-distilled-squad` |
| `text-generation` | 文本生成 | `gpt2` |
| `translation` | 翻译 | `Helsinki-NLP/opus-mt-zh-en` |
| `summarization` | 摘要 | `facebook/bart-large-cnn` |
| `fill-mask` | 填空（MLM） | `bert-base-uncased` |

### 2.3 Pipeline 使用示例

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 文本生成
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time")
# [{'generated_text': 'Once upon a time, there was...'}]

# 命名实体识别
ner = pipeline("ner", model="dslim/bert-base-NER")
result = ner("My name is John and I work at Google")
# [{'entity': 'B-PER', 'word': 'John'}, {'entity': 'B-ORG', 'word': 'Google'}]

# 问答
qa = pipeline("question-answering")
context = "The capital of France is Paris."
question = "What is the capital of France?"
result = qa(question=question, context=context)
# {'answer': 'Paris', 'score': 0.99}
```

---

## 3. Tokenizer 详解

### 3.1 Tokenizer 的作用

```
原始文本 → Tokenizer → 模型输入

"今天天气真好"
    ↓
分词: ['今', '天', '天', '气', '真', '好'] 或子词 [今天, 天气, 真好]
    ↓
加特殊token: ['[CLS]', '今', '天', '天', '气', '真', '好', '[SEP]']
    ↓
转ID: [101, 1234, 5678, 5678, 9012, 3456, 7890, 102]
    ↓
Padding: [101, 1234, 5678, 5678, 9012, 3456, 7890, 102, 0, 0, 0, 0]
    ↓
模型输入
```

### 3.2 分词算法

| 算法 | 特点 | 代表模型 |
|------|------|---------|
| **WordPiece** | 子词级别，平衡词表大小和OOV | BERT, DistilBERT |
| **BPE** (Byte Pair Encoding) | 从字符开始合并，简单高效 | GPT-2, RoBERTa |
| **Unigram** | 基于概率，更灵活 | ALBERT, T5 |
| **SentencePiece** | 语言无关，处理中文好 | XLNet, ALBERT |

### 3.3 Tokenizer 核心参数

```python
tokenizer(
    text,                          # 输入文本或列表
    max_length=512,                # 最大序列长度
    padding=True,                  # 是否填充
                                   # True: 填充到batch中最长
                                   # 'max_length': 填充到max_length
                                   # False: 不填充
    truncation=True,               # 是否截断
    return_tensors='pt',           # 返回格式: 'pt', 'np', 'tf'
    add_special_tokens=True,        # 是否加[CLS], [SEP]等
    return_attention_mask=True,    # 返回attention_mask
    return_token_type_ids=True,     # 返回token_type_ids (BERT)
)

# 返回结果
{
    'input_ids': tensor([[101, 1234, ...]]),       # token IDs
    'attention_mask': tensor([[1, 1, 1, ...]]),    # 1=真实token, 0=padding
    'token_type_ids': tensor([[0, 0, 0, ...]]),    # 句子A=0, 句子B=1 (BERT)
}
```

### 3.4 Tokenizer 使用示例

```python
from transformers import BertTokenizer

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 单条文本编码
encoded = tokenizer(
    "今天天气真好",
    max_length=512,
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# 批量编码
texts = ["今天天气真好", "这部电影太烂了"]
encoded = tokenizer(
    texts,
    max_length=512,
    padding=True,
    truncation=True,
    return_tensors='pt'
)
# encoded['input_ids'].shape: [2, 512]

# 解码
original = tokenizer.decode(encoded['input_ids'][0])
# '[CLS] 今天天气真好 [SEP]'

# 查看token和ID的对应
tokens = tokenizer.tokenize("今天天气真好")
# ['今', '天', '天', '气', '真', '好']
ids = tokenizer.convert_tokens_to_ids(tokens)
# [1234, 5678, 5678, 9012, 3456, 7890]
```

---

## 4. 加载和使用预训练模型

### 4.1 自动模型加载

```python
from transformers import AutoTokenizer, AutoModel

# 自动推断模型类型和架构
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
model = AutoModel.from_pretrained('bert-base-chinese')

# 同样的代码可以用于GPT、RoBERTa等任何模型
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModel.from_pretrained('gpt2')
```

### 4.2 BERT 用于分类

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2  # 二分类
)

# 准备文本
texts = ["今天天气真好", "这部电影太烂了"]

# 编码
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors='pt'
)

# 模型推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# 解码结果
id2label = {0: "负面", 1: "正面"}
for text, pred in zip(texts, predictions):
    print(f"{text} → {id2label[pred.item()]}")
```

### 4.3 GPT 用于生成

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 编码输入
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors='pt')

# 生成参数
output = model.generate(
    **inputs,
    max_length=100,           # 最大生成长度
    num_return_sequences=1,   # 返回序列数
    temperature=0.7,          # 温度（创造性 vs 确定性）
    top_k=50,                 # Top-K采样：只考虑概率最高的50个token
    top_p=0.95,               # Nucleus采样：累积概率达到0.95的token
    do_sample=True,           # 使用采样（False=贪心解码）
    repetition_penalty=1.2,   # 重复惩罚
    pad_token_id=tokenizer.eos_token_id
)

# 解码
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

### 4.4 生成参数详解

| 参数 | 作用 | 常用值 |
|------|------|-------|
| `max_length` | 最大生成长度 | 50-500 |
| `temperature` | 温度，越高越随机 | 0.7-1.0 |
| `top_k` | 只考虑概率最高的K个token | 20-50 |
| `top_p` (nucleus) | 累积概率达到P的token | 0.9-0.95 |
| `repetition_penalty` | 重复惩罚，>1减少重复 | 1.0-1.5 |
| `num_beams` | Beam search宽度 | 1, 3, 5 |
| `do_sample` | 是否采样（vs 贪心） | True/False |

---

## 5. 模型微调完整流程

### 5.1 数据准备

```python
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 示例数据
train_texts = ["这个产品很好", "非常差", "推荐", "不要买"]
train_labels = [1, 0, 1, 0]

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
```

### 5.2 优化器和学习率调度

```python
from transformers import AdamW, get_linear_schedule_with_warmup

# 优化器（分层学习率）
optimizer = AdamW([
    {'params': model.bert.encoder.layer[:10].parameters(), 'lr': 1e-5},  # 底层小学习率
    {'params': model.bert.encoder.layer[10:].parameters(), 'lr': 2e-5},  # 顶层大学习率
    {'params': model.classifier.parameters(), 'lr': 5e-5}  # 分类头最大学习率
], lr=2e-5)

# 学习率调度
num_epochs = 3
num_training_steps = num_epochs * len(dataloader)
warmup_steps = int(0.1 * num_training_steps)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)
```

### 5.3 训练循环

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # 数据移到GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        lr_scheduler.step()
        
        # 统计
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += len(batch['labels'])
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")
```

### 5.4 保存和加载

```python
# 保存模型
model.save_pretrained('./my_bert_model')
tokenizer.save_pretrained('./my_bert_model')

# 加载模型
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('./my_bert_model')
tokenizer = AutoTokenizer.from_pretrained('./my_bert_model')

# 上传到Hugging Face Hub（可选）
# model.push_to_hub("username/my-model")
```

---

## 6. 常用中文模型推荐

| 模型名 | 参数 | 适用任务 |
|--------|------|---------|
| `bert-base-chinese` | 102M | 通用中文理解 |
| `chinese-roberta-wwm-ext` | 102M | 全词掩码，效果更好 |
| `chinese-roberta-wwm-ext-large` | 330M | 大型全词掩码 |
| `hfl/chinese-bert-wwm-ext` | 108M | 哈工大全词掩码 |
| `uer/gpt2-chinese-cluecorpussmall` | 102M | 中文GPT-2 |
| `IDEA-CCNL/Erlangshen-Roberta-110M` | 110M | 二郎神系列 |

---

## 7. 性能优化技巧

### 7.1 使用 AutoModel 自动选择

```python
from transformers import AutoModel, AutoTokenizer

# 自动识别最佳模型实现
model = AutoModel.from_pretrained('bert-base-chinese', torch_dtype=torch.float16)
```

### 7.2 批量推理

```python
# 批量编码比单条快得多
texts = [text1, text2, text3, ...]  # 尽量多条一起编码
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

### 7.3 模型压缩

```python
# 量化（减少显存占用）
model = model.half()  # FP16

# 或者使用 Optimum 库进行INT8量化
from optimum.onnxruntime import ORTModelForSequenceClassification
```

### 7.4 使用 Fast Tokenizer

```python
# Rust实现的快速tokenizer（默认已使用）
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', use_fast=True)
```

---

## 8. 今日速查表

### 8.1 模板1：BERT分类

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 加载
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)

# 编码
texts = ["今天天气好", "电影难看"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
```

### 8.2 模板2：GPT生成

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 生成
text = "Once upon a time"
inputs = tokenizer(text, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
result = tokenizer.decode(outputs[0])
```

### 8.3 模板3：微调

```python
from transformers import AdamW

# 优化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练
model.train()
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 9. 下节课预告

**大模型应用开发 - LangChain和RAG**

- LangChain框架核心概念
- Chain、Prompt、Memory
- 检索增强生成（RAG）原理
- 向量数据库（FAISS, Chroma）
- Embedding模型
- 构建知识库问答系统

---

*学习日期：2026-05-24*
*Hugging Face实战完成！*
