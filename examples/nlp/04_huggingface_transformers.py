#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第18课：Hugging Face Transformers实战
========================================
本课程学习使用Hugging Face transformers库进行实战：
- 加载预训练模型和Tokenizer
- 文本编码和处理
- 使用BERT进行文本分类
- 使用GPT进行文本生成
- 模型微调完整流程

需要安装：pip install transformers datasets
"""

print("=" * 70)
print("第18课：Hugging Face Transformers实战 🚀")
print("=" * 70)

# ============================================================
# 第一部分：环境检查和安装
# ============================================================
print("\n" + "=" * 70)
print("第一部分：环境准备")
print("=" * 70)

try:
    import transformers
    print(f"✅ transformers 版本: {transformers.__version__}")
except ImportError:
    print("❌ 请先安装 transformers: pip install transformers")
    raise

try:
    import torch
    print(f"✅ torch 版本: {torch.__version__}")
except ImportError:
    print("❌ torch 未安装")
    raise

# ============================================================
# 第二部分：理解Pipeline
# ============================================================
print("\n" + "=" * 70)
print("第二部分：Pipeline - 最简使用方式")
print("=" * 70)

print("""
【Pipeline是什么？】
Pipeline是Hugging Face提供的"开箱即用"工具，一行代码搞定：
- 文本分类
- 情感分析
- 命名实体识别
- 问答
- 文本生成
- 翻译
- 摘要

【原理】
Pipeline自动完成：
1. 加载预训练模型
2. 加载对应Tokenizer
3. 文本编码
4. 模型推理
5. 结果解码
""")

# 由于可能需要下载模型，这里用条件执行
print("\n【示例1：情感分析Pipeline】")
print("代码示例（需要联网下载模型）：")
print("""
from transformers import pipeline

# 创建情感分析pipeline
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# 使用
results = classifier([
    "I love this product! It's amazing.",
    "This is terrible. I hate it."
])

# 输出
[
    {'label': 'POSITIVE', 'score': 0.9998},
    {'label': 'NEGATIVE', 'score': 0.9991}
]
""")

print("\n【示例2：文本生成Pipeline】")
print("""
from transformers import pipeline

# 创建文本生成pipeline
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# 生成文本
result = generator(
    "Once upon a time",
    max_length=50,
    num_return_sequences=1
)

# 输出
[{'generated_text': 'Once upon a time, there was a little girl...'}]
""")

# ============================================================
# 第三部分：Tokenizer详解
# ============================================================
print("\n" + "=" * 70)
print("第三部分：Tokenizer - 文本处理的桥梁")
print("=" * 70)

print("""
【Tokenizer的作用】
原始文本 → Tokenizer → 模型能理解的数字ID

处理流程：
1. 分词（Tokenization）
2. 映射到词表ID（Vocabulary mapping）
3. 添加特殊token（[CLS], [SEP], [PAD]）
4. 生成Attention Mask

【BERT Tokenizer示例】
输入: "今天天气真好"

处理过程:
1. 分词: ['今', '天', '天', '气', '真', '好'] 或子词 [今天, 天气, 真好]
2. 加特殊token: ['[CLS]', '今', '天', '天', '气', '真', '好', '[SEP]']
3. 转ID: [101, 1234, 5678, 5678, 9012, 3456, 7890, 102]
4. Padding: [101, 1234, 5678, 5678, 9012, 3456, 7890, 102, 0, 0, 0, 0]
   ↑ Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
""")

# 模拟Tokenizer的工作
class MockTokenizer:
    """模拟Tokenizer的工作流程"""
    
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        print(f"初始化MockTokenizer，词表大小: {vocab_size}")
        print(f"特殊token: {self.special_tokens}")
    
    def encode(self, text, max_length=20, padding=True, truncation=True):
        """模拟编码过程"""
        # 1. 简单分词（按字符）
        tokens = list(text)
        
        # 2. 映射到ID（简单哈希模拟）
        token_ids = [hash(t) % (self.vocab_size - 10) + 10 for t in tokens]
        
        # 3. 加特殊token
        token_ids = [self.special_tokens['[CLS]']] + token_ids + [self.special_tokens['[SEP]']]
        
        # 4. 截断
        if truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.special_tokens['[SEP]']]
        
        # 5. Padding
        attention_mask = [1] * len(token_ids)
        if padding and len(token_ids) < max_length:
            pad_length = max_length - len(token_ids)
            token_ids = token_ids + [self.special_tokens['[PAD]']] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': token_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, token_ids):
        """模拟解码"""
        reverse_special = {v: k for k, v in self.special_tokens.items()}
        result = []
        for id in token_ids:
            if id in reverse_special:
                result.append(reverse_special[id])
            else:
                # 模拟还原字符
                result.append(f'<token_{id}>')
        return ' '.join(result)

print("\n【示例3：Tokenizer工作流程演示】")

tokenizer = MockTokenizer(vocab_size=1000)

text = "你好世界"
print(f"\n输入文本: '{text}'")

encoded = tokenizer.encode(text, max_length=10)
print(f"\n编码结果:")
print(f"  input_ids: {encoded['input_ids']}")
print(f"  attention_mask: {encoded['attention_mask']}")

decoded = tokenizer.decode(encoded['input_ids'])
print(f"\n解码结果: {decoded}")

print("""
【Tokenizer的关键参数】

1. max_length: 最大序列长度（BERT常用512，GPT常用1024）
2. padding: 是否填充到max_length
   - True: 填充
   - 'max_length': 填充到max_length
   - False: 不填充
3. truncation: 是否截断超长序列
4. return_tensors: 返回格式
   - 'pt': PyTorch tensor
   - 'np': NumPy array
   - 'tf': TensorFlow tensor
""")

# ============================================================
# 第四部分：手动实现BERT分类流程
# ============================================================
print("\n" + "=" * 70)
print("第四部分：手动实现文本分类流程")
print("=" * 70)

print("""
【完整的BERT分类流程】

1. 加载Tokenizer和Model
2. 文本编码
3. 输入模型
4. 获取输出
5. 解码预测结果
""")

import torch
import torch.nn as nn
import torch.nn.functional as F

# 模拟BERT分类模型
class MockBERTClassifier(nn.Module):
    """模拟BERT用于分类"""
    
    def __init__(self, vocab_size, d_model=768, num_classes=2):
        super().__init__()
        
        # Embedding层
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 简化的Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)
        
        # Transformer
        if attention_mask is not None:
            # 转换mask格式
            mask = attention_mask.bool()
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        # 取[CLS]位置（第一个位置）的输出
        cls_output = x[:, 0, :]
        
        # 分类
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        return logits

print("\n【示例4：模拟BERT分类】")

# 创建模型
model = MockBERTClassifier(vocab_size=10000, num_classes=2)
tokenizer = MockTokenizer(vocab_size=10000)

# 准备数据
texts = ["今天天气真好", "这部电影太烂了"]
labels = [1, 0]  # 1=正面, 0=负面

print(f"\n样本:")
for i, (text, label) in enumerate(zip(texts, labels)):
    print(f"  {i+1}. '{text}' → {'正面' if label == 1 else '负面'}")

# 编码
batch_encoded = [tokenizer.encode(t, max_length=20) for t in texts]
input_ids = torch.tensor([e['input_ids'] for e in batch_encoded])
attention_mask = torch.tensor([e['attention_mask'] for e in batch_encoded])
labels_tensor = torch.tensor(labels)

print(f"\n编码后的形状:")
print(f"  input_ids: {input_ids.shape}")
print(f"  attention_mask: {attention_mask.shape}")
print(f"  labels: {labels_tensor.shape}")

# 模型推理
model.eval()
with torch.no_grad():
    logits = model(input_ids, attention_mask)
    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)

print(f"\n预测结果:")
for i, (text, pred, prob) in enumerate(zip(texts, predictions, probs)):
    confidence = prob[pred].item()
    sentiment = "正面" if pred == 1 else "负面"
    print(f"  '{text}'")
    print(f"    预测: {sentiment} (置信度: {confidence:.2%})")

# ============================================================
# 第五部分：模型微调流程
# ============================================================
print("\n" + "=" * 70)
print("第五部分：模型微调完整流程")
print("=" * 70)

print("""
【微调的核心步骤】

1. 准备数据
2. 创建DataLoader
3. 设置优化器和学习率调度
4. 训练循环
5. 评估和保存
""")

# 模拟训练流程
print("\n【示例5：模拟训练循环】")

from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """简单的文本数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=20):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoded = self.tokenizer.encode(
            self.texts[idx],
            max_length=self.max_length
        )
        return {
            'input_ids': torch.tensor(encoded['input_ids']),
            'attention_mask': torch.tensor(encoded['attention_mask']),
            'labels': torch.tensor(self.labels[idx])
        }

# 准备训练数据
train_texts = [
    "这个产品太棒了", "非常喜欢", "质量很好", "推荐购买",
    "太差了", "很失望", "浪费钱", "不要买"
]
train_labels = [1, 1, 1, 1, 0, 0, 0, 0]

# 创建Dataset和DataLoader
dataset = TextDataset(train_texts, train_labels, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 设置优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# 训练循环
print("\n开始训练...")
model.train()

num_epochs = 2
for epoch in range(num_epochs):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # 前向传播
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(logits, batch['labels'])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        correct += (predictions == batch['labels']).sum().item()
        total += len(batch['labels'])
    
    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2%}")

print("\n训练完成！")

# ============================================================
# 第六部分：真实Hugging Face代码示例
# ============================================================
print("\n" + "=" * 70)
print("第六部分：真实Hugging Face代码模板")
print("=" * 70)

print("""
【模板1：使用BERT进行文本分类】

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. 加载预训练模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2  # 分类类别数
)

# 2. 准备文本
texts = ["今天天气真好", "这部电影太烂了"]

# 3. 编码
inputs = tokenizer(
    texts,
    padding=True,           # 填充
    truncation=True,        # 截断
    max_length=512,         # 最大长度
    return_tensors='pt'     # 返回PyTorch tensor
)

# 4. 模型推理
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

# 5. 解码结果
id2label = {0: "负面", 1: "正面"}
for text, pred in zip(texts, predictions):
    print(f"{text} → {id2label[pred.item()]}")
```
""")

print("""
【模板2：使用GPT生成文本】

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 1. 加载模型和tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 2. 编码输入文本
input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors='pt')

# 3. 生成文本
outputs = model.generate(
    **inputs,
    max_length=100,           # 生成长度
    num_return_sequences=1,   # 生成序列数
    temperature=0.7,          # 温度（创造性）
    top_k=50,                 # Top-K采样
    top_p=0.95,               # Nucleus采样
    do_sample=True            # 使用采样而非贪心
)

# 4. 解码结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```
""")

print("""
【模板3：微调BERT完整代码】

```python
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from datasets import load_dataset

# 1. 加载预训练模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2
)

# 2. 准备数据（示例）
# 使用Hugging Face datasets库加载数据
dataset = load_dataset('imdb')  # 或其他数据集

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 3. 创建DataLoader
train_dataloader = DataLoader(
    tokenized_dataset['train'],
    batch_size=16,
    shuffle=True
)

# 4. 设置优化器和学习率调度
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# 5. 训练循环
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

# 6. 保存模型
model.save_pretrained('./my_model')
tokenizer.save_pretrained('./my_model')
```
""")

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 70)
print("总结与实践")
print("=" * 70)

print("""
【本课核心知识点】
✅ Pipeline: 一行代码搞定常用NLP任务
✅ Tokenizer: 文本→数字ID的桥梁
✅ BERT分类: [CLS] token输出 + 分类头
✅ GPT生成: 自回归生成文本
✅ 微调流程: 数据→训练→评估→保存

【关键API速查】

1. 加载模型和Tokenizer
   ```python
   from transformers import AutoTokenizer, AutoModel
   tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
   model = AutoModel.from_pretrained('bert-base-chinese')
   ```

2. 文本编码
   ```python
   inputs = tokenizer(text, padding=True, truncation=True, 
                      max_length=512, return_tensors='pt')
   # 返回: input_ids, attention_mask, token_type_ids
   ```

3. 模型推理
   ```python
   with torch.no_grad():
       outputs = model(**inputs)
       logits = outputs.logits  # 或 last_hidden_state
   ```

4. 模型微调
   ```python
   from transformers import AdamW
   optimizer = AdamW(model.parameters(), lr=2e-5)
   # 训练循环...
   ```

5. 保存和加载
   ```python
   model.save_pretrained('./my_model')
   tokenizer.save_pretrained('./my_model')
   # 加载
   model = AutoModel.from_pretrained('./my_model')
   ```

【课后实践】
1. 安装transformers并运行Pipeline示例
2. 使用bert-base-chinese做中文情感分类
3. 使用gpt2生成英文文本
4. 在自定义数据集上微调BERT
5. 尝试不同的生成参数（temperature, top_k, top_p）

【推荐模型】
- 中文理解: bert-base-chinese, chinese-roberta-wwm-ext
- 英文理解: bert-base-uncased, roberta-base
- 英文生成: gpt2, gpt2-medium
- 代码生成: microsoft/CodeGPT-small

【学习资源】
- Hugging Face文档: https://huggingface.co/docs/transformers
- 模型Hub: https://huggingface.co/models
- 课程: https://huggingface.co/course

【下节课预告】
大模型应用开发 - LangChain和RAG！
- 使用LangChain构建LLM应用
- 检索增强生成(RAG)原理和实现
- 向量数据库和Embedding
- 构建个人知识库助手
""")

print("\n" + "=" * 70)
print("第18课完成！Hugging Face实战掌握！🎉")
print("=" * 70)
