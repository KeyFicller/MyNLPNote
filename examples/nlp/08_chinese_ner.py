#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文命名实体识别（NER）实战 - BERT 序列标注项目
====================================================
项目名称：中文人名/地名/组织名识别
任务：使用 bert-base-chinese 进行序列标注，识别文本中的实体

学习目标：
- 理解序列标注与文本分类的区别
- 掌握 BIO 标注格式
- 学习 Token-level 分类的完整流程
- 掌握 NER 评估指标（F1-score、实体级精确率）

BIO 标注说明：
- B-PER：人名开始（Begin-Person）
- I-PER：人名内部（Inside-Person）
- B-LOC：地名开始（Begin-Location）
- I-LOC：地名内部（Inside-Location）
- B-ORG：组织开始（Begin-Organization）
- I-ORG：组织内部（Inside-Organization）
- O：非实体（Outside）
"""

import os

# ========== 配置 Hugging Face 镜像源 ==========
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

print(f"[NET] 使用镜像源: {os.environ['HF_ENDPOINT']}")
print(f"[CACHE] 模型缓存目录: {os.environ['HF_HOME']}")

# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
from tqdm import tqdm

print("=" * 70)
print("Chinese Named Entity Recognition (NER) Practice")
print("=" * 70)

# ============================================================
# 第一部分：数据准备 - BIO 格式
# ============================================================
print("\n" + "=" * 70)
print("第一部分：数据准备 - BIO 标注格式")
print("=" * 70)

print("""
【BIO 标注格式介绍】

对于每个字（token），标注其是否为实体的开始/内部/外部：

示例句子：马云 在 杭州 创立了 阿里巴巴 公司
标注结果：
  马  → B-PER   (人名开始)
  云  → I-PER   (人名内部)
  在  → O       (非实体)
  杭  → B-LOC   (地名开始)
  州  → I-LOC   (地名内部)
  创  → O       (非实体)
  立  → O       (非实体)
  了  → O       (非实体)
  阿  → B-ORG   (组织开始)
  里  → I-ORG   (组织内部)
  巴  → I-ORG   (组织内部)
  巴  → I-ORG   (组织内部)
  公  → O       (非实体)
  司  → O       (非实体)

标注集合（标签表）：
  O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
  共 7 个类别
""")

# 定义标签
LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

print(f"\n[LABEL] 标签表：")
for idx, label in enumerate(LABEL_LIST):
    print(f"   {idx}: {label}")

# 模拟 NER 训练数据
raw_data = [
    # ========== 人名 + 地名 + 组织 ==========
    ("马云在杭州创立了阿里巴巴公司",
     ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),

    ("李彦宏在北京创办了百度公司",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O']),

    ("马化腾在深圳创建了腾讯科技",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),

    ("任正非在东莞成立了华为技术",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),

    ("雷军在北京市海淀区创办了小米集团",
     ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG']),

    ("刘强东在江苏宿迁建立了京东商城",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG']),

    ("张一鸣在北京字节跳动工作",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),

    ("黄峥在上海创办了拼多多公司",
     ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),

    ("王兴在北京创办了美团公司",
     ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O']),

    ("程维在北京创立了滴滴出行",
     ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),

    # ========== 更多样本 ==========
    ("特斯拉公司在上海建立了超级工厂",
     ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']),

    ("苹果公司在中国设立了研发中心",
     ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O']),

    ("比尔盖茨创立了微软公司",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),

    ("马斯克在加州创办了太空探索公司",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),

    ("扎克伯格在哈佛大学创建了脸书公司",
     ['B-PER', 'I-PER', 'I-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),

    ("贝佐斯在西雅图创立了亚马逊公司",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),

    ("李开复在北京创办了创新工场",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),

    ("俞敏洪在北京中关村创立了新东方",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG']),

    ("马云和张勇在杭州召开了会议",
     ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O']),

    ("百度公司和阿里巴巴都在北京设有总部",
     ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O']),

    ("雷军和林斌在小米公司工作",
     ['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O']),

    ("腾讯总部在深圳南山区",
     ['B-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O']),

    ("华为公司在东莞松山湖有基地",
     ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O']),

    ("阿里巴巴在杭州西溪园区",
     ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'O', 'O']),

    ("字节跳动在北京中关村软件园",
     ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O']),

    ("李彦宏和马云都是中国互联网企业家",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O']),

    ("美团公司在北京市朝阳区设有办公点",
     ['B-ORG', 'I-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O']),

    ("京东商城的刘强东来自江苏",
     ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER', 'I-PER', 'O', 'O', 'B-LOC', 'I-LOC']),

    ("张一鸣毕业于南开大学",
     ['B-PER', 'I-PER', 'I-PER', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']),

    ("扎克伯格在哈佛大学学习计算机",
     ['B-PER', 'I-PER', 'I-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O']),
]

print(f"\n📊 数据集统计:")
print(f"   总样本数: {len(raw_data)}")

# 统计实体分布
entity_counts = defaultdict(int)
for text, labels in raw_data:
    for label in labels:
        if label != 'O':
            entity_type = label.split('-')[1]  # PER/LOC/ORG
            entity_counts[entity_type] += 1

print(f"\n📈 实体分布:")
print(f"   人名(PER): {entity_counts.get('PER', 0)} 个token")
print(f"   地名(LOC): {entity_counts.get('LOC', 0)} 个token")
print(f"   组织(ORG): {entity_counts.get('ORG', 0)} 个token")

# 划分数据集
texts = [item[0] for item in raw_data]
labels_list = [item[1] for item in raw_data]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_list, test_size=0.2, random_state=42
)

print(f"\n📈 数据划分:")
print(f"   训练集: {len(train_texts)} 条")
print(f"   测试集: {len(test_texts)} 条")

# 显示一个样本
print(f"\n📝 样本示例:")
sample_text = train_texts[0]
sample_labels = train_labels[0]
print(f"   文本: {sample_text}")
print(f"   标注: {sample_labels}")

# 可视化实体
print(f"\n🔍 实体可视化:")
entities = []
current_entity = None
current_tokens = []

for char, label in zip(sample_text, sample_labels):
    if label.startswith('B-'):
        if current_entity:
            entities.append((current_entity, ''.join(current_tokens)))
        current_entity = label[2:]  # PER/LOC/ORG
        current_tokens = [char]
    elif label.startswith('I-') and current_entity == label[2:]:
        current_tokens.append(char)
    else:
        if current_entity:
            entities.append((current_entity, ''.join(current_tokens)))
            current_entity = None
            current_tokens = []

if current_entity:
    entities.append((current_entity, ''.join(current_tokens)))

for ent_type, ent_text in entities:
    print(f"   [{ent_type}] {ent_text}")

# ============================================================
# 第二部分：创建 Dataset 和 DataLoader
# ============================================================
print("\n" + "=" * 70)
print("第二部分：创建 Dataset 和 DataLoader")
print("=" * 70)


class NERDataset(Dataset):
    """NER 数据集 - 序列标注任务"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels  # 每个字对应一个标签
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = LABEL2ID

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        char_labels = self.labels[idx]  # 字的标签列表

        # 将文本转为 token IDs（BERT 使用 WordPiece，可能将一个字拆成多个 subword）
        # 这里简单处理：按字分割，每个字作为一个 token
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 处理标签对齐问题
        # BERT 会自动添加 [CLS] 和 [SEP]，我们需要相应添加特殊标签
        # [CLS] 和 [SEP] 的标签设为 -100（PyTorch 会忽略）
        label_ids = [-100]  # [CLS]

        for label in char_labels[:self.max_length - 2]:  # 留位置给 [CLS] 和 [SEP]
            label_ids.append(self.label2id[label])

        label_ids.append(-100)  # [SEP]

        # Padding 到 max_length
        while len(label_ids) < self.max_length:
            label_ids.append(-100)  # padding 位置也设为 -100

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


print("\n🔤 加载 Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(f"   Tokenizer 加载完成")
print(f"   词表大小: {tokenizer.vocab_size}")

# 创建 Dataset
train_dataset = NERDataset(train_texts, train_labels, tokenizer)
test_dataset = NERDataset(test_texts, test_labels, tokenizer)

print(f"\n📦 Dataset 创建完成:")
print(f"   训练样本: {len(train_dataset)}")
print(f"   测试样本: {len(test_dataset)}")

# 查看一个样本
sample = train_dataset[0]
print(f"\n📝 样本示例:")
print(f"   Input IDs 形状: {sample['input_ids'].shape}")
print(f"   Attention Mask 形状: {sample['attention_mask'].shape}")
print(f"   Labels 形状: {sample['labels'].shape}")
print(f"   Labels (前20个): {sample['labels'][:20].tolist()}")

# 将 label id 转回标签名查看
valid_labels = [ID2LABEL.get(idx.item(), 'PAD/CLS/SEP') for idx in sample['labels'][:20]
                if idx.item() in ID2LABEL]
print(f"   标签名称: {valid_labels}")

# 创建 DataLoader
BATCH_SIZE = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"\n📦 DataLoader 创建完成:")
print(f"   训练批次: {len(train_loader)}")
print(f"   测试批次: {len(test_loader)}")

# ============================================================
# 第三部分：加载模型
# ============================================================
print("\n" + "=" * 70)
print("第三部分：加载 BERT 模型（Token Classification）")
print("=" * 70)

print("""
【Token Classification vs Sequence Classification】

文本分类（之前学的）：
  输入: [CLS] 马 云 创 立 了 阿 里 巴 巴 [SEP]
  输出: 单个分类结果（正面/负面）← 只用 [CLS] 位置的输出

序列标注（NER）：
  输入: [CLS] 马 云 创 立 了 阿 里 巴 巴 [SEP]
  输出:  -   B  I  O  O  O  B  I  I  I   -   ← 每个 token 都有标签

模型结构变化：
  - 分类头从 Linear(768, num_labels) 改为每个 token 位置都有一个
  - BertForTokenClassification: 输出 [batch, seq_len, num_labels]
""")

print("\n🤖 加载 bert-base-chinese...")
model = BertForTokenClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=len(LABEL_LIST),  # 7 个标签
    output_attentions=False,
    output_hidden_states=False
)

print(f"   模型加载完成!")
print(f"   模型类型: {type(model).__name__}")
print(f"   标签类别数: {model.num_labels}")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n💻 使用设备: {device}")
model = model.to(device)

# ============================================================
# 第四部分：训练配置
# ============================================================
print("\n" + "=" * 70)
print("第四部分：训练配置")
print("=" * 70)

EPOCHS = 10  # NER 需要更多 epoch
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0
MAX_GRAD_NORM = 1.0

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

print(f"\n⚙️ 训练配置:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   总训练步数: {total_steps}")

# ============================================================
# 第五部分：训练模型
# ============================================================
print("\n" + "=" * 70)
print("第五部分：训练模型")
print("=" * 70)

best_f1 = 0.0


# 实体提取函数（用于评估）
def extract_entities(text, predictions):
    """从预测结果中提取实体
    
    Args:
        text: 原始文本字符串
        predictions: 预测标签ID数组，包含 [CLS] (索引0), 正文标签, [SEP], padding (-100)
    
    Returns:
        提取的实体列表 [(实体类型, 实体文本), ...]
    """
    entities = []
    current_entity = None
    current_tokens = []

    # predictions[0] 是 [CLS] 的标签 (-100)，需要从 predictions[1] 开始对应 text[0]
    # 所以使用 enumerate 从索引 1 开始
    for i, char in enumerate(text):
        pred_idx = i + 1  # 跳过 [CLS] 位置
        if pred_idx >= len(predictions):
            break
        
        pred = predictions[pred_idx]
        if pred == -100:
            continue
        
        label = ID2LABEL[pred]

        if label.startswith('B-'):
            if current_entity:
                entities.append((current_entity, ''.join(current_tokens)))
            current_entity = label[2:]
            current_tokens = [char]
        elif label.startswith('I-') and current_entity == label[2:]:
            current_tokens.append(char)
        else:
            if current_entity:
                entities.append((current_entity, ''.join(current_tokens)))
                current_entity = None
                current_tokens = []

    if current_entity:
        entities.append((current_entity, ''.join(current_tokens)))

    return entities


for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)

    # ========== 训练阶段 ==========
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    print(f"   Train Loss: {avg_loss:.4f}")

    # ========== 验证阶段（每 epoch 都评估）==========
    model.eval()
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            # 只收集非 padding 位置的有效标签
            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j] != -100:
                        all_true_labels.append(labels[i][j].cpu().item())
                        all_predictions.append(predictions[i][j].cpu().item())

    # 计算 token-level 准确率
    token_acc = sum(1 for p, t in zip(all_predictions, all_true_labels) if p == t) / len(all_predictions)
    print(f"   Token Acc: {token_acc:.2%}")

    # 计算实体级 F1（简化的精确匹配）
    # 提取预测的实体和真实的实体
    # 这里简化处理，用 token-level 的 F1 近似
    from sklearn.metrics import f1_score

    f1_micro = f1_score(all_true_labels, all_predictions, average='micro', zero_division=0)
    print(f"   Token F1: {f1_micro:.2%}")

    if f1_micro > best_f1:
        best_f1 = f1_micro
        print(f"   ✨ 最佳模型！F1: {best_f1:.2%}")

print(f"\n🎉 训练完成！最佳 F1: {best_f1:.2%}")

# ============================================================
# 第六部分：详细评估
# ============================================================
print("\n" + "=" * 70)
print("第六部分：模型评估")
print("=" * 70)

model.eval()

print(f"\n🔍 测试集预测示例:")
for i in range(min(3, len(test_texts))):
    text = test_texts[i]
    true_labels = test_labels[i]

    # 编码
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    # 提取实体
    # 注意：extract_entities 内部会跳过 predictions[0]（[CLS]位置）
    # 所以 true_labels 也要加上 -100 来对齐
    true_label_ids_with_special = [-100] + [LABEL2ID[l] for l in true_labels] + [-100]
    true_entities = extract_entities(text, true_label_ids_with_special)
    pred_entities = extract_entities(text, predictions)

    print(f"\n   文本: {text}")
    print(f"   真实实体: {true_entities}")
    print(f"   预测实体: {pred_entities}")

# ============================================================
# 第七部分：保存模型
# ============================================================
print("\n" + "=" * 70)
print("第七部分：保存模型")
print("=" * 70)

save_dir = './saved_models/chinese_ner_bert'
os.makedirs(save_dir, exist_ok=True)

print(f"\n💾 保存模型到: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 保存标签映射
import json

config_info = {
    'model_name': 'bert-base-chinese',
    'num_labels': len(LABEL_LIST),
    'label_list': LABEL_LIST,
    'label2id': LABEL2ID,
    'id2label': ID2LABEL,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'final_f1': float(best_f1)
}

with open(os.path.join(save_dir, 'ner_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config_info, f, ensure_ascii=False, indent=2)

print(f"   ✅ 模型保存完成")

# ============================================================
# 第八部分：新样本预测
# ============================================================
print("\n" + "=" * 70)
print("第八部分：新样本预测")
print("=" * 70)

# 加载保存的模型
loaded_tokenizer = BertTokenizer.from_pretrained(save_dir)
loaded_model = BertForTokenClassification.from_pretrained(save_dir)
loaded_model = loaded_model.to(device)
loaded_model.eval()

print(f"\n🔄 模型加载完成")

test_samples = [
    "马化腾在深圳创立了腾讯",
    "乔布斯在加利福尼亚州创办了苹果公司",
    "李彦宏和马云在北京参加了会议",
]

print(f"\n📝 新样本预测:")
for text in test_samples:
    inputs = loaded_tokenizer(
        text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    entities = extract_entities(text, predictions)

    print(f"\n   文本: {text}")
    print(f"   识别实体: {entities}")

    # 可视化标注 - predictions[0]是[CLS]，predictions[1]对应text[0]
    print(f"   标注: ", end="")
    for i, char in enumerate(text):
        pred_idx = i + 1  # 跳过 [CLS] 位置
        if pred_idx >= len(predictions):
            break
        pred = predictions[pred_idx]
        if pred == -100:
            continue
        label = ID2LABEL[pred]
        if label != 'O':
            print(f"[{char}/{label}]", end="")
        else:
            print(char, end="")
    print()

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print("""
✅ 本项目完成了：

1. BIO 标注格式
   - B-XXX: 实体开始
   - I-XXX: 实体内部
   - O: 非实体

2. 序列标注 vs 文本分类
   - 分类：句子 → 一个标签
   - 标注：每个 token → 一个标签

3. BertForTokenClassification
   - 输出形状: [batch, seq_len, num_labels]
   - 对每个 token 位置做分类

4. 标签对齐问题
   - [CLS] 和 [SEP] 的标签设为 -100（忽略）
   - Padding 位置也设为 -100

5. 实体提取
   - 从 BIO 标注还原完整实体
   - 处理 B-I 连续标签

📚 关键区别：
   • 分类任务：只看 [CLS] token 的输出
   • NER任务：看每个 token 的输出
   • 损失计算：只计算非 -100 位置的损失

🚀 下一步建议：
   1. 使用真实 NER 数据集（如 CLUENER、MSRA）
   2. 添加更多实体类型（时间、产品名等）
   3. 学习 CRF（条件随机场）层，提升标注一致性
   4. 实现实体级精确率/召回率/F1 评估
   5. 尝试其他模型（RoBERTa-wwm、MacBERT）
""")

print("\n" + "=" * 70)
print("中文 NER 实战完成！🎉")
print("=" * 70)
