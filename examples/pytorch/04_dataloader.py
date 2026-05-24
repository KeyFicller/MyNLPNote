#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第14课：DataLoader数据加载（NLP前置课）
=========================================
本课程快速学习PyTorch数据加载机制：
- Dataset - 自定义数据集
- DataLoader - 批量加载、打乱、多进程
- collate_fn - 处理变长数据（NLP关键！）
- 文本数据加载实战
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

print("=" * 70)
print("第14课：DataLoader数据加载（NLP前置课）")
print("=" * 70)

# 设置随机种子
torch.manual_seed(42)

# ============================================================
# 第一部分：Dataset - 自定义数据集
# ============================================================
print("\n" + "=" * 70)
print("第一部分：Dataset - 定义数据长什么样")
print("=" * 70)

print("""
【核心概念】
Dataset是数据的"说明书"，告诉PyTorch：
- 数据有多少条？ → __len__
- 如何获取一条数据？ → __getitem__

类比：Dataset像餐厅的菜单，告诉顾客有什么菜，怎么点
""")

# 示例1：简单的数字数据集
class SimpleNumberDataset(Dataset):
    """简单的数字数据集：返回数字及其平方"""
    def __init__(self, max_num=100):
        self.numbers = list(range(max_num))
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.numbers)
    
    def __getitem__(self, idx):
        """获取第idx条数据"""
        num = self.numbers[idx]
        square = num ** 2
        return {
            'input': torch.tensor(num, dtype=torch.float32),
            'target': torch.tensor(square, dtype=torch.float32)
        }

# 创建数据集
dataset = SimpleNumberDataset(max_num=10)
print(f"\n【示例1：数字数据集】")
print(f"数据集大小: {len(dataset)}")
print(f"第3条数据: {dataset[3]}")
print(f"第7条数据: {dataset[7]}")

# 示例2：模拟文本分类数据集（NLP场景）
class TextClassificationDataset(Dataset):
    """
    模拟文本分类数据集
    为NLP学习做准备！
    """
    def __init__(self):
        # 模拟数据：句子 + 标签（0=负面，1=正面）
        self.samples = [
            {'text': '这部电影太棒了', 'label': 1},
            {'text': '完全浪费时间的烂片', 'label': 0},
            {'text': '演员演技出色', 'label': 1},
            {'text': '剧情毫无逻辑', 'label': 0},
            {'text': '值得一看的好电影', 'label': 1},
            {'text': '太差了，后悔买票', 'label': 0},
            {'text': '特效很震撼', 'label': 1},
            {'text': '无聊透顶', 'label': 0},
        ]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'text': sample['text'],
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'text_len': len(sample['text'])  # 文本长度（后续有用）
        }

text_dataset = TextClassificationDataset()
print(f"\n【示例2：文本分类数据集】")
print(f"数据集大小: {len(text_dataset)}")
print(f"第0条数据:")
print(f"  文本: {text_dataset[0]['text']}")
print(f"  标签: {text_dataset[0]['label']}")
print(f"  长度: {text_dataset[0]['text_len']}")

# ============================================================
# 第二部分：DataLoader - 批量加载数据
# ============================================================
print("\n" + "=" * 70)
print("第二部分：DataLoader - 批量加载、打乱、加速")
print("=" * 70)

print("""
【核心概念】
DataLoader是数据的"服务员"，负责：
- batch_size: 一次上几道菜（批量大小）
- shuffle: 是否打乱顺序（训练时打乱，测试时不打乱）
- num_workers: 几个服务员同时工作（多进程加载）
- collate_fn: 如何处理一桌的菜（处理变长数据的关键！）
""")

# 基础DataLoader
print("\n【示例3：基础DataLoader使用】")
dataloader = DataLoader(
    dataset,
    batch_size=4,      # 每批4个样本
    shuffle=False,     # 不打乱
    num_workers=0      # 主进程加载（Windows/Mac建议0）
)

print(f"批次数量: {len(dataloader)}")
for batch_idx, batch in enumerate(dataloader):
    print(f"\n批次 {batch_idx}:")
    print(f"  输入: {batch['input']}")
    print(f"  目标: {batch['target']}")
    print(f"  批次大小: {len(batch['input'])}")

# shuffle对比
print("\n【示例4：shuffle的作用】")
dataloader_shuffle = DataLoader(dataset, batch_size=4, shuffle=True)
print("第一次遍历（打乱）:")
first_batch = next(iter(dataloader_shuffle))
print(f"  输入: {first_batch['input']}")

print("\n第二次遍历（再次打乱）:")
second_batch = next(iter(dataloader_shuffle))
print(f"  输入: {second_batch['input']}")

print("""
【注意】
- 训练时设置 shuffle=True，让数据随机，防止模型记住顺序
- 测试/验证时设置 shuffle=False，保持顺序便于评估
""")

# ============================================================
# 第三部分：collate_fn - 处理变长数据（NLP关键！）
# ============================================================
print("\n" + "=" * 70)
print("第三部分：collate_fn - 处理变长文本（NLP必备）")
print("=" * 70)

print("""
【问题背景】
文本数据有个特点：长度不统一！
- "这部电影太棒了" → 7个字符
- "完全浪费时间的烂片" → 9个字符
- "太差了，后悔买票" → 9个字符

默认的DataLoader会把它们堆叠成张量，但长度不同会报错！

【解决方案】
collate_fn = 自定义"如何把多条数据打包成一个批次"
""")

# 先看看默认行为会出什么问题
default_loader = DataLoader(text_dataset, batch_size=4, shuffle=False)
try:
    batch = next(iter(default_loader))
    print(f"文本: {batch['text']}")  # 这是列表，没问题
    print(f"标签: {batch['label']}")  # 可以堆叠，没问题
except Exception as e:
    print(f"错误: {e}")

# 自定义collate_fn处理文本数据
print("\n【示例5：自定义collate_fn处理文本】")

def text_collate_fn(batch):
    """
    自定义collate函数处理文本批次
    """
    # batch是一个列表，包含多个样本
    # batch = [sample0, sample1, sample2, ...]
    
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([item['text_len'] for item in batch])
    
    return {
        'texts': texts,           # 保持为字符串列表（长度不一）
        'labels': labels,         # 堆叠成张量
        'lengths': lengths      # 记录每条文本长度
    }

text_loader = DataLoader(
    text_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=text_collate_fn  # 使用自定义collate
)

batch = next(iter(text_loader))
print(f"批次文本: {batch['texts']}")
print(f"批次标签: {batch['labels']}")
print(f"文本长度: {batch['lengths']}")

# ============================================================
# 第四部分：Padding - 处理变长序列（NLP核心技巧）
# ============================================================
print("\n" + "=" * 70)
print("第四部分：Padding - 把变长序列变成固定长度")
print("=" * 70)

print("""
【NLP标准做法】
为了使用神经网络处理文本，需要把变长文本变成固定长度：
1. 找到批次中最长的文本
2. 其他文本用特殊符号<PAD>填充到相同长度
3. 后续用mask告诉模型哪些是真实数据，哪些是padding
""")

# 将文本转为数字ID（简单模拟）
vocab = {'<PAD>': 0, '这': 1, '部': 2, '电': 3, '影': 4, '太': 5, 
         '棒': 6, '了': 7, '完': 8, '全': 9, '浪': 10, '费': 11, 
         '时': 12, '间': 13, '的': 14, '烂': 15, '片': 16,
         '演': 17, '员': 18, '技': 19, '出': 20, '色': 21,
         '剧': 22, '情': 23, '毫': 24, '无': 25, '逻': 26, '辑': 27,
         '值': 28, '得': 29, '一': 30, '看': 31, '好': 32,
         '，': 33, '差': 34, '后': 35, '悔': 36, '买': 37, '票': 38,
         '特': 39, '效': 40, '很': 41, '震': 42, '撼': 43,
         '无': 44, '聊': 45, '透': 46, '顶': 47}

def text_to_ids(text, max_len=10):
    """将文本转为ID序列，并padding到固定长度"""
    ids = [vocab.get(char, 0) for char in text]
    
    # 截断或填充
    if len(ids) > max_len:
        ids = ids[:max_len]  # 截断
    else:
        ids = ids + [0] * (max_len - len(ids))  # 填充<PAD>
    
    return ids, min(len(text), max_len)  # 返回ID和真实长度

# 带有padding的collate_fn
print("\n【示例6：带Padding的collate_fn】")

def padding_collate_fn(batch, max_len=10):
    """
    带有padding的collate函数
    """
    texts = []
    labels = []
    lengths = []
    masks = []  # 标记哪些是真实数据
    
    for item in batch:
        text_ids, true_len = text_to_ids(item['text'], max_len)
        texts.append(text_ids)
        labels.append(item['label'])
        lengths.append(true_len)
        
        # 创建mask：真实数据为1，padding为0
        mask = [1] * true_len + [0] * (max_len - true_len)
        masks.append(mask)
    
    return {
        'text_ids': torch.tensor(texts, dtype=torch.long),      # [batch, max_len]
        'labels': torch.stack(labels),                            # [batch]
        'lengths': torch.tensor(lengths),                        # [batch]
        'masks': torch.tensor(masks, dtype=torch.float32)       # [batch, max_len]
    }

# 创建新的DataLoader
padding_loader = DataLoader(
    text_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=lambda batch: padding_collate_fn(batch, max_len=10)
)

batch = next(iter(padding_loader))
print(f"文本ID形状: {batch['text_ids'].shape}")
print(f"第一条文本ID: {batch['text_ids'][0]}")
print(f"真实长度: {batch['lengths'][0]}")
print(f"Mask: {batch['masks'][0]}")

print(f"\n批次信息:")
print(f"  text_ids: {batch['text_ids'].shape} (批量, 最大长度)")
print(f"  labels: {batch['labels'].shape} (批量,)")
print(f"  lengths: {batch['lengths']}")
print(f"  masks: {batch['masks'].shape} (用于告诉模型忽略padding)")

# ============================================================
# 第五部分：完整训练循环模板
# ============================================================
print("\n" + "=" * 70)
print("第五部分：完整训练循环模板（使用DataLoader）")
print("=" * 70)

# 创建一个简单的分类模型
class SimpleTextClassifier(torch.nn.Module):
    """简化版文本分类器（仅用于演示DataLoader使用）"""
    def __init__(self, vocab_size=50, embed_dim=16, num_classes=2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc = torch.nn.Linear(embed_dim, num_classes)
    
    def forward(self, text_ids, masks):
        # embedding: [batch, max_len] -> [batch, max_len, embed_dim]
        embedded = self.embedding(text_ids)
        
        # 使用mask做平均池化（只考虑真实token）
        mask_expanded = masks.unsqueeze(-1)  # [batch, max_len, 1]
        sum_embedded = (embedded * mask_expanded).sum(dim=1)  # [batch, embed_dim]
        avg_embedded = sum_embedded / masks.sum(dim=1, keepdim=True).clamp(min=1)
        
        return self.fc(avg_embedded)

# 模拟训练
print("\n【模拟训练过程】")
model = SimpleTextClassifier()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
model.train()
for epoch in range(2):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(padding_loader):
        # 前向传播
        outputs = model(batch['text_ids'], batch['masks'])
        loss = criterion(outputs, batch['labels'])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch['labels']).sum().item()
        total += len(batch['labels'])
    
    print(f"Epoch {epoch+1}: Loss={total_loss/len(padding_loader):.4f}, Acc={100*correct/total:.1f}%")

print(f"\n训练完成！使用了 {len(padding_loader)} 个批次")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("总结 - DataLoader使用要点")
print("=" * 70)

print("""
【核心API】
1. Dataset: 定义__len__和__getitem__
2. DataLoader: batch_size, shuffle, num_workers, collate_fn
3. collate_fn: 处理变长数据的关键（NLP必备）

【NLP数据处理流程】
原始文本
    ↓
Dataset.__getitem__ 返回单个样本
    ↓
collate_fn 收集批次样本
    ↓
- 分词/转为ID
- padding到相同长度
- 生成mask
    ↓
DataLoader 返回批次张量
    ↓
模型训练

【下节课预告】
正式进入NLP！我们将学习：
1. 文本预处理 - 分词、清洗
2. 词嵌入 - Word2Vec原理
3. Transformer架构 - BERT/GPT的基础

准备进入自然语言处理的世界！
""")

print("\n" + "=" * 70)
print("第14课完成！准备进入NLP！🚀")
print("=" * 70)
