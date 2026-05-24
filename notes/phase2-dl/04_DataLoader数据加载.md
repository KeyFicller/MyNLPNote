# 第14课：DataLoader数据加载（NLP前置课）

## 学习目标
- [x] 理解 Dataset 和 DataLoader 的关系
- [x] 学会自定义 Dataset
- [x] 掌握 DataLoader 的关键参数
- [x] 理解 collate_fn 的作用（NLP关键）
- [x] 学会使用 Padding 处理变长序列

---

## 1. Dataset - 数据说明书

### 1.1 核心概念

Dataset 回答两个问题：
1. **有多少数据？** → `__len__`
2. **怎么获取一条数据？** → `__getitem__`

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
```

### 1.2 Dataset的作用

| 方法 | 作用 | 返回值 |
|-----|------|-------|
| `__init__` | 初始化数据 | 无 |
| `__len__` | 数据集大小 | int |
| `__getitem__` | 获取第idx条数据 | 单个样本 |

---

## 2. DataLoader - 数据服务员

### 2.1 核心参数

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,           # 数据集
    batch_size=32,     # 每批样本数
    shuffle=True,      # 是否打乱（训练时True，测试时False）
    num_workers=0,     # 加载数据的进程数（Mac/Win建议0）
    collate_fn=None,   # 自定义批次处理方式
    drop_last=False    # 是否丢弃最后一个不完整的批次
)
```

### 2.2 参数速查表

| 参数 | 作用 | 建议值 |
|-----|------|-------|
| `batch_size` | 每批样本数 | 16/32/64/128（根据显存） |
| `shuffle` | 是否打乱 | 训练True，测试False |
| `num_workers` | 多进程加载 | 0（调试时），4+（训练时） |
| `collate_fn` | 自定义批次处理 | NLP必备！ |

---

## 3. collate_fn - 处理变长数据（NLP关键！）

### 3.1 为什么需要 collate_fn？

**问题**：文本数据长度不统一，默认的 DataLoader 无法直接堆叠成张量。

```
文本1: "这部电影太棒了"      → 7个字符
文本2: "完全浪费时间的烂片"  → 9个字符
文本3: "值得一看"            → 4个字符

无法直接堆叠成 [batch, seq_len] 的张量！
```

### 3.2 collate_fn 的作用

collate_fn 接收一个批次的数据列表，返回处理后的批次。

```python
def collate_fn(batch):
    """
    batch: [sample1, sample2, ..., sampleN]
    返回: 处理后的批次数据
    """
    # 解包批次
    texts = [item['text'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    # 处理变长文本（padding）
    # ...
    
    return {
        'texts': texts,
        'labels': labels
    }
```

---

## 4. Padding - 处理变长序列的标准方法

### 4.1 标准流程

```
原始文本
    ↓
分词/数字化（转为ID）
    ↓
找到批次中最长长度
    ↓
其他序列用 <PAD> 填充
    ↓
生成 mask 标记真实数据位置
    ↓
输入模型
```

### 4.2 代码示例

```python
def padding_collate_fn(batch, max_len=100, pad_id=0):
    """
    带 padding 的 collate 函数
    """
    texts = []
    labels = []
    lengths = []
    masks = []
    
    for item in batch:
        # 文本转ID（这里简化处理）
        text_ids = text_to_ids(item['text'])
        
        # 记录真实长度
        true_len = len(text_ids)
        lengths.append(true_len)
        
        # Padding
        if len(text_ids) > max_len:
            text_ids = text_ids[:max_len]  # 截断
        else:
            text_ids = text_ids + [pad_id] * (max_len - len(text_ids))
        
        texts.append(text_ids)
        labels.append(item['label'])
        
        # 生成 mask：1表示真实数据，0表示padding
        mask = [1] * min(true_len, max_len) + [0] * (max_len - min(true_len, max_len))
        masks.append(mask)
    
    return {
        'text_ids': torch.tensor(texts),      # [batch, max_len]
        'labels': torch.tensor(labels),         # [batch]
        'lengths': torch.tensor(lengths),     # [batch]
        'masks': torch.tensor(masks)            # [batch, max_len]
    }
```

### 4.3 Padding 的要点

| 概念 | 说明 |
|-----|------|
| `<PAD>` token | 填充符号，通常ID为0 |
| `padding_idx` | Embedding层中忽略padding的索引 |
| `mask` | 标记哪些位置是真实数据 |
| 截断 | 超过max_len的序列从右侧截断 |

---

## 5. 在模型中使用 mask

```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=0  # 忽略padding位置
        )
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, text_ids, masks):
        # embedding: [batch, max_len] -> [batch, max_len, embed_dim]
        embedded = self.embedding(text_ids)
        
        # 使用 mask 做平均池化
        mask_expanded = masks.unsqueeze(-1)  # [batch, max_len, 1]
        masked_embedded = embedded * mask_expanded
        
        # 只对真实token求平均
        sum_embedded = masked_embedded.sum(dim=1)
        avg_embedded = sum_embedded / masks.sum(dim=1, keepdim=True).clamp(min=1)
        
        return self.fc(avg_embedded)
```

---

## 6. 完整训练模板

```python
# 1. 准备数据
dataset = TextDataset(data)
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=padding_collate_fn
)

# 2. 模型、损失、优化器
model = TextClassifier(vocab_size=10000, embed_dim=128, num_classes=2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 3. 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch in dataloader:
        # 数据移到GPU
        text_ids = batch['text_ids'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['masks'].to(device)
        
        # 前向传播
        outputs = model(text_ids, masks)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 7. 速查表

### 7.1 Dataset 模板

```python
class MyDataset(Dataset):
    def __init__(self, data_path):
        # 加载数据
        self.data = load_data(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'input': process_input(sample),
            'target': sample['label']
        }
```

### 7.2 DataLoader 模板

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,          # 训练时True，测试时False
    num_workers=4,
    collate_fn=collate_fn  # NLP必备
)

# 遍历
for batch in dataloader:
    # batch 是一个字典/列表，包含一批样本
    pass
```

### 7.3 collate_fn 模板

```python
def collate_fn(batch):
    """
    batch: [sample1, sample2, ...]
    每个 sample 是 __getitem__ 返回的字典
    """
    # 解包
    inputs = [item['input'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
    
    # 处理变长数据（如果需要）
    # ...
    
    return {
        'inputs': inputs,
        'targets': targets
    }
```

---

## 8. 下节课预告

正式进入 **NLP基础**！

1. **文本预处理** - 分词、清洗、规范化
2. **词嵌入（Word2Vec）** - 文本如何变成向量
3. **Transformer架构** - Attention is All You Need！

准备进入自然语言处理的世界！

---

*学习日期：2026-05-24*
