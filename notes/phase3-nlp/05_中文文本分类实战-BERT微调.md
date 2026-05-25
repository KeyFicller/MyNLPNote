# 中文文本分类实战 - BERT 微调项目

**项目**: 电商评论情感分析  
**技术栈**: BERT-base-chinese, PyTorch, Transformers  
**任务**: 二分类（正面/负面）

---

## 1. 项目架构

```
中文文本分类项目
├── 数据准备（模拟电商评论）
├── Dataset/DataLoader 构建
├── BERT 模型加载
├── 训练配置（AdamW + 学习率调度）
├── 训练循环（Train/Eval）
├── 模型评估
├── 保存与加载
└── 新样本预测
```

---

## 2. 数据准备

### 2.1 数据格式

```python
raw_data = [
    ("这个手机太棒了，拍照效果很好！", 1),  # 正面
    ("太差了，完全不能用", 0),              # 负面
    ...
]
```

### 2.2 数据划分

```python
from sklearn.model_selection import train_test_split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels,
    test_size=0.2,      # 20% 测试集
    random_state=42,
    stratify=labels     # 保持类别比例
)
```

> **stratify=labels**: 确保训练集和测试集中正负样本比例相同，避免数据不均衡。

---

## 3. 自定义 Dataset

### 3.1 核心流程

```
文本 → Tokenizer → {input_ids, attention_mask, labels}
```

### 3.2 代码实现

```python
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,      # [CLS] + 文本 + [SEP]
            max_length=self.max_length,
            padding='max_length',        # 填充到 max_length
            truncation=True,             # 超长截断
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx])
        }
```

### 3.3 关键参数解释

| 参数 | 值 | 说明 |
|------|-----|------|
| `max_length` | 128 | BERT 最大 512，短文本用 128 即可 |
| `padding` | `max_length` | 填充到固定长度，便于 batch 训练 |
| `truncation` | True | 超长文本从右侧截断 |
| `return_tensors` | `pt` | 返回 PyTorch tensor |

---

## 4. BERT 模型加载

### 4.1 模型选择

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2,           # 二分类
    output_attentions=False,
    output_hidden_states=False
)
```

### 4.2 模型结构

```
输入文本 → Tokenizer → input_ids[128] + attention_mask[128]
                            ↓
                    BERT Encoder (12层 Transformer)
                            ↓
                    [CLS] token 的隐藏状态 [768]
                            ↓
                    Dropout(0.1) → Linear(768 → 2)
                            ↓
                         分类结果
```

### 4.3 参数量

| 参数 | 数值 |
|------|------|
| 总参数量 | ~102M |
| BERT Encoder | ~85M |
| 分类头 | ~1.5K |

---

## 5. 训练配置

### 5.1 优化器：AdamW

```python
from torch.optim import AdamW  # 注意：新版从 torch.optim 导入

optimizer = AdamW(
    model.parameters(),
    lr=2e-5,           # BERT 微调常用学习率（很小！）
    eps=1e-8
)
```

> **为什么学习率要很小？**
> - BERT 已经预训练好了，只需要微调
> - 学习率太大会破坏预训练权重
> - 经验值：2e-5 ~ 5e-5

### 5.2 学习率调度

```python
from transformers import get_linear_schedule_with_warmup

# 线性衰减
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)
```

**Warmup 作用**：训练初期用小学习率热身，防止震荡。

### 5.3 梯度裁剪

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

> 防止梯度爆炸，RNN/Transformer 训练的标准做法。

---

## 6. 训练循环

### 6.1 流程图

```
Epoch Loop
    └── Batch Loop
            ├── Forward → loss, logits
            ├── Backward → loss.backward()
            ├── Clip Grad → clip_grad_norm_()
            ├── Update → optimizer.step()
            └── Step LR → scheduler.step()
```

### 6.2 关键代码

```python
model.train()
for batch in dataloader:
    # 前向
    outputs = model(**batch)
    loss = outputs.loss
    
    # 反向
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 更新
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

### 6.3 评估模式

```python
model.eval()
with torch.no_grad():  # 不计算梯度，节省内存
    for batch in test_loader:
        outputs = model(**batch)
        ...
```

> **`model.eval()` 的重要性**：关闭 Dropout 和 BatchNorm 的随机性，确保预测稳定。

---

## 7. 网络问题解决方案

### 7.1 使用镜像源

```python
import os

# HF-Mirror 镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 或 ModelScope 镜像
os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'
```

### 7.2 设置缓存目录

```python
os.environ['HF_HOME'] = './.cache/huggingface'
```

模型下载后会缓存，下次直接从本地加载。

---

## 8. 模型保存与加载

### 8.1 保存

```python
# 保存模型权重 + tokenizer
model.save_pretrained('./saved_models/chinese_sentiment_bert')
tokenizer.save_pretrained('./saved_models/chinese_sentiment_bert')

# 保存配置
config = {
    'model_name': 'bert-base-chinese',
    'num_labels': 2,
    'max_length': 128,
    'final_accuracy': 0.95
}
```

### 8.2 加载

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('./saved_models/chinese_sentiment_bert')
model = BertForSequenceClassification.from_pretrained('./saved_models/chinese_sentiment_bert')
```

---

## 9. 新样本预测

```python
text = "这家餐厅的菜太好吃了！"

# 编码
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# 预测
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    pred = torch.argmax(logits).item()
    
sentiment = '正面' if pred == 1 else '负面'
confidence = probs[0][pred].item()

print(f"{sentiment} (置信度: {confidence:.2%})")
```

---

## 10. 关键知识点总结

| 知识点 | 说明 |
|--------|------|
| **微调 vs 预训练** | 预训练需要大量无标注数据，微调只需少量标注数据 |
| **学习率选择** | BERT 微调用 2e-5，不能太大 |
| **batch size** | 显存小用 4-8，显存大用 16-32 |
| **max_length** | 短文本 128，长文本 256-512 |
| **数据增强** | 同义词替换、回译等可以增加训练数据 |
| **早停机制** | 验证集准确率不再提升时停止，防止过拟合 |

---

## 11. 常见问题

### Q1: 为什么准确率上不去？
- 检查数据是否标注正确
- 增加训练数据量
- 尝试更大的模型（如 RoBERTa）
- 调整学习率

### Q2: 显存不够怎么办？
- 减小 `batch_size`
- 减小 `max_length`
- 使用梯度累积
- 使用更小的模型（如 DistilBERT）

### Q3: 如何加快训练？
- 使用 GPU（CUDA）
- 增大 `batch_size`
- 使用混合精度训练（fp16）
- 冻结 BERT 前几层，只训练分类头

---

## 12. 扩展建议

1. **使用真实数据集**: 如 weibo_senti_100k、online_shopping_10_cats
2. **尝试其他中文模型**: macbert-base-chinese、chinese-roberta-wwm-ext
3. **多分类任务**: 正面/中性/负面 三分类
4. **更复杂的任务**: 细粒度情感分析（如 1-5 星评分）
5. **部署上线**: 使用 Flask/FastAPI 封装成 API 服务

---

**实践项目文件**: `examples/nlp/06_chinese_text_classification.py`

**运行命令**:
```bash
python examples/nlp/06_chinese_text_classification.py
```
