# 中文命名实体识别（NER）实战 - BERT 序列标注

**项目**: 人名/地名/组织名识别  
**技术栈**: BERT-base-chinese, PyTorch, Transformers  
**任务**: 序列标注（BIO格式）

---

## 1. 项目概览

### 1.1 什么是 NER

命名实体识别（Named Entity Recognition）是从文本中识别**专有名词**的任务：

- **人名（PER）**: 马云、李彦宏、马化腾
- **地名（LOC）**: 北京、杭州、深圳
- **组织名（ORG）**: 阿里巴巴、腾讯、百度

### 1.2 与文本分类的区别

| 任务 | 输入 | 输出 | 模型 |
|------|------|------|------|
| 文本分类 | 句子 | 1个标签 | `BertForSequenceClassification` |
| NER | 句子 | N个标签（每字一个） | `BertForTokenClassification` |

---

## 2. BIO 标注格式

### 2.1 标注规则

| 标签 | 含义 | 示例 |
|------|------|------|
| `B-PER` | 人名开始 | **马**云 |
| `I-PER` | 人名内部 | 马**云** |
| `B-LOC` | 地名开始 | **杭**州 |
| `I-LOC` | 地名内部 | 杭**州** |
| `B-ORG` | 组织开始 | **阿**里巴巴 |
| `I-ORG` | 组织内部 | 阿**里**巴巴 |
| `O` | 非实体 | 在、了、的 |

### 2.2 标注示例

```
文本: 马云在杭州创立了阿里巴巴公司
标注: B-PER I-PER O B-LOC I-LOC O O O B-ORG I-ORG I-ORG I-ORG O O
对应: 马   云  在 杭   州  创 立 了 阿    里    巴    巴    公 司
```

---

## 3. 核心代码解析

### 3.1 标签定义

```python
LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
```

### 3.2 数据格式

```python
raw_data = [
    ("马云在杭州创立了阿里巴巴公司",
     ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 
      'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O']),
    # ...
]
```

**关键**：文本长度必须等于标签列表长度！

### 3.3 Dataset 处理 - 标签对齐

```python
class NERDataset(Dataset):
    def __getitem__(self, idx):
        text = self.texts[idx]
        char_labels = self.labels[idx]
        
        # Tokenize（BERT会自动添加[CLS]和[SEP]）
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,  # 添加 [CLS] + [SEP]
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 关键：标签对齐！
        # [CLS] 和 [SEP] 的标签设为 -100（PyTorch会忽略）
        label_ids = [-100]  # [CLS] 位置
        
        for label in char_labels:
            label_ids.append(self.label2id[label])
        
        label_ids.append(-100)  # [SEP] 位置
        
        # Padding 位置也设为 -100
        while len(label_ids) < self.max_length:
            label_ids.append(-100)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }
```

### 3.4 标签对齐原理

```
文本:        马   云   在   杭   州   ...
原始标签:    B-PER I-PER O    B-LOC I-LOC ...
            ↓ 添加特殊token ↓
BERT输入:   [CLS] 马   云   在   杭   州   ... [SEP] [PAD] [PAD]
BERT标签:   -100 B-PER I-PER O    B-LOC I-LOC ... -100  -100  -100
             ↑                                    ↑
           忽略                                 忽略

计算loss时，-100 的位置会被自动跳过
```

---

## 4. 模型加载

### 4.1 Token Classification 模型

```python
from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=7,  # O + 6个B/I标签
    output_attentions=False,
    output_hidden_states=False
)
```

### 4.2 模型输出

```python
# 输出形状: [batch_size, seq_len, num_labels]
# 每个 token 位置都有一个分类结果
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
logits = outputs.logits  # [batch, 128, 7]

# 取每个位置概率最高的标签
predictions = torch.argmax(logits, dim=-1)  # [batch, 128]
```

### 4.3 模型结构对比

```
SequenceClassification（文本分类）:
  BERT Encoder → [CLS] token → Classifier → 1个结果
  
TokenClassification（NER）:
  BERT Encoder → All tokens → Classifier(每个位置) → N个结果
  
输出对比:
  分类: [batch, num_labels]
  NER:  [batch, seq_len, num_labels]
```

---

## 5. 实体提取

### 5.1 从 BIO 还原实体

```python
def extract_entities(text, predictions):
    """从预测标签还原完整实体"""
    entities = []
    current_entity = None
    current_tokens = []
    
    # 注意：predictions[0]是[CLS]，从predictions[1]开始对应text[0]
    for i, char in enumerate(text):
        pred_idx = i + 1  # 跳过[CLS]
        pred = predictions[pred_idx]
        
        if pred == -100:
            continue
            
        label = ID2LABEL[pred]
        
        if label.startswith('B-'):
            # 新实体开始
            if current_entity:
                entities.append((current_entity, ''.join(current_tokens)))
            current_entity = label[2:]  # PER/LOC/ORG
            current_tokens = [char]
            
        elif label.startswith('I-') and current_entity == label[2:]:
            # 继续当前实体
            current_tokens.append(char)
            
        else:
            # 实体结束
            if current_entity:
                entities.append((current_entity, ''.join(current_tokens)))
                current_entity = None
                current_tokens = []
    
    # 处理最后一个实体
    if current_entity:
        entities.append((current_entity, ''.join(current_tokens)))
    
    return entities

# 示例
# 输入: "马云在杭州创立了阿里巴巴公司"
# 预测: [O, B-PER, I-PER, O, B-LOC, I-LOC, O, O, O, B-ORG, I-ORG, I-ORG, I-ORG, O, O]
# 输出: [('PER', '马云'), ('LOC', '杭州'), ('ORG', '阿里巴巴')]
```

---

## 6. 训练配置

### 6.1 超参数

```python
EPOCHS = 10              # NER通常需要更多epoch
LEARNING_RATE = 2e-5     # 标准微调学习率
BATCH_SIZE = 4           # 根据显存调整
MAX_LENGTH = 128         # 短文本够用
```

### 6.2 训练循环

```python
model.train()
for batch in train_loader:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels  # 形状: [batch, seq_len]，包含-100
    )
    
    loss = outputs.loss  # 自动忽略-100位置
    loss.backward()
    optimizer.step()
```

**注意**：`labels` 中的 `-100` 会被 `CrossEntropyLoss` 自动忽略，只计算有效位置的损失。

---

## 7. 评估方法

### 7.1 Token-level 评估

```python
from sklearn.metrics import f1_score, accuracy_score

# 计算每个token的准确率
accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, average='micro')
```

### 7.2 Entity-level 评估（更严格）

```python
# 精确匹配整个实体才算正确
true_entities = extract_entities(text, true_labels)
pred_entities = extract_entities(text, pred_labels)

# 计算实体级的P/R/F1
# 精确率 = 正确识别的实体数 / 识别的实体总数
# 召回率 = 正确识别的实体数 / 真实实体总数
```

---

## 8. 常见错误与修复

### 8.1 标签长度不匹配

```python
# ❌ 错误示例
text = "乔布斯在加利福尼亚州创办了苹果公司"  # 16个字符
labels = ['B-PER', 'I-PER', 'O', ...]  # 14个标签 ← 长度不匹配！

# ✅ 修复后
text = "乔布斯在加利福尼亚州创办了苹果公司"
labels = ['B-PER', 'I-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 
          'I-LOC', 'I-LOC', 'I-LOC', 'I-LOC', 'O', 'O', 'O',
          'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O']
# 19个标签 = 19个字符（包括中文标点）
```

### 8.2 中英文混合问题

```python
# ❌ 英文单词会被拆分成多个字符
"SpaceX"  # len=6，需要6个标签

# ✅ 建议统一用中文实体名
"太空探索公司"  # 明确的中文实体
```

### 8.3 预测时索引错位

```python
# ❌ 错误：predictions[0]对应text[0]，但predictions[0]是[CLS]
for char, pred in zip(text, predictions):
    ...

# ✅ 正确：predictions[1]才开始对应text[0]
for i, char in enumerate(text):
    pred = predictions[i + 1]  # 跳过[CLS]
    ...
```

---

## 9. 进阶方向

### 9.1 添加 CRF 层

```python
# 使用CRF提升标注一致性（避免 B-PER 后面直接跟 I-LOC）
from torchcrf import CRF

class BertCRF(nn.Module):
    def __init__(self):
        self.bert = BertForTokenClassification(...)
        self.crf = CRF(num_tags=7, batch_first=True)
    
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids, attention_mask)
        emissions = outputs.logits
        loss = -self.crf(emissions, labels, mask=attention_mask.bool())
        return loss
```

### 9.2 使用更优的中文模型

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| bert-base-chinese | 基础 | 通用 |
| chinese-roberta-wwm-ext | 全词掩码 | 中文理解更好 |
| macbert-base-chinese | 纠正BERT缺点 | 中文NER推荐 |
| ernie-3.0-base-zh | 知识增强 | 实体识别强 |

### 9.3 真实数据集

- **CLUENER2020**: 10种实体类型，大规模
- **MSRA**: 微软亚洲研究院中文NER数据集
- **人民日报**: 1998年人民日报标注数据

---

## 10. 总结

### 关键知识点

1. **BIO标注**: B-开始, I-内部, O-非实体
2. **标签对齐**: [CLS]/[SEP]/padding → -100
3. **索引偏移**: predictions[1:] 对应 text[0:]
4. **实体提取**: B-I配对还原完整实体
5. **模型选择**: `BertForTokenClassification`

### 核心代码模板

```python
# 1. 定义标签
LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']

# 2. 准备数据（确保长度匹配）
raw_data = [(text, labels)]  # len(text) == len(labels)

# 3. Dataset处理标签对齐
def __getitem__(self, idx):
    label_ids = [-100]  # [CLS]
    for label in char_labels:
        label_ids.append(LABEL2ID[label])
    label_ids.append(-100)  # [SEP]
    # padding到max_length...

# 4. 加载模型
model = BertForTokenClassification.from_pretrained(
    'bert-base-chinese', num_labels=7
)

# 5. 训练（自动忽略-100位置）
outputs = model(input_ids, attention_mask, labels=label_ids)
loss = outputs.loss

# 6. 实体提取（注意索引+1）
for i, char in enumerate(text):
    pred = predictions[i + 1]  # 跳过[CLS]
    ...
```

---

**实践项目文件**: `examples/nlp/08_chinese_ner.py`

**运行命令**:
```bash
python examples/nlp/08_chinese_ner.py
```
