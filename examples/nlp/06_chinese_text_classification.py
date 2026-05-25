#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文文本分类实战 - BERT 微调项目
=================================
项目名称：电商评论情感分析
任务：使用 bert-base-chinese 微调，对中文商品评论进行正面/负面分类

学习目标：
- 使用真实数据进行 BERT 微调
- 掌握完整的训练、评估、保存流程
- 学习中文 NLP 任务的特殊处理

运行前请确保：
- 虚拟环境已激活
- 已安装 transformers, datasets, torch
"""

import os

# ========== 配置 Hugging Face 镜像源（解决网络问题）==========
# 方案1：使用 HF-Mirror 镜像（推荐，国内访问快）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 方案2：使用 ModelScope 镜像（阿里出品，中文模型友好）
# 取消下面的注释来使用 ModelScope
# os.environ['HF_ENDPOINT'] = 'https://www.modelscope.cn'

# 配置缓存目录（可选）
os.environ['HF_HOME'] = os.path.join(os.path.dirname(__file__), '..', '..', '.cache', 'huggingface')
os.makedirs(os.environ['HF_HOME'], exist_ok=True)

print(f"🌐 使用镜像源: {os.environ['HF_ENDPOINT']}")
print(f"📁 模型缓存目录: {os.environ['HF_HOME']}")

# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

print("=" * 70)
print("中文文本分类实战 - 电商评论情感分析")
print("=" * 70)

# ============================================================
# 第一部分：准备数据
# ============================================================
print("\n" + "=" * 70)
print("第一部分：准备数据")
print("=" * 70)

# 模拟电商评论数据（真实场景中可以从文件读取或数据库获取）
# 数据格式: (评论文本, 标签)  1=正面, 0=负面
raw_data = [
    # 正面评论 (标签=1)
    ("这个手机太棒了，拍照效果特别好，很喜欢！", 1),
    ("物流很快，包装完整，商品质量超出预期，好评！", 1),
    ("客服态度很好，解决问题很耐心，会继续购买。", 1),
    ("性价比很高，做工精细，功能齐全，非常满意。", 1),
    ("用了几天才来评价，确实不错，推荐给朋友了。", 1),
    ("外观漂亮，手感舒适，运行流畅， worth buying！", 1),
    ("这家店的商品质量很好，已经是第三次购买了。", 1),
    ("产品符合描述，没有色差，穿着很舒服。", 1),
    ("配送及时，安装师傅很专业，服务态度好。", 1),
    ("价格实惠，质量可靠，会再次光顾的。", 1),
    ("包装精美，送人很有面子，对方很喜欢。", 1),
    ("功能强大，操作简单，老人也能轻松使用。", 1),
    ("音质清晰，续航能力不错，整体很满意。", 1),
    ("尺寸标准，面料柔软，洗了不褪色。", 1),
    ("设计新颖，很有创意，朋友都说好看。", 1),
    ("充电速度快，电池耐用，出差必备。", 1),
    ("售后服务很好，有问题及时处理，放心购买。", 1),
    ("食材新鲜，味道鲜美，包装也很用心。", 1),
    ("课程内容丰富，老师讲解清晰，收获很大。", 1),
    ("酒店环境优雅，服务周到，下次还会选择。", 1),
    # 负面评论 (标签=0)
    ("太差了，收到的商品破损，客服还不理人。", 0),
    ("物流慢得要死，等了一周才到，包装还烂了。", 0),
    ("完全不符合描述，色差严重，退货还麻烦。", 0),
    ("质量一般，用了两天就出问题了，后悔购买。", 0),
    ("客服态度恶劣，问个问题爱答不理的。", 0),
    ("价格贵不说，质量还差，根本不值这个价。", 0),
    ("充电发热严重，担心安全问题，不敢用了。", 0),
    ("尺码不准，穿着不合身，退货运费还要自己出。", 0),
    ("声音杂音很大，听不清，产品质量有问题。", 0),
    ("包装简陋，收到时盒子都压扁了，东西也坏了。", 0),
    ("广告夸大其词，实际效果差远了，上当的感觉。", 0),
    ("安装后漏水，找售后推三阻四，体验极差。", 0),
    ("配料表不清楚，吃起来味道怪怪的，不敢吃了。", 0),
    ("课程退款困难，态度强硬，消费者权益没保障。", 0),
    ("房间卫生差，床单有污渍，要求换房被拒绝。", 0),
    ("软件闪退严重，根本无法正常使用，浪费钱。", 0),
    ("电池不耐用，半天就要充电，宣传严重不符。", 0),
    ("面料粗糙，穿着扎皮肤，洗一次就变形了。", 0),
    ("承诺的发票迟迟不给，催了多次都没用。", 0),
    ("系统卡顿，运行速度慢，影响工作效率。", 0),
]

print(f"\n📊 数据集统计:")
print(f"   总样本数: {len(raw_data)}")
print(f"   正面评论: {sum(1 for _, label in raw_data if label == 1)} 条")
print(f"   负面评论: {sum(1 for _, label in raw_data if label == 0)} 条")

# 划分训练集和测试集
texts = [item[0] for item in raw_data]
labels = [item[1] for item in raw_data]

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"\n📈 数据划分:")
print(f"   训练集: {len(train_texts)} 条")
print(f"   测试集: {len(test_texts)} 条")

# ============================================================
# 第二部分：创建 Dataset 和 DataLoader
# ============================================================
print("\n" + "=" * 70)
print("第二部分：创建 Dataset 和 DataLoader")
print("=" * 70)

class CommentDataset(Dataset):
    """中文评论数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,      # 添加 [CLS] 和 [SEP]
            max_length=self.max_length,    # 最大长度
            padding='max_length',        # 填充到 max_length
            truncation=True,             # 超长截断
            return_tensors='pt'          # 返回 PyTorch tensor
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),      # [max_length]
            'attention_mask': encoding['attention_mask'].flatten(),  # [max_length]
            'labels': torch.tensor(label, dtype=torch.long)    # 标量
        }


# 加载 tokenizer
print("\n🔤 加载 Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(f"   Tokenizer 加载完成")
print(f"   词表大小: {tokenizer.vocab_size}")

# 创建 Dataset
train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
test_dataset = CommentDataset(test_texts, test_labels, tokenizer)

# 创建 DataLoader
BATCH_SIZE = 4  # 小批量，适合学习

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # 训练时打乱
    num_workers=0  # Windows 建议设为 0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # 测试时不打乱
    num_workers=0
)

print(f"\n📦 DataLoader 创建完成:")
print(f"   训练批次: {len(train_loader)} (batch_size={BATCH_SIZE})")
print(f"   测试批次: {len(test_loader)} (batch_size={BATCH_SIZE})")

# 查看一个样本
sample = train_dataset[0]
print(f"\n📝 样本示例:")
print(f"   Input IDs 形状: {sample['input_ids'].shape}")
print(f"   Attention Mask 形状: {sample['attention_mask'].shape}")
print(f"   Label: {sample['labels']}")

# 解码回文本查看
sample_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
print(f"   解码文本: {sample_text}")

# ============================================================
# 第三部分：加载模型
# ============================================================
print("\n" + "=" * 70)
print("第三部分：加载 BERT 模型")
print("=" * 70)

print("\n🤖 加载 bert-base-chinese...")
# 加载预训练 BERT，num_labels=2 表示二分类
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=2,           # 正面/负面 二分类
    output_attentions=False,
    output_hidden_states=False
)

print(f"   模型加载完成!")
print(f"   模型类型: {type(model).__name__}")
print(f"   分类类别: {model.num_labels}")

# 查看模型结构（简要）
print(f"\n📐 模型结构:")
print(f"   BERT Encoder: 12层 Transformer")
print(f"   隐藏层维度: 768")
print(f"   分类头: Linear(768 -> 2)")

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 参数量统计:")
print(f"   总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"   可训练参数: {trainable_params:,}")

# 设备选择
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"\n💻 使用设备: {device}")
model = model.to(device)

# ============================================================
# 第四部分：训练配置
# ============================================================
print("\n" + "=" * 70)
print("第四部分：训练配置")
print("=" * 70)

# 训练超参数
EPOCHS = 5
LEARNING_RATE = 2e-5          # BERT 微调常用学习率
WARMUP_STEPS = 0
MAX_GRAD_NORM = 1.0           # 梯度裁剪阈值

# 优化器 - AdamW 是 Transformer 的标准选择
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)

# 学习率调度器 - 线性衰减
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps
)

# 损失函数（交叉熵）
criterion = nn.CrossEntropyLoss()

print(f"\n⚙️ 训练配置:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   优化器: AdamW")
print(f"   调度器: Linear warmup & decay")
print(f"   总训练步数: {total_steps}")

# ============================================================
# 第五部分：训练循环
# ============================================================
print("\n" + "=" * 70)
print("第五部分：训练模型")
print("=" * 70)

# 存储训练历史
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

best_accuracy = 0.0

for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch + 1}/{EPOCHS}")
    print("-" * 50)
    
    # ========== 训练阶段 ==========
    model.train()
    total_train_loss = 0
    train_correct = 0
    train_total = 0
    
    # 使用 tqdm 显示进度条
    progress_bar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch in progress_bar:
        # 数据移至设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        
        # 更新参数
        optimizer.step()
        scheduler.step()
        
        # 统计
        total_train_loss += loss.item()
        predictions = torch.argmax(logits, dim=-1)
        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = total_train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    training_history['train_loss'].append(avg_train_loss)
    training_history['train_acc'].append(train_accuracy)
    
    print(f"   Train Loss: {avg_train_loss:.4f} | Acc: {train_accuracy:.2%}")
    
    # ========== 验证阶段 ==========
    model.eval()
    total_val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_val_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = total_val_loss / len(test_loader)
    val_accuracy = val_correct / val_total
    
    training_history['val_loss'].append(avg_val_loss)
    training_history['val_acc'].append(val_accuracy)
    
    print(f"   Val Loss:   {avg_val_loss:.4f} | Acc: {val_accuracy:.2%}")
    
    # 保存最佳模型
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        print(f"   ✨ 最佳模型！准确率: {best_accuracy:.2%}")

print(f"\n🎉 训练完成！最佳验证准确率: {best_accuracy:.2%}")

# ============================================================
# 第六部分：详细评估
# ============================================================
print("\n" + "=" * 70)
print("第六部分：模型评估")
print("=" * 70)

# 获取最终预测
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算指标
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n📊 最终评估结果:")
print(f"   准确率 (Accuracy): {accuracy:.2%}")

print(f"\n📋 分类报告:")
target_names = ['负面', '正面']
print(classification_report(all_labels, all_preds, target_names=target_names))

# 查看预测示例
print(f"\n🔍 预测示例:")
id2label = {0: '负面', 1: '正面'}
for i in range(min(5, len(test_texts))):
    text = test_texts[i]
    true_label = id2label[test_labels[i]]
    pred_label = id2label[all_preds[i]]
    correct = "✅" if test_labels[i] == all_preds[i] else "❌"
    print(f"   {correct} 文本: {text[:30]}...")
    print(f"      真实: {true_label} | 预测: {pred_label}")
    print()

# ============================================================
# 第七部分：保存模型
# ============================================================
print("\n" + "=" * 70)
print("第七部分：保存模型")
print("=" * 70)

# 创建保存目录
save_dir = './saved_models/chinese_sentiment_bert'
os.makedirs(save_dir, exist_ok=True)

# 保存模型和 tokenizer
print(f"\n💾 保存模型到: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 保存训练配置
config_info = {
    'model_name': 'bert-base-chinese',
    'num_labels': 2,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'max_length': 128,
    'final_accuracy': accuracy
}

import json
with open(os.path.join(save_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
    json.dump(config_info, f, ensure_ascii=False, indent=2)

print(f"   ✅ 模型权重保存完成")
print(f"   ✅ Tokenizer 保存完成")
print(f"   ✅ 训练配置保存完成")

# ============================================================
# 第八部分：模型加载与预测演示
# ============================================================
print("\n" + "=" * 70)
print("第八部分：模型加载与预测")
print("=" * 70)

print(f"\n🔄 演示：从保存的模型加载并预测...")

# 加载保存的模型
loaded_tokenizer = BertTokenizer.from_pretrained(save_dir)
loaded_model = BertForSequenceClassification.from_pretrained(save_dir)
loaded_model = loaded_model.to(device)
loaded_model.eval()

print(f"   ✅ 模型加载完成")

# 新样本预测
test_samples = [
    "这家餐厅的菜太好吃了，服务员态度也很好！",
    "买到的商品完全不能用，要求退货还要收手续费，太坑了！",
    "一般般吧，没有想象中那么好，但也不差。",
]

print(f"\n📝 新样本预测:")
for text in test_samples:
    # Tokenize
    inputs = loaded_tokenizer(
        text,
        padding='max_length',
        max_length=128,
        truncation=True,
        return_tensors='pt'
    )
    
    # 移至设备
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # 预测
    with torch.no_grad():
        outputs = loaded_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][pred].item()
    
    sentiment = '正面' if pred == 1 else '负面'
    print(f"   文本: {text}")
    print(f"   预测: {sentiment} (置信度: {confidence:.2%})")
    print()

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("项目总结")
print("=" * 70)

print("""
✅ 本项目完成了：

1. 数据准备
   - 创建了40条中文电商评论数据
   - 划分训练集(32条)和测试集(8条)

2. 数据处理
   - 使用 bert-base-chinese Tokenizer
   - 实现自定义 Dataset 和 DataLoader
   - 处理 padding 和 truncation

3. 模型训练
   - 加载预训练 BERT
   - 使用 AdamW 优化器
   - 线性学习率调度
   - 梯度裁剪防止爆炸
   - 训练 5 个 epoch

4. 模型评估
   - 计算准确率和分类报告
   - 查看预测示例

5. 模型保存与加载
   - 保存模型权重和 tokenizer
   - 演示从新样本预测

📚 关键知识点：
   • BERT 微调需要较小的学习率 (2e-5)
   • 中文任务使用 bert-base-chinese
   • 训练时 shuffle=True，测试时 shuffle=False
   • 使用 tqdm 可以清晰看到训练进度
   • 梯度裁剪是 Transformer 训练的标准做法

🚀 下一步建议：
   1. 使用真实数据集（如 weibo_senti_100k）
   2. 尝试其他中文预训练模型（如 macbert, roberta-wwm）
   3. 添加早停机制（Early Stopping）
   4. 使用更大的 batch size 和更多的 epoch
   5. 尝试多分类任务（正面/中性/负面）
""")

print("\n" + "=" * 70)
print("中文文本分类实战完成！🎉")
print("=" * 70)
