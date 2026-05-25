#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文星级评价分类实战 - BERT 微调项目（多分类）
=========================================
项目名称：电商评论星级预测
任务：使用 bert-base-chinese 微调，对中文商品评论进行 1-5 星评级

学习目标：
- 掌握多分类任务（5分类）的完整流程
- 理解 BERT 微调的关键参数
- 学习中文多分类任务的评估方法

星级含义：
1星 = 很差, 2星 = 较差, 3星 = 一般, 4星 = 较好, 5星 = 很好
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

# 模拟用户评论和星级评价
# 星级含义：1=很差, 2=较差, 3=一般, 4=较好, 5=很好
raw_data = [
    # ========== 1星 - 很差 (20条) ==========
    ("完全不能用，刚收到就坏了，退货还一直拖着，太坑了！", 1),
    ("假冒伪劣产品，和描述完全不一样，大家千万别买。", 1),
    ("物流慢得要死，等了两周才到，包装还烂得不行。", 1),
    ("客服态度恶劣，问了几个问题就不耐烦，直接拉黑我。", 1),
    ("质量太差了，用了一天就散架了，钱全打水漂。", 1),
    ("收到货发现是二手的，有使用痕迹，卖家不承认。", 1),
    ("充电发热严重，担心会爆炸，已经扔了不敢用。", 1),
    ("图片和实物差距巨大，严重欺骗消费者，要求退货。", 1),
    ("安装师傅不专业，装完还漏水，打电话不接。", 1),
    ("配料表造假，吃完肚子不舒服，食品安全有问题。", 1),
    ("课程质量极差，老师念PPT，退款被拒绝。", 1),
    ("房间脏乱差，床单发黄，还有异味，要求换房被拒绝。", 1),
    ("软件闪退严重，根本打不开，客服也不解决。", 1),
    ("电池是假的，显示100%电量，半小时就没电了。", 1),
    ("衣服掉色严重，洗一次就废了，还染坏了其他衣服。", 1),
    ("承诺的发票迟迟不给，催了十几次都没用，影响报销。", 1),
    ("系统卡顿严重，打字都有延迟，影响工作效率。", 1),
    ("收到的货和下单的型号不一样，卖家说是发错了。", 1),
    ("说明书是英文的，没有中文版，老年人根本看不懂。", 1),
    ("保修卡是假的，打电话去厂家查询说没有这款产品。", 1),

    # ========== 2星 - 较差 (20条) ==========
    ("质量一般吧，用了一周就出现小毛病，不太满意。", 2),
    ("物流比预计慢了好几天，包装也有点破损，凑合用吧。", 2),
    ("功能没描述的那么强大，实际效果打折，有点失望。", 2),
    ("客服回复太慢了，等半天才回一句，问题也没解决。", 2),
    ("价格偏贵，质量配不上这个价位，性价比不高。", 2),
    ("外观还行，但做工粗糙，细节处理不到位。", 2),
    ("安装过程很麻烦，说明书不清楚，费了很大劲。", 2),
    ("味道有点大，放了几天还是有异味，不敢马上用。", 2),
    ("尺码偏小，按正常码买的结果穿不上，懒得退了。", 2),
    ("音质有杂音，高音刺耳，不太适合听人声。", 2),
    ("电池续航一般，用半天就要充电，和宣传不符。", 2),
    ("软件界面太复杂，操作不友好，新手很难上手。", 2),
    ("颜色有色差，实际比图片暗一些，没那么好看。", 2),
    ("配件少了一个，联系客服说补发，等了快一周。", 2),
    ("网络连接不稳定，经常断线，体验不太好。", 2),
    ("按键手感不好，用久了手指疼，设计不合理。", 2),
    ("清洗麻烦，死角很多，不容易弄干净。", 2),
    ("售后服务响应慢，维修要排队等很久。", 2),
    ("升级后反而更卡了，想降级又不知道怎么操作。", 2),
    ("赠品质量很差，根本不能用，不如不送。", 2),

    # ========== 3星 - 一般 (20条) ==========
    ("还可以吧，没有特别好也没有特别差，对得起价格。", 3),
    ("物流正常，包装完好，产品功能基本满足需求。", 3),
    ("做工中规中矩，材质一般，不过这个价格也就这样了。", 3),
    ("客服态度还行，回答问题比较及时，解决了一半问题。", 3),
    ("性价比中等，不是最好的选择，但也不至于踩雷。", 3),
    ("用起来还行，有些小功能用不上，核心功能OK。", 3),
    ("外观设计普通，没什么亮点，也不难看。", 3),
    ("说明书还算清楚，照着步骤能装好，花了半小时。", 3),
    ("味道不算大，通风一天就差不多了。", 3),
    ("尺码还算准，稍微紧一点点，能接受。", 3),
    ("音质中规中矩，听个响没问题，发烧友绕道。", 3),
    ("续航能力一般，正常使用一天一充。", 3),
    ("软件功能齐全，但界面有点老旧，希望改进。", 3),
    ("颜色和图片描述基本一致，没啥惊喜。", 3),
    ("配件齐全，都是标准件，质量普通。", 3),
    ("WiFi连接正常，偶尔需要重新连一下。", 3),
    ("按键手感普通，用习惯了也就好了。", 3),
    ("清洗还算方便，就是有些边边角角不好处理。", 3),
    ("售后能解决基本问题，就是流程有点繁琐。", 3),
    ("升级后功能多了，但多了一些广告，喜忧参半。", 3),

    # ========== 4星 - 较好 (20条) ==========
    ("质量不错，用了两周感觉挺好，性价比挺高的。", 4),
    ("物流很快，包装严实，开箱体验很好，满意。", 4),
    ("功能很实用，操作简单，老人家也能轻松学会。", 4),
    ("客服很耐心，解决了我所有疑问，服务态度好。", 4),
    ("超出预期，比想象中好，会推荐给朋友。", 4),
    ("做工精细，细节处理到位，看着很高端。", 4),
    ("安装简单，有视频教程，20分钟搞定。", 4),
    ("基本没味道，拿出来就能用，材质应该不错。", 4),
    ("尺码标准，按推荐的买刚好合适，穿着舒服。", 4),
    ("音质清晰，低音有力，看电影很带感。", 4),
    ("续航不错，中度使用能用一天半，挺省心的。", 4),
    ("软件界面清爽，功能实用，运行流畅。", 4),
    ("颜色和图片描述基本一致，没有色差。", 4),
    ("配件质量都很好，不是那种廉价货，良心。", 4),
    ("网络稳定，没出现过断连的情况，信号强。", 4),
    ("按键反馈清晰，手感舒适，打字效率高。", 4),
    ("容易清洗，设计合理，没有卫生死角。", 4),
    ("售后响应快，保修政策清楚，买着放心。", 4),
    ("系统升级后有新功能，运行依然流畅。", 4),
    ("赠品很实用，不是凑数的，卖家很用心。", 4),

    # ========== 5星 - 很好 (20条) ==========
    ("太棒了！完全符合描述，质量超出预期，强烈推荐！", 5),
    ("物流神速，包装精美，开箱很有仪式感，爱了！", 5),
    ("功能强大，操作流畅，用户体验满分，良心产品！", 5),
    ("客服超级好，有问必答，还给了使用建议，感动！", 5),
    ("性价比超高，同价位没有对手，已经回购第二次！", 5),
    ("颜值爆表，做工精细，拿在手里很有质感，喜欢！", 5),
    ("安装超简单，傻瓜式操作，5分钟就搞定了！", 5),
    ("完全没异味，材质环保安全，给孩子用很放心。", 5),
    ("尺码完美，版型很好，显瘦显高，朋友都问在哪买的！", 5),
    ("音质绝了！高音清澈低音浑厚，听人声太享受了！", 5),
    ("续航逆天！用了三天还有电，出差神器！", 5),
    ("软件设计人性化，功能强大还不臃肿，体验一流！", 5),
    ("实物比图片还好看，颜色高级，没有色差！", 5),
    ("配件品质好，连小细节都处理得很到位，专业！", 5),
    ("网络稳得一批，打游戏延迟超低，上分利器！", 5),
    ("按键手感绝了，机械键盘的快感，码字停不下来！", 5),
    ("清洗超方便，设计合理，一冲就干净，省心！", 5),
    ("售后无忧，质保时间长，有问题随时能找到人！", 5),
    ("系统优化得好，用了半年依然流畅，不卡顿！", 5),
    ("赠品质量都超棒，不是糊弄人的，卖家太实在了！", 5),
]

print("=" * 70)
print("中文星级评价分类实战 - 电商评论 1-5 星预测")
print("=" * 70)

# ============================================================
# 第一部分：数据准备
# ============================================================
print("\n" + "=" * 70)
print("第一部分：数据准备")
print("=" * 70)

print(f"\n📊 数据集统计:")
print(f"   总样本数: {len(raw_data)}")
for star in range(1, 6):
    count = sum(1 for _, label in raw_data if label == star)
    bar = "⭐" * star + "☆" * (5 - star)
    print(f"   {bar} ({star}星 → 标签{star-1}): {count} 条")
print(f"\n⚠️ 标签转换: 1-5星 → 0-4索引 (BERT模型需要0-based索引)")

# 划分训练集和测试集
texts = [item[0] for item in raw_data]
# 标签映射：1-5星 → 0-4索引（BERT模型需要0-based索引）
labels = [item[1] - 1 for item in raw_data]  # 1→0, 2→1, 3→2, 4→3, 5→4

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

class CommentDataset(Dataset):
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
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# ============================================================
# 第二部分：创建 Dataset 和 DataLoader
# ============================================================
print("\n" + "=" * 70)
print("第二部分：创建 Dataset 和 DataLoader")
print("=" * 70)

print("\n🔤 加载 Tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print(f"   Tokenizer 加载完成")
print(f"   词表大小: {tokenizer.vocab_size}")

# 创建 Dataset
train_dataset = CommentDataset(train_texts, train_labels, tokenizer)
test_dataset = CommentDataset(test_texts, test_labels, tokenizer)

print(f"\n📦 Dataset 创建完成:")
print(f"   训练样本: {len(train_dataset)}")
print(f"   测试样本: {len(test_dataset)}")

# 查看一个样本
sample = train_dataset[0]
print(f"\n📝 样本示例:")
print(f"   Input IDs 形状: {sample['input_ids'].shape}")
print(f"   Attention Mask 形状: {sample['attention_mask'].shape}")
print(f"   Label: {sample['labels'].item()} (对应 {sample['labels'].item()+1} 星)")

# 解码回文本查看
sample_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
print(f"   解码文本: {sample_text[:30]}...")

# DataLoader
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

# ============================================================
# 第三部分：加载模型
# ============================================================
print("\n" + "=" * 70)
print("第三部分：加载 BERT 模型（5分类）")
print("=" * 70)

print("\n🤖 加载 bert-base-chinese...")
# 加载预训练 BERT，num_labels=5 表示5分类（1-5星）
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=5,           # 1-5星评价
    output_attentions=False,
    output_hidden_states=False
)

print(f"   模型加载完成!")
print(f"   模型类型: {type(model).__name__}")
print(f"   分类类别: {model.num_labels} (1-5星)")

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

print(f"\n⚙️ 训练配置:")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   优化器: AdamW")
print(f"   调度器: Linear warmup & decay")
print(f"   总训练步数: {total_steps}")

# 训练历史记录
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# ============================================================
# 第五部分：训练模型
# ============================================================
print("\n" + "=" * 70)
print("第五部分：训练模型")
print("=" * 70)

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
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 转换回1-5星范围计算指标
all_labels_star = [label + 1 for label in all_labels]    # 0-4 → 1-5
all_preds_star = [pred + 1 for pred in all_preds]        # 0-4 → 1-5

accuracy = accuracy_score(all_labels, all_preds)
mae = mean_absolute_error(all_labels_star, all_preds_star)  # 在1-5星范围计算
mse = mean_squared_error(all_labels_star, all_preds_star)

print(f"\n📊 评估结果:")
print(f"   准确率 (Accuracy): {accuracy:.2%}")
print(f"   平均星级误差 (MAE): {mae:.2f} 星")
print(f"   均方星级误差 (MSE): {mse:.2f}")

print(f"\n📋 分类报告:")
# 标签映射：0-4 → 1-5星
id2label = {0: '1星-很差', 1: '2星-较差', 2: '3星-一般', 3: '4星-较好', 4: '5星-很好'}
target_names = [id2label[i] for i in range(5)]
print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

# 查看预测示例
print(f"\n🔍 预测示例:")
for i in range(min(5, len(test_texts))):
    text = test_texts[i]
    true_label_idx = test_labels[i]      # 0-4
    pred_label_idx = all_preds[i]        # 0-4
    true_star = true_label_idx + 1      # 1-5
    pred_star = pred_label_idx + 1        # 1-5
    correct = "✅" if true_label_idx == pred_label_idx else "❌"
    print(f"   {correct} 文本: {text[:30]}...")
    print(f"      真实: {true_star}星 (标签{true_label_idx}) | 预测: {pred_star}星 (标签{pred_label_idx})")
    print()

# ============================================================
# 第七部分：保存模型
# ============================================================
print("\n" + "=" * 70)
print("第七部分：保存模型")
print("=" * 70)

# 创建保存目录
save_dir = './saved_models/chinese_star_rating_bert'
os.makedirs(save_dir, exist_ok=True)

# 保存模型和 tokenizer
print(f"\n💾 保存模型到: {save_dir}")
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

# 保存训练配置
import json
config_info = {
    'model_name': 'bert-base-chinese',
    'num_labels': 5,
    'label_mapping': {'0': '1星-很差', '1': '2星-较差', '2': '3星-一般', '3': '4星-较好', '4': '5星-很好'},
    'note': '标签0-4对应1-5星评价',
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'max_length': 128,
    'final_accuracy': float(accuracy),
    'final_mae': float(mae)
}

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
    ("这个手机太棒了，拍照效果特别好，很喜欢！", 5),
    ("物流很快，包装完整，商品质量超出预期，好评！", 4),
    ("还可以吧，没有特别好也没有特别差，对得起价格。", 3),
    ("质量一般吧，用了一周就出现小毛病，不太满意。", 2),
    ("太差了，收到的商品破损，客服还不理人。", 1),
]

print(f"\n📝 新样本预测:")
for text, expected_star in test_samples:
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
        pred_idx = torch.argmax(logits).item()    # 0-4
        confidence = probs[0][pred_idx].item()
    
    pred_star = pred_idx + 1  # 0-4 → 1-5
    star_bar = "⭐" * pred_star + "☆" * (5 - pred_star)
    correct = "✅" if pred_star == expected_star else "❌"
    print(f"   {correct} 文本: {text}")
    print(f"      预测: {pred_star}星 {star_bar} (标签{pred_idx}, 置信度: {confidence:.2%})")
    print(f"      真实期望: {expected_star}星")
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
   - 创建了100条中文电商评论数据（1-5星各20条）
   - 数据覆盖多种场景：质量、物流、客服、性价比等
   - 使用 stratify 保持各星级比例一致

2. 多分类模型
   - BERT 输出层改为 5 分类（num_labels=5）
   - 使用 CrossEntropyLoss 自动处理多分类

3. 特殊指标
   - 除准确率外，还计算 MAE（平均绝对误差）
   - 对星级预测，误差越小越好（3星预测成2星比预测成1星更好）

4. 训练流程
   - 与二分类基本相同
   - 预测时使用 argmax 取概率最高的星级

📚 多分类 vs 二分类的关键区别：
   • num_labels=5 而非 2
   • 损失函数 CrossEntropyLoss 自动支持多分类
   • 评估时关注 MAE 而不仅是准确率
   • 混淆矩阵分析更有意义（看哪些星级容易混淆）

🚀 下一步建议：
   1. 使用真实数据集（如外卖评论、豆瓣电影评论）
   2. 尝试回归任务（直接预测 1.0-5.0 连续值）
   3. 增加模型复杂度（如考虑评论长度、关键词等特征）
   4. 使用更小的学习率和更多 epoch
""")

print("\n" + "=" * 70)
print("中文星级评价分类实战完成！🎉")
print("=" * 70)