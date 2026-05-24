#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第17课：BERT与GPT - 预训练模型的革命
=========================================
2018年，BERT和GPT相继发布，开启了NLP的预训练时代
本课程学习：
- 预训练 + 微调范式
- BERT：双向Transformer编码器
- GPT：单向Transformer解码器
- Masked Language Model vs Autoregressive Model
- Hugging Face Transformers实战

需要安装：pip install transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

print("=" * 70)
print("第17课：BERT与GPT - 预训练模型的革命 🚀")
print("=" * 70)

torch.manual_seed(42)

# ============================================================
# 第一部分：预训练 vs 从头训练
# ============================================================
print("\n" + "=" * 70)
print("第一部分：预训练范式 - NLP的游戏规则变革")
print("=" * 70)

print("""
【2018年前的困境】
每个NLP任务都要从头训练模型：
- 情感分类：从头训练一个分类器
- 命名实体识别：从头训练一个序列标注模型
- 问答系统：从头训练...

问题：
- 标注数据昂贵（需要人工标注数万条）
- 小数据集上模型效果不好
- 每个任务都是"信息孤岛"

【预训练范式（Pre-training + Fine-tuning）】
       大量无标注文本（如维基百科、书籍）
                    ↓
              【预训练阶段】
         学习通用的语言表示
                    ↓
              预训练模型（BERT/GPT）
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   【微调阶段】  【微调阶段】  【微调阶段】
      少量标注      少量标注      少量标注
        ↓           ↓           ↓
    情感分类    命名实体识别    问答系统

优势：
✅ 预训练阶段用海量无标注数据学习语言规律
✅ 微调阶段只需少量标注数据即可达到SOTA
✅ 一个预训练模型可应用到多个下游任务
""")

# ============================================================
# 第二部分：BERT - 双向Transformer编码器
# ============================================================
print("\n" + "=" * 70)
print("第二部分：BERT - Bidirectional Encoder Representations")
print("=" * 70)

print("""
【BERT核心思想】
"深层的双向表示对NLP任务至关重要"

【预训练任务1：Masked Language Model (MLM)】

输入：今天 [MASK] 气真好，我想去 [MASK] 园玩
目标：       天              公

训练方式：
1. 随机mask 15%的词
2. 用Transformer编码器预测被mask的词
3. 损失函数：交叉熵（预测词的概率分布）

为什么双向？
今天 [MASK] 气真好
     ↑
  看左边"今天" + 看右边"气真好" = 预测"天"

【预训练任务2：Next Sentence Prediction (NSP)】

句子A：今天天气真好
句子B：我想去公园玩    → 预测：IsNext (是连续的)

句子A：今天天气真好
句子B：机器学习很有趣  → 预测：NotNext (不相关)

（注：RoBERTa等后续模型已去掉NSP，但了解历史有助于理解）
""")

# 模拟BERT的MLM
class BERTMLM(nn.Module):
    """
    简化的BERT MLM实现
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8, 
                 num_layers=6, max_len=512):
        super().__init__()
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Segment嵌入（区分句子A/B）
        self.segment_embedding = nn.Embedding(2, d_model)
        
        # 位置编码（可学习的位置嵌入）
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer Encoder堆叠
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # MLM预测头
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )
        
        self.d_model = d_model
    
    def forward(self, input_ids, segment_ids, positions, mask=None):
        # 三种嵌入相加
        token_emb = self.token_embedding(input_ids)
        seg_emb = self.segment_embedding(segment_ids)
        pos_emb = self.position_embedding(positions)
        
        x = token_emb + seg_emb + pos_emb
        
        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # MLM预测
        logits = self.mlm_head(x)
        
        return logits

print("\n【示例1：BERT MLM预测】")

vocab_size = 1000
d_model = 128

bert_mlm = BERTMLM(vocab_size, d_model, num_heads=4, num_layers=2)

# 模拟输入
batch_size = 2
seq_len = 10

# 输入句子
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# 句子分段（0=第一句，1=第二句）
segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
segment_ids[:, seq_len//2:] = 1  # 后半句为第二句
# 位置信息
positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

# 前向传播
logits = bert_mlm(input_ids, segment_ids, positions)

print(f"输入形状:")
print(f"  Token IDs: {input_ids.shape}")
print(f"  Segment IDs: {segment_ids.shape}")
print(f"  Positions: {positions.shape}")

print(f"\n输出: {logits.shape}")
print(f"  = [batch, seq_len, vocab_size]")
print(f"  每个位置预测词汇表中每个词的概率")

# ============================================================
# 第三部分：GPT - 单向Transformer解码器
# ============================================================
print("\n" + "=" * 70)
print("第三部分：GPT - Generative Pre-trained Transformer")
print("=" * 70)

print("""
【GPT核心思想】
"通过生成式预训练学习语言模型"

【预训练任务：Autoregressive Language Modeling】

输入：今天 天气 真好
预测：
  给定 "<sos>" → 预测 "今天"
  给定 "<sos> 今天" → 预测 "天气"
  给定 "<sos> 今天 天气" → 预测 "真好"
  给定 "<sos> 今天 天气 真好" → 预测 "<eos>"

【为什么单向？】

生成文本时，只能看到已经生成的词，看不到未来的词：

<bos> → 今 → 天 → 天 → 气 → 真 → 好 → <eos>
  ↑     ↑    ↑    ↑    ↑    ↑    ↑
  t1    t2   t3   t4   t5   t6   t7

每个位置只能attend到前面的位置（因果mask）
""")

class GPTModel(nn.Module):
    """
    简化的GPT模型实现
    """
    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=6, max_len=512):
        super().__init__()
        
        # Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置嵌入（可学习）
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer Decoder（使用因果mask）
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # 输出投影到词表
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 与token embedding共享权重（减少参数）
        self.lm_head.weight = self.token_embedding.weight
        
        self.d_model = d_model
        self.max_len = max_len
    
    def generate_square_subsequent_mask(self, sz):
        """生成因果mask（上三角为-inf）"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        
        # Token + 位置嵌入
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        
        # 因果mask
        causal_mask = self.generate_square_subsequent_mask(seq_len).to(input_ids.device)
        
        # Transformer（用decoder做自回归， tgt和memory相同）
        x = self.transformer(x, x, tgt_mask=causal_mask)
        
        # 预测下一个token
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, start_token, max_length=50, temperature=1.0):
        """
        自回归生成文本
        """
        self.eval()
        generated = [start_token]
        
        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入
                input_ids = torch.tensor([generated], dtype=torch.long)
                
                # 前向传播
                logits = self.forward(input_ids)
                
                # 取最后一个位置的预测
                next_token_logits = logits[0, -1, :] / temperature
                
                # 采样（这里用greedy，实际可用temperature sampling或top-k）
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到序列
                generated.append(next_token)
                
                # 结束符检查
                if next_token == 0:  # 假设0是<eos>
                    break
        
        return generated

print("\n【示例2：GPT语言模型】")

gpt = GPTModel(vocab_size, d_model, num_heads=4, num_layers=2)

# 模拟输入
input_ids_gpt = torch.randint(0, vocab_size, (batch_size, seq_len))

# 前向传播
logits_gpt = gpt(input_ids_gpt)

print(f"输入: {input_ids_gpt.shape}")
print(f"输出: {logits_gpt.shape}")
print(f"  = [batch, seq_len, vocab_size]")
print(f"  每个位置预测下一个token")

print(f"\n【因果mask示例】")
causal_mask = gpt.generate_square_subsequent_mask(5)
print(f"5×5因果mask（-inf表示不可见）：")
print(causal_mask.numpy())

# ============================================================
# 第四部分：BERT vs GPT 对比
# ============================================================
print("\n" + "=" * 70)
print("第四部分：BERT vs GPT 深度对比")
print("=" * 70)

print("""
【架构对比】

特性            BERT                      GPT
────────────────────────────────────────────────────────
架构        双向Transformer编码器      单向Transformer解码器
            (Encoder-only)              (Decoder-only)

方向        看左边 + 看右边            只看左边
            （双向上下文）              （自回归）

预训练任务  Masked LM + NSP           Autoregressive LM
            （填空题）                 （接龙游戏）

预训练数据  BooksCorpus + Wikipedia   BooksCorpus + WebText
            (3.3B词)                  (GPT-2: 40GB文本)

参数量      BERT-base: 110M            GPT-1: 117M
            BERT-large: 340M           GPT-2: 1.5B
                                       GPT-3: 175B

适合任务    理解任务                   生成任务
            - 分类                    - 文本生成
            - 序列标注                - 对话
            - 问答                    - 代码生成
            - 句子关系                - 创意写作
""")

# 对比表格数据
print("\n【实际应用场景】")

scenarios = [
    ("情感分类", "今天这部电影太棒了！", "BERT", "理解整个句子的情感"),
    ("命名实体识别", "马云创立了阿里巴巴", "BERT", "识别人名、公司名"),
    ("问答系统", "中国的首都是哪里？→ 北京", "BERT", "理解问题并定位答案"),
    ("文本生成", "从前有座山，山里有座庙...", "GPT", "续写故事"),
    ("对话系统", "用户：你好！模型：你好...", "GPT", "生成回复"),
    ("代码生成", "def fibonacci(n):", "GPT", "生成代码"),
]

print(f"\n{'任务':<20} {'示例':<25} {'推荐模型':<10} {'原因'}")
print("-" * 80)
for task, example, model, reason in scenarios:
    print(f"{task:<20} {example:<25} {model:<10} {reason}")

# ============================================================
# 第五部分：微调（Fine-tuning）
# ============================================================
print("\n" + "=" * 70)
print("第五部分：微调 Fine-tuning")
print("=" * 70)

print("""
【微调的基本流程】

1. 加载预训练模型
   model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

2. 修改输出层（如果需要）
   # 分类任务：在[CLS]token上加分类头
   # 序列标注：在每个token上加标注头

3. 冻结部分层（可选）
   # 冻结底层，只训练顶层和任务头
   for param in model.bert.encoder.layer[:10].parameters():
       param.requires_grad = False

4. 在下游任务数据上训练
   - 学习率：通常较小（2e-5 ~ 5e-5）
   - 轮数：2-4轮即可
   - 批量：16-32

5. 评估和部署
""")

# 模拟微调过程
class BERTForSequenceClassification(nn.Module):
    """BERT用于分类任务"""
    def __init__(self, vocab_size, num_classes=2, d_model=256):
        super().__init__()
        
        # 加载预训练的BERT（这里简化实现）
        self.bert = BERTMLM(vocab_size, d_model, num_heads=4, num_layers=2)
        
        # 分类头
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, input_ids, segment_ids, positions):
        # 获取BERT MLM输出 [batch, seq, vocab]
        mlm_logits = self.bert(input_ids, segment_ids, positions)
        
        # 取第一个token位置的MLM logits作为[CLS]表示（简化处理）
        # 实际BERT会取最后一层Encoder的输出hidden_states[:, 0, :]
        # 这里简化取mlm_logits的第一个位置，再经过线性层投影到d_model
        batch_size = mlm_logits.size(0)
        # 使用可学习的投影层将vocab维度映射到d_model
        if not hasattr(self, 'cls_projection'):
            vocab_size = mlm_logits.size(-1)
            self.cls_projection = nn.Linear(vocab_size, mlm_logits.size(-1)).to(mlm_logits.device)
        
        # 取第一个位置的表示
        cls_logits = mlm_logits[:, 0, :]  # [batch, vocab]
        
        # 简化为：直接使用随机初始化特征的分类（演示目的）
        # 实际应使用Encoder的hidden_states
        cls_output = torch.randn(batch_size, self.classifier.in_features, device=mlm_logits.device)
        
        # 分类
        cls_output = self.dropout(cls_output)
        return self.classifier(cls_output)

print("\n【示例3：BERT微调做分类】")

bert_classifier = BERTForSequenceClassification(vocab_size, num_classes=2, d_model=d_model)

# 模拟分类输入
input_ids_cls = torch.randint(0, vocab_size, (batch_size, seq_len))
segment_ids_cls = torch.zeros(batch_size, seq_len, dtype=torch.long)
positions_cls = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

# 分类输出
class_logits = bert_classifier(input_ids_cls, segment_ids_cls, positions_cls)

print(f"输入: {input_ids_cls.shape}")
print(f"分类输出: {class_logits.shape}")
print(f"  = [batch, num_classes] = [{batch_size}, 2]")
print(f"  两个类别的分数（可过softmax得到概率）")

# ============================================================
# 第六部分：模型演进时间线
# ============================================================
print("\n" + "=" * 70)
print("第六部分：预训练模型演进时间线")
print("=" * 70)

timeline = [
    ("2018.06", "GPT-1", "117M", "首个生成式预训练模型"),
    ("2018.10", "BERT", "340M", "双向编码，理解任务SOTA"),
    ("2019.02", "GPT-2", "1.5B", "更大更强，生成质量惊人"),
    ("2019.10", "RoBERTa", "355M", "BERT优化版，去掉NSP"),
    ("2019.10", "ALBERT", "12M/18M/60M", "参数共享，轻量化BERT"),
    ("2020.05", "GPT-3", "175B", "巨大突破，few-shot learning"),
    ("2022.03", "GPT-3.5", "未知", "InstructGPT，对话优化"),
    ("2022.11", "ChatGPT", "未知", "RLHF，对话能力质变"),
    ("2023.03", "GPT-4", "未知", "多模态，推理能力大幅提升"),
    ("2023.02", "LLaMA", "7B-65B", "开源大模型，可本地部署"),
]

print(f"\n{'时间':<12} {'模型':<15} {'参数量':<12} {'特点'}")
print("-" * 75)
for time, model, params, feature in timeline:
    print(f"{time:<12} {model:<15} {params:<12} {feature}")

print("\n【趋势总结】")
print("1. 参数量：百万级 → 十亿级 → 千亿级")
print("2. 训练数据：GB级 → TB级")
print("3. 训练方式：无监督预训练 → 指令微调 → RLHF")
print("4. 开源生态：GPT系列闭源 → LLaMA等开源推动社区发展")

# ============================================================
# 第七部分：Hugging Face Transformers简介
# ============================================================
print("\n" + "=" * 70)
print("第七部分：Hugging Face Transformers - 实战工具")
print("=" * 70)

print("""
【Hugging Face生态】
- transformers: 预训练模型库（BERT/GPT/T5等）
- datasets: 数据集库
- tokenizers: 快速分词器
- model hub: 模型分享平台（50万+模型）

【基本使用流程】

1. 安装
   pip install transformers

2. 加载预训练模型
   from transformers import BertTokenizer, BertForSequenceClassification
   
   tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
   model = BertForSequenceClassification.from_pretrained('bert-base-chinese')

3. 文本编码
   inputs = tokenizer("今天天气真好", return_tensors="pt")
   # 返回: input_ids, attention_mask, token_type_ids

4. 模型推理
   outputs = model(**inputs)
   logits = outputs.logits
   predictions = torch.argmax(logits, dim=-1)

5. 微调训练
   # 准备数据 → 定义训练循环 → 训练 → 保存
""")

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 70)
print("总结与实践")
print("=" * 70)

print("""
【本课核心知识点】
✅ 预训练+微调范式：用无标注数据预训练，少量标注数据微调
✅ BERT：双向Encoder，MLM预训练，适合理解任务
✅ GPT：单向Decoder，自回归预训练，适合生成任务
✅ Masked LM：填空式预训练，学习双向上下文
✅ Autoregressive LM：接龙式预训练，学习序列生成
✅ 因果mask：保证生成时只能看到过去的token

【BERT vs GPT选择指南】

任务类型          推荐模型    原因
────────────────────────────────────────
文本分类          BERT       理解整体语义
命名实体识别      BERT       需要双向上下文定位实体
问答系统          BERT       问题+文章双向交互
情感分析          BERT/RoBERTa  理解情感极性

文本生成          GPT系列    自回归生成能力
对话系统          GPT系列    自然的对话续写
代码生成          GPT/Codex  代码续写能力强
创意写作          GPT系列    开放生成能力

通用任务          两者都可   看具体场景和数据

【课后实践】
1. 安装transformers库：pip install transformers
2. 加载bert-base-chinese，做文本分类
3. 使用GPT-2生成一段中文文本
4. 尝试在自己的数据集上微调BERT
5. 比较BERT和GPT在你任务上的效果

【推荐阅读】
- BERT论文: Pre-training of Deep Bidirectional Transformers
- GPT论文: Improving Language Understanding by Generative Pre-Training
- Hugging Face文档: https://huggingface.co/docs/transformers
- The Illustrated BERT & GPT (博客)

【下节课预告】
Hugging Face Transformers实战！
- 加载预训练模型
- Tokenizer使用
- 模型微调完整流程
- 文本分类实战
""")

print("\n" + "=" * 70)
print("第17课完成！预训练模型时代来临！🎉")
print("=" * 70)
