#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第12课：神经网络构建与训练
=====================
本课程学习PyTorch中构建和训练神经网络的核心概念：
- nn.Module - 神经网络的基类
- 构建多层感知机（MLP）
- 激活函数
- 损失函数
- 优化器
- 完整的训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("=" * 60)
print("第12课：神经网络构建与训练")
print("=" * 60)

# 设置随机种子保证可复现
torch.manual_seed(42)

# ============================================================
# 第一部分：nn.Module基础
# ============================================================
print("\n" + "=" * 60)
print("第一部分：nn.Module - 神经网络的乐高积木")
print("=" * 60)

print("""
【核心概念】
nn.Module是所有神经网络的基类，它提供了：
1. 参数管理 - 自动追踪所有可学习的参数
2. GPU加速 - 一键将模型和数据移到GPU
3. 保存加载 - 方便地保存和恢复模型
4. 模块化设计 - 像搭积木一样构建复杂网络
""")

# 定义一个简单的线性层
print("\n【示例1：单个线性层】")
linear = nn.Linear(in_features=10, out_features=5)
print(f"线性层: {linear}")
print(f"输入维度: 10 → 输出维度: 5")

# 查看参数
print(f"\n权重矩阵形状: {linear.weight.shape}")  # [5, 10]
print(f"偏置向量形状: {linear.bias.shape}")      # [5]

# 前向传播
x = torch.randn(3, 10)  # 3个样本，每个10维
output = linear(x)
print(f"\n输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"线性变换: y = x @ W^T + b")

# ============================================================
# 第二部分：激活函数
# ============================================================
print("\n" + "=" * 60)
print("第二部分：激活函数 - 给神经网络引入非线性")
print("=" * 60)

print("""
【为什么需要激活函数？】
如果没有激活函数，多层神经网络就等价于单层线性变换：
Linear(Linear(x)) = W2 @ (W1 @ x + b1) + b2 = (W2 @ W1) @ x + (W2 @ b1 + b2)

激活函数引入非线性，让网络可以学习复杂的模式。
""")

# 常用激活函数
activations = {
    'ReLU': nn.ReLU(),           # 最常用，计算简单，缓解梯度消失
    'Sigmoid': nn.Sigmoid(),     # 输出0-1，适合二分类
    'Tanh': nn.Tanh(),           # 输出-1到1，以0为中心
    'LeakyReLU': nn.LeakyReLU(), # ReLU的改进版，解决"死亡ReLU"问题
    'GELU': nn.GELU(),           # Transformer中常用，更平滑
}

x = torch.linspace(-3, 3, 100).unsqueeze(1)  # 输入值

print("\n【常用激活函数对比】")
for name, activation in activations.items():
    y = activation(x)
    print(f"{name:12}: 输出范围 [{y.min():.2f}, {y.max():.2f}]")

print("""
【使用建议】
- 隐藏层: ReLU（默认选择）、GELU（Transformer）
- 输出层: 
  * 二分类: Sigmoid
  * 多分类: Softmax
  * 回归问题: 无激活函数（线性输出）
""")

# ============================================================
# 第三部分：构建多层感知机（MLP）
# ============================================================
print("\n" + "=" * 60)
print("第三部分：构建多层感知机（MLP）")
print("=" * 60)

print("""
【MLP结构】
输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
  ↓         ↓          ↓            ↓
特征    非线性变换   非线性变换     预测结果
""")

# 定义一个MLP类
class SimpleMLP(nn.Module):
    """
    简单多层感知机
    结构: Input(784) → Hidden1(256) → Hidden2(128) → Output(10)
    """
    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        super(SimpleMLP, self).__init__()
        
        # 定义网络层
        self.layer1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  # 防止过拟合
        
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.layer3 = nn.Linear(hidden2, num_classes)
        # 输出层不加激活函数，后面用CrossEntropyLoss会自动加Softmax
    
    def forward(self, x):
        # 前向传播
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)
        return x

# 创建模型实例
model = SimpleMLP(input_size=784, hidden1=256, hidden2=128, num_classes=10)
print("\n【模型结构】")
print(model)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 测试前向传播
batch_size = 4
x_test = torch.randn(batch_size, 784)
output = model(x_test)
print(f"\n输入形状: {x_test.shape}")
print(f"输出形状: {output.shape}")
print(f"输出样例（第一个样本的10个类别分数）:")
print(f"  {output[0].detach().numpy().round(3)}")

# ============================================================
# 第四部分：损失函数
# ============================================================
print("\n" + "=" * 60)
print("第四部分：损失函数 - 衡量预测与真实的差距")
print("=" * 60)

print("""
【常见损失函数】
1. 回归问题：
   - MSELoss（均方误差）: 预测连续值
   - L1Loss（平均绝对误差）: 对异常值更鲁棒

2. 分类问题：
   - CrossEntropyLoss（交叉熵）: 多分类标准选择
   - BCELoss（二元交叉熵）: 二分类
""")

# 回归损失示例
print("\n【回归损失示例】")
predictions = torch.tensor([2.5, 0.0, 2.1, 1.6])
targets = torch.tensor([3.0, -0.5, 2.0, 1.0])

mse_loss = nn.MSELoss()
loss = mse_loss(predictions, targets)
print(f"预测值: {predictions.numpy()}")
print(f"真实值: {targets.numpy()}")
print(f"MSE损失: {loss.item():.4f}")
print(f"计算: mean((3-2.5)² + (-0.5-0)² + (2-2.1)² + (1-1.6)²) = {loss.item():.4f}")

# 分类损失示例
print("\n【分类损失示例 - CrossEntropyLoss】")
# 模拟3个样本，4个类别的原始分数（logits）
logits = torch.tensor([
    [2.0, 1.0, 0.1, 0.5],   # 样本1，模型认为类别0最可能
    [0.5, 2.5, 0.3, 0.2],   # 样本2，模型认为类别1最可能
    [0.1, 0.2, 3.0, 0.5]    # 样本3，模型认为类别2最可能
])
# 真实标签
labels = torch.tensor([0, 1, 2])  # 样本1属于类别0，样本2属于类别1，样本3属于类别2

ce_loss = nn.CrossEntropyLoss()
loss = ce_loss(logits, labels)

# 手动验证（Softmax + 负对数似然）
probs = torch.softmax(logits, dim=1)
print(f"\n原始分数 (logits):")
for i, logit in enumerate(logits):
    print(f"  样本{i+1}: {logit.numpy()}")

print(f"\nSoftmax概率:")
for i, prob in enumerate(probs):
    print(f"  样本{i+1}: {prob.detach().numpy().round(4)} (和={prob.sum():.4f})")

print(f"\n真实标签: {labels.numpy()}")
print(f"每个样本取真实类别的概率:")
for i in range(len(labels)):
    print(f"  样本{i+1}类别{labels[i]}的概率: {probs[i, labels[i]].item():.4f}")

print(f"\nCrossEntropyLoss = -mean(log(真实类别概率)) = {loss.item():.4f}")

# ============================================================
# 第五部分：优化器
# ============================================================
print("\n" + "=" * 60)
print("第五部分：优化器 - 参数更新的策略")
print("=" * 60)

print("""
【优化器的作用】
根据梯度来更新模型参数，不同的优化器有不同的更新策略。

【常用优化器】
1. SGD（随机梯度下降）: 最基础，需要调学习率
2. SGD + Momentum: 加入动量，加速收敛
3. Adam（Adaptive Moment Estimation）: 自适应学习率，最常用
4. AdamW: Adam + 权重衰减，Transformer训练首选
""")

# 创建一个简单模型演示优化器
simple_model = nn.Linear(10, 1)

# 不同优化器
optimizers = {
    'SGD': optim.SGD(simple_model.parameters(), lr=0.01),
    'SGD+Momentum': optim.SGD(simple_model.parameters(), lr=0.01, momentum=0.9),
    'Adam': optim.Adam(simple_model.parameters(), lr=0.001),
    'AdamW': optim.AdamW(simple_model.parameters(), lr=0.001, weight_decay=0.01),
}

print("\n【优化器配置示例】")
for name, opt in optimizers.items():
    print(f"{name:15}: {opt}")

print("""
【学习率建议】
- SGD: 0.01 ~ 0.1
- Adam/AdamW: 0.0001 ~ 0.001 (常用: 1e-4, 3e-4, 1e-3)
- 学习率衰减: 训练后期逐渐减小学习率
""")

# ============================================================
# 第六部分：完整训练流程 - 手写数字识别
# ============================================================
print("\n" + "=" * 60)
print("第六部分：完整训练流程 - 手写数字识别（MNIST简化版）")
print("=" * 60)

# 模拟MNIST数据（实际使用时应加载真实数据）
def generate_mnist_like_data(n_samples=1000):
    """生成模拟的MNIST风格数据"""
    # 输入: 28x28 = 784维
    X = torch.randn(n_samples, 784)
    # 10个类别（0-9数字）
    y = torch.randint(0, 10, (n_samples,))
    return X, y

# 生成数据
X_train, y_train = generate_mnist_like_data(1000)
X_test, y_test = generate_mnist_like_data(200)

print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")

# 创建模型、损失函数、优化器
model = SimpleMLP(input_size=784, hidden1=256, hidden2=128, num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"\n模型配置:")
print(f"  优化器: Adam")
print(f"  学习率: 0.001")
print(f"  损失函数: CrossEntropyLoss")

# 训练循环
print("\n【开始训练】")
n_epochs = 10
batch_size = 32
n_batches = len(X_train) // batch_size

for epoch in range(n_epochs):
    model.train()  # 训练模式（启用Dropout）
    total_loss = 0
    correct = 0
    
    # 随机打乱数据
    indices = torch.randperm(len(X_train))
    
    for i in range(n_batches):
        # 获取一个batch
        batch_indices = indices[i*batch_size:(i+1)*batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
    
    # 计算epoch级别的统计
    avg_loss = total_loss / n_batches
    accuracy = 100 * correct / len(X_train)
    
    if (epoch + 1) % 2 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{n_epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")

# 测试
model.eval()  # 评估模式（禁用Dropout）
with torch.no_grad():  # 不计算梯度，节省内存
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs, 1)
    test_accuracy = 100 * (predicted == y_test).sum().item() / len(y_test)

print(f"\n测试集准确率: {test_accuracy:.2f}%")

# ============================================================
# 第七部分：GPU加速训练（MPS/CUDA）
# ============================================================
print("\n" + "=" * 60)
print("第七部分：GPU加速训练（适配M4芯片MPS）")
print("=" * 60)

# 检测可用设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用Apple Silicon MPS加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ 使用NVIDIA CUDA加速")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU训练")

print(f"设备: {device}")

# 创建模型并移到GPU
model_gpu = SimpleMLP(input_size=784, hidden1=512, hidden2=256, num_classes=10)
model_gpu = model_gpu.to(device)

# 生成更大规模数据测试GPU加速
X_large = torch.randn(10000, 784).to(device)
y_large = torch.randint(0, 10, (10000,)).to(device)

print(f"\n数据已移动到{device}")
print(f"输入张量设备: {X_large.device}")
print(f"模型参数设备: {next(model_gpu.parameters()).device}")

# 简单benchmark
import time

# Warm up（MPS需要预热）
for _ in range(10):
    _ = model_gpu(X_large[:100])

if device.type == 'mps':
    torch.mps.synchronize()

# 正式计时
start = time.time()
n_iterations = 50

for _ in range(n_iterations):
    outputs = model_gpu(X_large)
    loss = criterion(outputs, y_large)
    
    optimizer_gpu = optim.Adam(model_gpu.parameters())
    optimizer_gpu.zero_grad()
    loss.backward()
    optimizer_gpu.step()

if device.type == 'mps':
    torch.mps.synchronize()

elapsed = time.time() - start
print(f"\nGPU训练速度:")
print(f"  {n_iterations}轮前向+反向传播: {elapsed:.3f}秒")
print(f"  每轮平均: {elapsed/n_iterations*1000:.1f}ms")

# ============================================================
# 第八部分：模型保存与加载
# ============================================================
print("\n" + "=" * 60)
print("第八部分：模型保存与加载")
print("=" * 60)

# 保存模型
model_path = "/tmp/mnist_model.pth"  # 使用临时目录

try:
    # 保存完整的模型状态
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': n_epochs,
        'loss': avg_loss,
    }, model_path)
    print(f"✅ 模型已保存到: {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path)
    
    # 创建新模型实例并加载参数
    new_model = SimpleMLP(input_size=784, hidden1=256, hidden2=128, num_classes=10)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ 模型已从epoch {checkpoint['epoch']}加载")
    print(f"   训练损失: {checkpoint['loss']:.4f}")
except Exception as e:
    print(f"⚠️ 模型保存/加载演示（临时目录可能不可用）: {e}")

print("""
【保存加载最佳实践】
1. 只保存state_dict（推荐）：
   torch.save(model.state_dict(), "model.pth")
   model.load_state_dict(torch.load("model.pth"))

2. 保存完整checkpoint（包含优化器状态）：
   torch.save({
       'epoch': epoch,
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'loss': loss,
   }, "checkpoint.pth")
""")

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 60)
print("总结与实践")
print("=" * 60)

print("""
【本课核心知识点】
✅ nn.Module - 所有网络的基类
✅ 激活函数 - ReLU、GELU、Sigmoid等
✅ 损失函数 - MSE（回归）、CrossEntropy（分类）
✅ 优化器 - Adam最常用，AdamW用于Transformer
✅ 训练流程 - 前向 → 计算损失 → 反向 → 更新
✅ GPU加速 - model.to(device), data.to(device)
✅ 模型保存 - state_dict方式最推荐

【神经网络构建模板】
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(输入, 隐藏层)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(隐藏层, 输出)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# 训练模板
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
```
""")

# 实践练习
print("\n【课后练习】")
print("1. 修改SimpleMLP的隐藏层大小，观察参数量和训练效果变化")
print("2. 尝试不同的优化器（SGD vs Adam），比较收敛速度")
print("3. 增加训练轮数，观察过拟合现象")
print("4. 加载真实MNIST数据（torchvision.datasets.MNIST）进行完整训练")
print("\n下节课预告：卷积神经网络（CNN）- 处理图像的强大武器")

print("\n" + "=" * 60)
print("第12课完成！🎉")
print("=" * 60)
