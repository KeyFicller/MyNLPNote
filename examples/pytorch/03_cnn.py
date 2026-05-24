#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第13课：卷积神经网络（CNN）
============================
本课程学习处理图像的强大神经网络架构：
- 为什么CNN更适合图像？
- 卷积层的工作原理（卷积核、步长、填充）
- 池化层
- 批归一化
- 经典架构：LeNet、ResNet
- 图像分类实战
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

print("=" * 70)
print("第13课：卷积神经网络（CNN）")
print("=" * 70)

# 设置随机种子
torch.manual_seed(42)

# ============================================================
# 第一部分：为什么需要CNN？
# ============================================================
print("\n" + "=" * 70)
print("第一部分：为什么CNN更适合图像？")
print("=" * 70)

print("""
【全连接层的困境】
如果用MLP处理图像：
- 一张224×224的彩色图像 = 224 × 224 × 3 = 150,528 个输入特征
- 第一层隐藏层256个神经元 → 150,528 × 256 ≈ 3800万参数！
- 参数太多 → 过拟合、计算量巨大、训练困难

【CNN的优势】
1. 局部连接：卷积核只关注局部区域（如3×3）
2. 权值共享：同一个卷积核在整个图像上滑动
3. 平移不变性：猫在图片左上角和右下角，CNN都能识别
4. 层次特征：
   - 浅层：边缘、纹理
   - 中层：形状、部件
   - 深层：物体、语义
""")

# ============================================================
# 第二部分：卷积层详解
# ============================================================
print("\n" + "=" * 70)
print("第二部分：卷积层 - 特征提取的核心")
print("=" * 70)

print("""
【卷积运算的本质】
卷积核（Filter/Kernel）在图像上滑动，计算局部区域的点积
就像一个"特征检测器"，检测边缘、纹理等模式
""")

# 创建一个卷积层示例
print("\n【示例1：基础卷积层】")
conv1 = nn.Conv2d(
    in_channels=3,      # 输入通道（RGB图像）
    out_channels=16,    # 输出通道（16个卷积核）
    kernel_size=3,      # 卷积核大小 3×3
    stride=1,           # 步长：每次移动1像素
    padding=1           # 填充：保持输出尺寸
)

print(f"卷积层: {conv1}")
print(f"输入通道: 3 (RGB)")
print(f"输出通道: 16 (16个卷积核)")
print(f"卷积核大小: 3×3")
print(f"参数量: 16 × 3 × 3 × 3 + 16 = {sum(p.numel() for p in conv1.parameters()):,}")

# 测试卷积层
batch_size = 2
input_tensor = torch.randn(batch_size, 3, 32, 32)  # [N, C, H, W]
output = conv1(input_tensor)

print(f"\n输入形状: {input_tensor.shape}")
print(f"  - 批量大小(N): {input_tensor.shape[0]}")
print(f"  - 通道数(C): {input_tensor.shape[1]}")
print(f"  - 高度(H): {input_tensor.shape[2]}")
print(f"  - 宽度(W): {input_tensor.shape[3]}")

print(f"\n输出形状: {output.shape}")
print(f"  - 输出尺寸 = (输入尺寸 - 卷积核 + 2×填充) / 步长 + 1")
print(f"  - (32 - 3 + 2×1) / 1 + 1 = 32 ✓")

# 不同参数对比
print("\n【示例2：不同卷积参数对比】")
conv_configs = [
    ("保持尺寸", {'kernel_size': 3, 'stride': 1, 'padding': 1}),
    ("下采样2x", {'kernel_size': 3, 'stride': 2, 'padding': 1}),
    ("下采样2x(更平滑)", {'kernel_size': 4, 'stride': 2, 'padding': 1}),
    ("感受野更大", {'kernel_size': 5, 'stride': 1, 'padding': 2}),
]

print(f"输入: [2, 3, 32, 32]")
for name, config in conv_configs:
    conv = nn.Conv2d(3, 16, **config)
    out = conv(input_tensor)
    print(f"  {name:15} kernel={config['kernel_size']}, stride={config['stride']}, padding={config['padding']} → 输出: {tuple(out.shape)}")

# ============================================================
# 第三部分：池化层
# ============================================================
print("\n" + "=" * 70)
print("第三部分：池化层 - 降维与特征聚合")
print("=" * 70)

print("""
【池化的作用】
1. 降低空间维度，减少计算量
2. 提供平移不变性
3. 聚合特征，提取主要信息
""")

# 创建测试数据
test_input = torch.randn(1, 16, 32, 32)

print("\n【池化方式对比】")
poolings = [
    ("MaxPool2d(2)", nn.MaxPool2d(2)),
    ("MaxPool2d(2,2)", nn.MaxPool2d(kernel_size=2, stride=2)),
    ("AvgPool2d(2)", nn.AvgPool2d(2)),
    ("AdaptiveAvgPool2d(1)", nn.AdaptiveAvgPool2d(1)),
]

for name, pool in poolings:
    out = pool(test_input)
    print(f"  {name:25} 输入: [1, 16, 32, 32] → 输出: {tuple(out.shape)}")

print("""
【池化选择建议】
- MaxPool：保留显著特征（边缘、纹理），更常用
- AvgPool：保留整体信息，用于全局特征聚合
- AdaptivePool：指定输出尺寸，自动计算池化参数
""")

# ============================================================
# 第四部分：批归一化（BatchNorm）
# ============================================================
print("\n" + "=" * 70)
print("第四部分：批归一化（BatchNorm）- 加速训练的利器")
print("=" * 70)

print("""
【BatchNorm的作用】
1. 解决内部协变量偏移问题
2. 允许使用更大的学习率，加速收敛
3. 有一定正则化效果，减少过拟合
4. 使网络对初始化不那么敏感

【原理】
对每个batch的每个通道，归一化为均值为0、方差为1：
  x̂ = (x - μ) / √(σ² + ε)
然后学习缩放和偏移：
  y = γ × x̂ + β
""")

# 创建BatchNorm层
print("\n【示例：BatchNorm2d】")
bn = nn.BatchNorm2d(16)
bn_input = torch.randn(4, 16, 8, 8)  # [batch, channels, H, W]
bn_output = bn(bn_input)

print(f"输入形状: {bn_input.shape}")
print(f"输出形状: {bn_output.shape}")
print(f"\n归一化效果验证（第一个通道）：")
print(f"  输入均值: {bn_input[:, 0].mean().item():.4f}")
print(f"  输出均值: {bn_output[:, 0].mean().item():.4f}（接近0）")
print(f"  输入标准差: {bn_input[:, 0].std().item():.4f}")
print(f"  输出标准差: {bn_output[:, 0].std().item():.4f}（接近1）")

# ============================================================
# 第五部分：构建完整的CNN
# ============================================================
print("\n" + "=" * 70)
print("第五部分：构建CNN - 从LeNet到现代架构")
print("=" * 70)

# LeNet-5风格网络
class LeNet(nn.Module):
    """
    LeNet-5风格的CNN
    适合CIFAR-10等小图像分类
    """
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 卷积层组1: 3×32×32 → 16×16×16
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 卷积层组2: 6×16×16 → 16×8×8
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # 卷积层1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 卷积层2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("\n【LeNet架构】")
lenet = LeNet(num_classes=10)
print(lenet)

# 计算参数量
total_params = sum(p.numel() for p in lenet.parameters())
print(f"\n总参数量: {total_params:,}")

# 测试前向传播
test_img = torch.randn(2, 3, 32, 32)
output = lenet(test_img)
print(f"\n输入: {test_img.shape} (2张32×32RGB图像)")
print(f"输出: {output.shape} (2张图像的10类分类分数)")

# ============================================================
# 第六部分：现代CNN技巧
# ============================================================
print("\n" + "=" * 70)
print("第六部分：现代CNN设计技巧")
print("=" * 70)

print("""
【现代CNN的核心技巧】

1. **残差连接（ResNet）**
   - 解决深层网络梯度消失问题
   - F(x) = H(x) + x，学习残差而非直接映射

2. **瓶颈结构（Bottleneck）**
   - 1×1卷积降维 → 3×3卷积 → 1×1卷积升维
   - 减少计算量，提高效率

3. **全局平均池化（GAP）**
   - 替换全连接层，减少参数量
   - 提高泛化能力
""")

# 残差块实现
class ResidualBlock(nn.Module):
    """残差块：核心创新，让网络可以很深"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 关键：残差连接
        out = F.relu(out)
        return out

print("\n【残差块结构】")
res_block = ResidualBlock(64)
print(res_block)

# 测试残差块
res_input = torch.randn(2, 64, 32, 32)
res_output = res_block(res_input)
print(f"\n输入: {res_input.shape}")
print(f"输出: {res_output.shape}")
print(f"尺寸保持: ✓ (残差连接保持空间维度)")

# ============================================================
# 第七部分：计算卷积层输出尺寸
# ============================================================
print("\n" + "=" * 70)
print("第七部分：卷积层输出尺寸计算")
print("=" * 70)

def calc_conv_output_size(input_size, kernel, stride=1, padding=0):
    """计算卷积后的输出尺寸"""
    return (input_size - kernel + 2 * padding) // stride + 1

def calc_pool_output_size(input_size, kernel, stride=None):
    """计算池化后的输出尺寸"""
    if stride is None:
        stride = kernel
    return input_size // stride

print("\n【尺寸计算示例】")
print("假设输入: 224×224 的 RGB 图像")

layers = [
    ("Conv 7×7, s=2, p=3", 224, lambda x: calc_conv_output_size(x, 7, 2, 3)),
    ("MaxPool 3×3, s=2", None, lambda x: calc_pool_output_size(x, 3, 2)),
    ("Conv 3×3, s=1, p=1", None, lambda x: calc_conv_output_size(x, 3, 1, 1)),
    ("Conv 3×3, s=1, p=1", None, lambda x: calc_conv_output_size(x, 3, 1, 1)),
    ("MaxPool 2×2, s=2", None, lambda x: calc_pool_output_size(x, 2, 2)),
]

size = 224
print(f"\n初始尺寸: {size}×{size}")
for name, input_s, calc_fn in layers:
    if input_s:
        size = input_s
    size = calc_fn(size)
    print(f"  {name:25} → {size}×{size}")

# ============================================================
# 第八部分：CNN训练实战
# ============================================================
print("\n" + "=" * 70)
print("第八部分：CNN训练实战（模拟CIFAR-10）")
print("=" * 70)

# 检测设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ 使用 Apple Silicon MPS 加速")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ 使用 NVIDIA CUDA 加速")
else:
    device = torch.device("cpu")
    print("⚠️ 使用 CPU 训练")

# 创建现代CNN
class ModernCNN(nn.Module):
    """
    现代CNN设计：
    - 使用BatchNorm
    - 使用残差连接
    - 使用GAP替代全连接
    """
    def __init__(self, num_classes=10):
        super(ModernCNN, self).__init__()
        
        # 第一阶段
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 第二阶段
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 第三阶段
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 全局平均池化 + 分类
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 阶段1: 32×32 → 16×16
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # 阶段2: 16×16 → 8×8
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # 阶段3: 8×8 → 4×4
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        # GAP: 4×4 → 1×1
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 创建模型
model = ModernCNN(num_classes=10).to(device)
print(f"\n【ModernCNN架构】")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# 生成模拟数据
def generate_cifar_like_data(n_samples=1000):
    """模拟CIFAR-10数据"""
    X = torch.randn(n_samples, 3, 32, 32)
    y = torch.randint(0, 10, (n_samples,))
    return X, y

X_train, y_train = generate_cifar_like_data(500)
X_test, y_test = generate_cifar_like_data(100)

print(f"\n训练集: {X_train.shape}")
print(f"测试集: {X_test.shape}")

# 配置训练
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
print("\n【开始训练】")
n_epochs = 5
batch_size = 32

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    correct = 0
    n_batches = len(X_train) // batch_size
    
    indices = torch.randperm(len(X_train))
    
    for i in range(n_batches):
        batch_idx = indices[i*batch_size:(i+1)*batch_size]
        batch_x = X_train[batch_idx].to(device)
        batch_y = y_train[batch_idx].to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
    
    train_acc = 100 * correct / (n_batches * batch_size)
    print(f"  Epoch {epoch+1}/{n_epochs}: Loss={total_loss/n_batches:.4f}, Acc={train_acc:.1f}%")

# 测试
model.eval()
with torch.no_grad():
    X_test_device = X_test.to(device)
    outputs = model(X_test_device)
    _, predicted = torch.max(outputs, 1)
    test_acc = 100 * (predicted == y_test.to(device)).sum().item() / len(y_test)

print(f"\n测试集准确率: {test_acc:.1f}%")

# ============================================================
# 第九部分：CNN vs MLP 参数量对比
# ============================================================
print("\n" + "=" * 70)
print("第九部分：CNN vs MLP 参数量对比")
print("=" * 70)

# 处理32×32图像的不同方式
print("\n【处理 32×32×3 图像的不同方式对比】")

# 1. MLP方式
class FlattenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

mlp = FlattenMLP()
mlp_params = sum(p.numel() for p in mlp.parameters())

# 2. CNN方式（现代设计）
cnn = ModernCNN()
cnn_params = sum(p.numel() for p in cnn.parameters())

print(f"\n{'架构':<20} {'参数量':>15} {'参数量对比':>15}")
print("-" * 55)
print(f"{'MLP (Flatten)':<20} {mlp_params:>15,} {'基准':>15}")
print(f"{'Modern CNN':<20} {cnn_params:>15,} {f'{mlp_params/cnn_params:.1f}×更少':>15}")

print(f"\nCNN参数量只有MLP的 {100*cnn_params/mlp_params:.1f}%，但性能通常更好！")

# ============================================================
# 总结与实践
# ============================================================
print("\n" + "=" * 70)
print("总结与实践")
print("=" * 70)

print("""
【本课核心知识点】
✅ 卷积层：局部连接、权值共享，提取空间特征
✅ 池化层：降维、平移不变性
✅ BatchNorm：加速训练、稳定梯度
✅ 残差连接：解决深层网络退化问题
✅ 现代设计：GAP替代全连接，减少参数

【CNN架构演进】
LeNet (1998) → AlexNet (2012) → VGGNet (2014) → 
ResNet (2015) → DenseNet (2017) → EfficientNet (2019)

【卷积层参数速查】
- kernel_size：卷积核大小，常用3×3
- stride：步长，下采样用2
- padding：填充，保持尺寸用kernel//2
- dilation：空洞卷积，增大感受野

【设计原则】
1. 小卷积核（3×3）多次堆叠 > 大卷积核
2. 逐渐下采样（stride=2）+ 增加通道数
3. BatchNorm + ReLU 紧跟卷积层
4. 最后用GAP + 全连接层分类
""")

print("\n【课后练习】")
print("1. 修改ModernCNN的层数，观察参数量和训练速度变化")
print("2. 尝试添加残差连接，训练更深层的网络")
print("3. 使用torchvision加载真实CIFAR-10数据集进行训练")
print("4. 可视化卷积核权重，看看网络学到了什么")
print("\n下节课预告：循环神经网络（RNN/LSTM）- 处理序列数据的利器")

print("\n" + "=" * 70)
print("第13课完成！🎉")
print("=" * 70)
