# 第13课：卷积神经网络（CNN）

## 学习目标
- [x] 理解为什么CNN比MLP更适合图像
- [x] 掌握卷积层的工作原理（卷积核、步长、填充）
- [x] 理解池化层的作用和类型
- [x] 学会使用批归一化加速训练
- [x] 掌握现代CNN设计技巧
- [ ] 能够设计自己的CNN架构

---

## 1. 为什么需要CNN？

### 1.1 全连接层的困境

**问题：用MLP处理图像代价巨大**

| 图像尺寸 | 输入特征数 | 第一层256神经元 | 参数量 |
|---------|-----------|---------------|-------|
| 32×32×3 (CIFAR-10) | 3,072 | 3,072 × 256 | 78.6万 |
| 224×224×3 (ImageNet) | 150,528 | 150,528 × 256 | **3,854万** |

**问题**：
- 参数太多 → 容易过拟合
- 计算量巨大 → 训练困难
- 忽略空间结构 → 像素平移等于全新输入

### 1.2 CNN的三大核心优势

| 优势 | 解释 | 效果 |
|-----|------|------|
| **局部连接** | 卷积核只关注局部区域（如3×3） | 参数大幅减少 |
| **权值共享** | 同一个卷积核在整个图像滑动 | 进一步减少参数 |
| **平移不变性** | 同一特征在不同位置都能检测 | 更好的泛化能力 |

### 1.3 层次特征学习

```
输入图像
    ↓
[浅层] 边缘、颜色、纹理
    ↓
[中层] 形状、角点、模式
    ↓
[深层] 物体部件、语义
    ↓
[输出] 分类结果
```

---

## 2. 卷积层详解

### 2.1 卷积运算原理

```
输入图像 (5×5)          卷积核 (3×3)          输出特征图 (3×3)
┌───┬───┬───┬───┬───┐   ┌───┬───┬───┐
│ 1 │ 1 │ 1 │ 0 │ 0 │   │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┤ × ├───┼───┼───┤  =  滑动窗口点积求和
│ 0 │ 1 │ 1 │ 1 │ 0 │   │ 0 │ 1 │ 0 │
├───┼───┼───┼───┼───┤   ├───┼───┼───┤
│ 0 │ 0 │ 1 │ 1 │ 1 │   │ 1 │ 0 │ 1 │
├───┼───┼───┼───┼───┤   └───┴───┴───┘
│ 0 │ 0 │ 1 │ 1 │ 0 │
├───┼───┼───┼───┼───┤
│ 0 │ 1 │ 1 │ 0 │ 0 │
└───┴───┴───┴───┴───┘
```

**卷积核** = 特征检测器，每个核学习一种特征（边缘、纹理等）

### 2.2 PyTorch卷积层

```python
nn.Conv2d(
    in_channels=3,      # 输入通道数（RGB=3）
    out_channels=16,    # 输出通道数 = 卷积核数量
    kernel_size=3,      # 卷积核大小，常用3×3
    stride=1,           # 步长，默认1
    padding=1,          # 填充，保持尺寸用kernel//2
    dilation=1,         # 空洞率，默认1
    groups=1,           # 分组卷积，默认1
    bias=True           # 是否使用偏置
)
```

### 2.3 输出尺寸计算公式

$$
\text{Output} = \frac{\text{Input} - \text{Kernel} + 2 \times \text{Padding}}{\text{Stride}} + 1
$$

**常用配置速查表**：

| 输入 | 卷积核 | 步长 | 填充 | 输出 | 作用 |
|-----|-------|-----|------|------|------|
| 32 | 3 | 1 | 1 | 32 | 保持尺寸 |
| 32 | 3 | 2 | 1 | 16 | 下采样2× |
| 32 | 4 | 2 | 1 | 16 | 更平滑的下采样 |
| 32 | 5 | 1 | 2 | 32 | 更大感受野 |

### 2.4 代码示例

```python
import torch.nn as nn

# 保持尺寸
conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
# 32×32 → 32×32

# 下采样
conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
# 32×32 → 16×16

# 1×1卷积（降维/升维）
conv3 = nn.Conv2d(128, 64, kernel_size=1)
# 16×16×128 → 16×16×64（通道减半）
```

---

## 3. 池化层

### 3.1 池化的作用

1. **降维**：减少空间尺寸，降低计算量
2. **特征聚合**：保留主要特征
3. **平移不变性**：轻微位置变化不影响结果

### 3.2 池化类型

```python
# 最大池化 - 保留显著特征
nn.MaxPool2d(kernel_size=2, stride=2)

# 平均池化 - 保留整体信息
nn.AvgPool2d(kernel_size=2, stride=2)

# 自适应池化 - 指定输出尺寸
nn.AdaptiveMaxPool2d((1, 1))  # 输出1×1
nn.AdaptiveAvgPool2d(1)       # 输出1×1
```

### 3.3 使用建议

| 场景 | 推荐池化 | 原因 |
|-----|---------|------|
| 中间层 | MaxPool | 保留显著特征 |
| 全局特征 | AdaptiveAvgPool | 聚合全局信息 |
| 输出层（替代全连接）| GAP (Global Average Pooling) | 减少参数，提高泛化 |

---

## 4. 批归一化（BatchNorm）

### 4.1 为什么需要BN？

**问题**：深层网络中，每一层的输入分布会随着前层参数变化而变化（内部协变量偏移）

**BN的作用**：
1. 稳定每层的输入分布
2. 允许使用更大学习率
3. 减少对初始化的敏感
4. 轻微正则化效果

### 4.2 原理

```python
# 对每个batch、每个通道
μ = mean(x)          # 计算均值
σ² = var(x)          # 计算方差
x̂ = (x - μ) / √(σ² + ε)  # 归一化
y = γ × x̂ + β        # 缩放和偏移（可学习参数）
```

### 4.3 PyTorch实现

```python
# 2D数据（图像）
nn.BatchNorm2d(num_features=64)

# 1D数据（序列）
nn.BatchNorm1d(num_features=128)

# 使用位置
conv → bn → activation
```

### 4.4 使用注意事项

```python
# 训练模式：使用batch统计
model.train()

# 评估模式：使用训练时保存的running统计
model.eval()
```

---

## 5. 构建CNN

### 5.1 LeNet（经典基础架构）

```python
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 阶段1: 32×32 → 16×16
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 阶段2: 16×16 → 8×8
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 全连接
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

### 5.2 现代CNN设计原则

```python
class ModernCNN(nn.Module):
    """现代CNN设计模板"""
    def __init__(self, num_classes=10):
        super().__init__()
        
        # 阶段1: 保持通道数，下采样
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32→16
        )
        
        # 阶段2: 通道数翻倍
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16→8
        )
        
        # 阶段3: 通道数再翻倍
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 8→4
        )
        
        # GAP + 分类
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)  # [B, 128, 4, 4] → [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

### 5.3 设计原则总结

| 原则 | 说明 |
|-----|------|
| 通道递增 | 空间分辨率下降时，通道数翻倍（32→64→128） |
| 小卷积核 | 3×3卷积多次堆叠 > 大卷积核 |
| BatchNorm | 每个卷积层后都加BN |
| 池化策略 | 用stride=2卷积或MaxPool进行下采样 |
| GAP结尾 | 用全局平均池化替代全连接层 |

---

## 6. 残差连接（ResNet核心）

### 6.1 为什么需要残差？

**问题**：深层网络（>20层）出现退化问题，训练准确率反而下降

**原因**：梯度消失，深层特征难以有效传递

### 6.2 残差块原理

```
        ┌───────────────────────┐
        │                       │
  x ────┼──→ [Conv→BN→ReLU] ──→ [Conv→BN] ──→ + ──→ ReLU ──→ y
        │                                           ↑
        └───────────────────────────────────────────┘
                       残差连接 (skip connection)

y = F(x) + x
  = (Conv(Conv(x))) + x
```

学习残差 $F(x)$ 比直接学习 $H(x)$ 更容易！

### 6.3 残差块实现

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
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
```

### 6.4 ResNet架构规模

| 模型 | 层数 | 参数量 | ImageNet Top-1 |
|-----|------|-------|---------------|
| ResNet-18 | 18 | 11.7M | 69.8% |
| ResNet-34 | 34 | 21.8M | 73.3% |
| ResNet-50 | 50 | 25.6M | 76.1% |
| ResNet-101 | 101 | 44.5M | 77.4% |

---

## 7. CNN vs MLP 参数量对比

### 7.1 处理32×32×3图像

| 架构 | 参数量 | 相对比例 |
|-----|-------|---------|
| MLP (3层) | ~80万 | 100% |
| LeNet | ~6万 | 7.5% |
| ModernCNN | ~20万 | 25% |

**结论**：CNN参数量更少，但性能通常更好！

### 7.2 参数量计算

```python
# 卷积层参数量 = (输入通道 × 卷积核高 × 卷积核宽 + 1) × 输出通道
conv_params = (in_ch * k_h * k_w + 1) * out_ch

# 全连接层参数量 = (输入特征 + 1) × 输出特征
fc_params = (in_features + 1) * out_features
```

---

## 8. 今日要点总结

### 8.1 核心公式

```
卷积输出尺寸:  O = (I - K + 2P) / S + 1

感受野计算:    RF_out = RF_in + (K - 1) × S_in
```

### 8.2 速查表

| 组件 | 常用配置 | 作用 |
|-----|---------|------|
| Conv2d | kernel=3, stride=1, padding=1 | 特征提取，保持尺寸 |
| MaxPool2d | kernel=2, stride=2 | 下采样，保留显著特征 |
| BatchNorm2d | 通道数 | 加速训练，稳定梯度 |
| AdaptiveAvgPool2d | output_size=1 | 全局特征聚合 |

### 8.3 CNN架构演进

```
1998  LeNet      → 奠基之作，5层
2012  AlexNet    → ReLU+Dropout，8层
2014  VGGNet     → 小卷积核3×3，16-19层
2015  ResNet     → 残差连接，可训练152+层
2017  DenseNet   → 密集连接，特征复用
2019  EfficientNet → 复合缩放，效率最优
```

---

## 9. 课后练习

1. **修改网络深度**：增加或减少卷积层，观察参数量和性能变化

2. **添加残差连接**：在ModernCNN中加入残差块，训练深层网络

3. **真实数据训练**：
   ```python
   import torchvision
   trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
   ```

4. **可视化卷积核**：
   ```python
   # 查看第一层卷积核学到的特征
   conv_weights = model.conv1.weight.data
   ```

5. **感受野计算**：手动计算你的网络最后一层卷积的感受野大小

---

## 10. 下节课预告

**循环神经网络（RNN/LSTM）**
- 处理序列数据的挑战
- RNN的基本结构
- LSTM解决长依赖问题
- 应用于文本、时间序列

---

*学习日期：2026-05-24*
