# PyTorch第一课：张量基础

## 学习目标
- [ ] 理解PyTorch张量（Tensor）与NumPy数组的关系
- [ ] 掌握张量的创建和操作
- [ ] 了解GPU加速支持
- [ ] 初步认识自动求导（Autograd）

## PyTorch简介

**为什么选择PyTorch？**
1. 易用性：Pythonic的API设计
2. 灵活性：动态图，随时调试
3. 生态好：Hugging Face、各种预训练模型都基于PyTorch
4. 大模型：GPT、BERT、LLaMA等都是PyTorch构建

**检查GPU**：
```python
import torch
print(torch.cuda.is_available())  # 是否有GPU
print(torch.cuda.get_device_name(0))  # GPU型号
```

## 张量（Tensor）

### 创建张量

```python
# 从列表创建
x = torch.tensor([1, 2, 3, 4, 5])

# 指定数据类型
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

# 从NumPy创建（共享内存）
import numpy as np
np_array = np.array([1, 2, 3])
x = torch.from_numpy(np_array)

# 常用创建函数
torch.zeros(3, 4)        # 全0
torch.ones(2, 3)         # 全1
torch.rand(3, 3)         # 均匀分布随机
torch.randn(3, 3)        # 标准正态分布
torch.arange(0, 10, 2)   # 等差数列
torch.linspace(0, 1, 5)  # 线性空间
```

### 张量属性

```python
x.shape      # 形状
x.ndim       # 维度数
x.numel()    # 元素总数
x.dtype      # 数据类型
x.device     # 所在设备（CPU/GPU）
```

### 形状操作

```python
# 重塑形状
x.view(3, 4)      # 共享内存
x.reshape(3, 4)   # 可能拷贝

# 展平
x.flatten()

# 转置
x.t()

# 增加/删除维度
x.unsqueeze(0)    # 在位置0增加维度
x.squeeze(0)      # 删除大小为1的维度
```

### 索引和切片

```python
x[0, 0]       # 单个元素
x[1, :]       # 第1行
x[:, 0]       # 第0列
x[0:2, 0:2]   # 子矩阵
```

## 张量运算

### 基本运算

```python
# 逐元素运算
a + b
a - b
a * b      # 逐元素相乘
a / b
a ** 2

# 矩阵乘法
torch.mm(a, b)  # 或 a @ b
```

### 广播机制

```python
# 与NumPy相同
a = torch.ones(2, 3)
b = torch.tensor([1, 2, 3])
a + b  # b广播到每一行
```

### 数学函数

```python
torch.sin(x)
torch.exp(x)
torch.log(x)
```

## GPU加速

### NVIDIA GPU (CUDA)
```python
# CPU -> GPU
x_gpu = x.cuda()
x_gpu = x.to('cuda')

# GPU -> CPU
x_cpu = x_gpu.cpu()
x_cpu = x_gpu.to('cpu')
```

### Apple Silicon (MPS - Metal Performance Shaders)
M1/M2/M3/M4 芯片使用 MPS 后端加速：

```python
# 检查MPS是否可用
print(torch.backends.mps.is_available())  # True on M4

# CPU -> MPS
x_mps = x.to('mps')

# MPS -> CPU
x_cpu = x_mps.cpu()

# 自动选择最佳设备
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
x = x.to(device)
```

**M4芯片使用提示：**
- 小矩阵（<1000x1000）可能CPU更快（数据传输开销）
- 大矩阵和神经网络运算 MPS 优势明显
- 使用 `torch.mps.synchronize()` 确保计时准确

## 与NumPy互操作

```python
# Tensor -> NumPy（共享内存）
np_array = x.numpy()

# NumPy -> Tensor
x = torch.from_numpy(np_array)   # 共享内存
x = torch.as_tensor(np_array)    # 共享内存
x = torch.tensor(np_array)       # 拷贝
```

## 自动求导（Autograd）

```python
# 创建需要求导的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 定义计算
y = x ** 2
z = y.sum()

# 反向传播
z.backward()

# 查看梯度
print(x.grad)  # dz/dx
```

## 遇到问题记录
（在这里记录学习过程中遇到的任何问题）

## 练习记录
- [ ] 完成 `01_pytorch_basics.py` 中的5个练习
- [ ] 尝试创建不同形状的张量

## 下节课预告
- 自动求导深入
- 神经网络构建
