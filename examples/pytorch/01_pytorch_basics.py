"""
PyTorch第一课：张量基础
学习目标：
1. 理解PyTorch张量（Tensor）与NumPy数组的关系
2. 掌握张量的创建和操作
3. 了解GPU加速支持
4. 初步认识自动求导（Autograd）

PyTorch是深度学习的主流框架，也是构建大模型的基础工具！
"""

import torch
import numpy as np

# ==================== PyTorch简介 ====================
print("=" * 60)
print("🔥 PyTorch简介")
print("=" * 60)

print(f"""
PyTorch是什么？
- Facebook开发的深度学习框架
- 动态计算图，调试方便
- 与NumPy无缝衔接
- 支持GPU加速（CUDA）
- 学术界和工业界的主流选择

为什么选择PyTorch？
1. 易用性：Pythonic的API设计
2. 灵活性：动态图，随时调试
3. 生态好：Hugging Face、各种预训练模型都基于PyTorch
4. 大模型：GPT、BERT、LLaMA等都是PyTorch构建

当前PyTorch版本: {torch.__version__}
""")

# 检查GPU可用性
print(f"\n【硬件环境】")

# 检查CUDA (NVIDIA)
cuda_available = torch.cuda.is_available()
print(f"CUDA可用: {cuda_available}")
if cuda_available:
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")

# 检查MPS (Apple Silicon - M1/M2/M3/M4)
mps_available = torch.backends.mps.is_available()
print(f"\nMPS可用: {mps_available}")
if mps_available:
    print(f"Apple Silicon GPU (MPS) 已启用")
    print(f"MPS设备已构建: {torch.backends.mps.is_built()}")

if not cuda_available and not mps_available:
    print("\n使用CPU进行计算（演示目的足够）")


# ==================== 张量（Tensor）基础 ====================
print("\n" + "=" * 60)
print("📦 张量（Tensor）基础")
print("=" * 60)

print("""
张量（Tensor）是PyTorch的核心数据结构：
- 0维：标量（Scalar）
- 1维：向量（Vector）
- 2维：矩阵（Matrix）
- 3维及以上：高维张量

与NumPy数组的区别：
1. 可以在GPU上运行
2. 支持自动求导（Autograd）
3. 可以跟踪计算历史
""")

# 创建张量
print("\n【创建张量】")

# 从列表创建
x_list = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建: {x_list}")
print(f"类型: {type(x_list)}, 数据类型: {x_list.dtype}")

# 指定数据类型
x_float = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
x_int = torch.tensor([1, 2, 3], dtype=torch.int64)
print(f"\nFloat32: {x_float}")
print(f"Int64: {x_int}")

# 从NumPy数组创建
np_array = np.array([1, 2, 3, 4, 5])
x_from_np = torch.from_numpy(np_array)
print(f"\n从NumPy创建: {x_from_np}")

# NumPy和张量共享内存
np_array[0] = 100
print(f"修改NumPy后，张量也改变: {x_from_np}")

# 常用创建函数
print("\n【常用创建函数】")

# 全0、全1、空张量
zeros = torch.zeros(3, 4)
ones = torch.ones(2, 3)
empty = torch.empty(2, 2)  # 未初始化，值是随机的

print(f"zeros(3, 4):\n{zeros}")
print(f"\nones(2, 3):\n{ones}")

# 随机张量（均匀分布）
rand = torch.rand(3, 3)
print(f"\nrand(3, 3):\n{rand}")

# 随机张量（标准正态分布）
randn = torch.randn(3, 3)
print(f"\nrandn(3, 3):\n{randn}")

# 范围张量（类似arange）
arange = torch.arange(0, 10, 2)
print(f"\narange(0, 10, 2): {arange}")

# 线性空间（类似linspace）
linspace = torch.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {linspace}")


# ==================== 张量属性 ====================
print("\n" + "=" * 60)
print("📏 张量属性")
print("=" * 60)

x = torch.randn(3, 4, 5)
print(f"张量 x:\n{x}")
print(f"\n形状 (shape): {x.shape}")
print(f"维度数 (ndim): {x.ndim}")
print(f"元素个数 (numel): {x.numel()}")
print(f"数据类型 (dtype): {x.dtype}")
print(f"设备 (device): {x.device}")


# ==================== 张量操作 ====================
print("\n" + "=" * 60)
print("🔧 张量操作")
print("=" * 60)

# 索引和切片
print("\n【索引和切片】")
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"张量 x:\n{x}")
print(f"\nx[0, 0] = {x[0, 0]}")
print(f"x[1, :] = {x[1, :]}")
print(f"x[:, 0] = {x[:, 0]}")
print(f"x[0:2, 0:2] = \n{x[0:2, 0:2]}")

# 形状操作
print("\n【形状操作】")
x = torch.arange(12)
print(f"原张量: {x}, shape={x.shape}")

x_reshaped = x.view(3, 4)  # view共享内存
print(f"\nview(3, 4):\n{x_reshaped}")

x_reshaped2 = x.reshape(4, 3)  # reshape可能拷贝
print(f"\nreshape(4, 3):\n{x_reshaped2}")

# 展平
x_flat = x_reshaped.flatten()
print(f"\nflatten(): {x_flat}")

# 转置
x_t = x_reshaped.t()
print(f"\n转置 t():\n{x_t}")

# 增加/删除维度
x_unsqueeze = x_reshaped.unsqueeze(0)  # 在位置0增加维度
print(f"\nunsqueeze(0), shape={x_unsqueeze.shape}:\n{x_unsqueeze}")

x_squeeze = x_unsqueeze.squeeze(0)  # 删除大小为1的维度
print(f"\nsqueeze(0), shape={x_squeeze.shape}:\n{x_squeeze}")


# ==================== 张量运算 ====================
print("\n" + "=" * 60)
print("➕ 张量运算")
print("=" * 60)

# 基本运算
print("\n【基本运算】")
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"a:\n{a}")
print(f"\nb:\n{b}")

print(f"\na + b = \n{a + b}")
print(f"\na - b = \n{a - b}")
print(f"\na * b = \n{a * b}")  # 逐元素相乘
print(f"\na / b = \n{a / b}")
print(f"\na ** 2 = \n{a ** 2}")

# 矩阵乘法
print("\n【矩阵乘法】")
matmul = torch.mm(a, b)  # 或 a @ b
print(f"torch.mm(a, b) = \n{matmul}")
print(f"a @ b = \n{a @ b}")

# 广播机制
print("\n【广播机制】")
c = torch.tensor([1, 2, 3])
d = torch.ones(2, 3)
print(f"c: {c}")
print(f"d:\n{d}")
print(f"d + c:\n{d + c}")  # c广播到每一行

# 逐元素函数
print("\n【数学函数】")
x = torch.tensor([0, 1, 2], dtype=torch.float32)
print(f"x: {x}")
print(f"sin(x): {torch.sin(x)}")
print(f"exp(x): {torch.exp(x)}")
print(f"log(x+1): {torch.log(x + 1)}")


# ==================== GPU加速 ====================
print("\n" + "=" * 60)
print("🚀 GPU加速")
print("=" * 60)

# 创建CPU张量（使用更大矩阵展示MPS优势）
matrix_size = 3000
x_cpu = torch.randn(matrix_size, matrix_size)
print(f"CPU张量 ({matrix_size}x{matrix_size}): {x_cpu.device}")

# 检测最佳可用设备
if torch.cuda.is_available():
    device = 'cuda'
    device_name = torch.cuda.get_device_name(0)
    print(f"\n✅ 使用NVIDIA GPU: {device_name}")
elif torch.backends.mps.is_available():
    device = 'mps'
    device_name = 'Apple Silicon GPU (MPS)'
    print(f"\n✅ 使用{device_name} - M4芯片加速！")
else:
    device = None
    print("\n⚠️ 没有可用的GPU，使用CPU")

# 如果有GPU，进行加速演示
if device:
    # 转移到GPU
    x_gpu = x_cpu.to(device)
    print(f"\nGPU张量: {x_gpu.device}")
    
    # 再转回CPU
    x_back = x_gpu.cpu()
    print(f"转回CPU: {x_back.device}")
    
    # GPU运算速度对比
    import time
    
    # 预热GPU
    if device == 'mps':
        _ = x_gpu @ x_gpu  # MPS需要预热
        torch.mps.synchronize()  # 等待MPS完成
    
    print("\n【速度对比】")
    # CPU
    start = time.time()
    result_cpu = x_cpu @ x_cpu
    cpu_time = time.time() - start
    print(f"CPU矩阵乘法用时: {cpu_time:.4f}秒")
    
    # GPU
    if device == 'mps':
        torch.mps.synchronize()  # 确保MPS准备就绪
    
    start = time.time()
    result_gpu = x_gpu @ x_gpu
    
    if device == 'mps':
        torch.mps.synchronize()  # 等待MPS计算完成
    
    gpu_time = time.time() - start
    print(f"{device.upper()}矩阵乘法用时: {gpu_time:.4f}秒")
    
    if gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU加速比: {speedup:.1f}x")
        if speedup > 1:
            print(f"GPU比CPU快 {speedup:.1f} 倍！")
        else:
            print(f"注意：小矩阵在GPU上可能不如CPU快（数据传输开销）")
            print(f"尝试增大矩阵大小（如5000x5000）看效果")

    print("\n💡 M4芯片使用MPS加速提示：")
    print("   - 对于大矩阵运算（>1000x1000）效果显著")
    print("   - 神经网络训练和推理速度大幅提升")
    print("   - 使用 .to('mps') 即可自动启用")
else:
    print("\n没有可用的GPU，跳过GPU演示")


# ==================== 与NumPy互操作 ====================
print("\n" + "=" * 60)
print("🔄 与NumPy互操作")
print("=" * 60)

# Tensor -> NumPy
x_tensor = torch.tensor([1, 2, 3, 4, 5])
x_numpy = x_tensor.numpy()
print(f"Tensor: {x_tensor}")
print(f"NumPy: {x_numpy}")
print(f"NumPy类型: {type(x_numpy)}")

# NumPy -> Tensor（注意：不再共享内存）
np_array = np.array([10, 20, 30])
from_numpy = torch.from_numpy(np_array)  # 共享内存
as_tensor = torch.as_tensor(np_array)     # 共享内存（如果可能）
tensor_new = torch.tensor(np_array)       # 拷贝

print(f"\nfrom_numpy: {from_numpy}")
print(f"as_tensor: {as_tensor}")
print(f"tensor: {tensor_new}")

# 验证共享内存
np_array[0] = 999
print(f"\n修改NumPy数组后:")
print(f"from_numpy也变了: {from_numpy}")
print(f"tensor_new没变: {tensor_new}")


# ==================== 自动求导（Autograd）初探 ====================
print("\n" + "=" * 60)
print("🎓 自动求导（Autograd）初探")
print("=" * 60)

print("""
Autograd是PyTorch的自动微分引擎：
- 自动计算梯度
- 支持反向传播
- 神经网络训练的核心
""")

# 创建需要求导的张量
x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"x = {x}")
print(f"requires_grad = {x.requires_grad}")

# 定义计算
y = x ** 2  # y = [4, 9]
z = y.sum()  # z = 13

print(f"y = x^2 = {y}")
print(f"z = sum(y) = {z}")

# 反向传播
z.backward()

# 查看梯度
# dz/dx = [2*x] = [4, 6]
print(f"\n梯度 dz/dx = {x.grad}")

# 验证手动计算
print(f"手动验证: 2*x = {2 * x}")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：数据预处理
print("\n【场景1：数据预处理】")
# 模拟图像数据：3通道，224x224，batch_size=4
batch_size = 4
channels = 3
height = 224
width = 224

images = torch.randn(batch_size, channels, height, width)
print(f"图像批次 shape: {images.shape}")
print(f"  批量大小 (batch): {images.shape[0]}")
print(f"  通道数 (channels): {images.shape[1]}")
print(f"  高度 (height): {images.shape[2]}")
print(f"  宽度 (width): {images.shape[3]}")

# 归一化
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
normalized = (images - mean) / std
print(f"\n归一化后 shape: {normalized.shape}")

# 场景2：线性回归（简单神经网络）
print("\n【场景2：简单线性回归】")
print("数据: y = 2x + 0, 我们训练模型学习这个关系")

# 数据
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 初始化权重（随机初始化）
torch.manual_seed(42)  # 固定随机种子，结果可复现
w = torch.randn(1, 1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

print(f"\n初始权重: w={w.item():.4f}, b={b.item():.4f}")
print(f"目标权重: w=2.0, b=0.0")

# 训练参数
learning_rate = 0.05  # 增大学习率
epochs = 1000  # 增加迭代次数

# 训练循环
for epoch in range(epochs):
    # 前向传播
    y_pred = X @ w + b
    
    # 计算损失（均方误差）
    loss = ((y_pred - y) ** 2).mean()
    
    # 反向传播
    loss.backward()
    
    # 更新参数（在no_grad环境下，不记录梯度）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清零梯度（重要！）
        w.grad.zero_()
        b.grad.zero_()
    
    # 每200轮打印一次进度
    if (epoch + 1) % 200 == 0:
        print(f"  Epoch {epoch+1}: Loss={loss.item():.6f}, w={w.item():.4f}, b={b.item():.4f}")

print(f"\n训练后权重: w={w.item():.4f}, b={b.item():.4f}")
print(f"目标权重: w=2.0, b=0.0")
print(f"误差: Δw={abs(w.item() - 2.0):.4f}, Δb={abs(b.item()):.4f}")
print(f"\n✅ 学习结果: y ≈ {w.item():.2f} * x + {b.item():.2f}")

# 验证
print("\n预测验证:")
for i in range(len(X)):
    pred = (X[i] * w + b).item()
    actual = y[i].item()
    print(f"  x={X[i].item():.0f}: 预测={pred:.2f}, 实际={actual:.0f}, 误差={abs(pred-actual):.4f}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：张量创建
创建以下张量：
- 3x3的随机张量（标准正态分布）
- 5x5的全0张量，对角线为1（类似单位矩阵）
- 从列表[1, 2, 3, 4, 5, 6]创建2x3的Float32张量

练习2：张量操作
给定张量 x = torch.arange(24).view(4, 6)：
- 提取第2行（索引1）
- 提取最后2列
- 将形状改为(2, 3, 4)

练习3：广播运算
给定 a = torch.ones(3, 4) 和 b = torch.tensor([1, 2, 3, 4])：
- 使用广播将b加到a的每一行
- 计算结果的均值

练习4：GPU转移（M4芯片使用MPS）
- 创建大小为3000x3000的随机张量
- 转移到MPS设备（.to('mps')）
- 进行矩阵乘法并计算时间
- 对比CPU和MPS的速度

练习5：自动求导
给定 x = torch.tensor([3.0], requires_grad=True)：
- 定义 y = x^2 + 2*x + 1
- 计算dy/dx
- 手动验证结果（应该是 2*3 + 2 = 8）
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1
print("\n练习1：张量创建")
x1 = torch.randn(3, 3)
print(f"3x3随机张量:\n{x1}")

x2 = torch.eye(5)
print(f"\n5x5单位矩阵:\n{x2}")

x3 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32).view(2, 3)
print(f"\n2x3 Float32:\n{x3}")

# 练习2
print("\n练习2：张量操作")
x = torch.arange(24).view(4, 6)
print(f"原张量 shape={x.shape}:\n{x}")
print(f"\n第2行: {x[1]}")
print(f"最后2列:\n{x[:, -2:]}")
x_reshaped = x.view(2, 3, 4)
print(f"\nreshape为(2,3,4) shape={x_reshaped.shape}")

# 练习3
print("\n练习3：广播运算")
a = torch.ones(3, 4)
b = torch.tensor([1, 2, 3, 4])
result = a + b
print(f"a + b:\n{result}")
print(f"均值: {result.mean().item()}")

# 练习4
print("\n练习4：GPU转移（M4 MPS加速）")
if torch.backends.mps.is_available() or torch.cuda.is_available():
    device = 'mps' if torch.backends.mps.is_available() else 'cuda'
    large_cpu = torch.randn(3000, 3000)
    large_gpu = large_cpu.to(device)
    
    # 预热
    if device == 'mps':
        _ = large_gpu @ large_gpu
        torch.mps.synchronize()
    
    start = time.time()
    result = large_gpu @ large_gpu
    if device == 'mps':
        torch.mps.synchronize()
    gpu_time = time.time() - start
    print(f"3000x3000矩阵乘法在{device.upper()}用时: {gpu_time:.4f}秒")
    
    # CPU对比
    start = time.time()
    result_cpu = large_cpu @ large_cpu
    cpu_time = time.time() - start
    print(f"3000x3000矩阵乘法在CPU用时: {cpu_time:.4f}秒")
    print(f"加速比: {cpu_time/gpu_time:.1f}x")
else:
    print("无GPU可用，跳过")

# 练习5
print("\n练习5：自动求导")
x = torch.tensor([3.0], requires_grad=True)
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1
y.backward()
print(f"x = {x.item()}")
print(f"y = {y.item()}")
print(f"dy/dx = {x.grad.item()}")
print(f"手动验证: 2*3 + 2 = {2*3 + 2}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ PyTorch简介：深度学习主流框架
✅ 张量（Tensor）：PyTorch的核心数据结构
✅ 张量创建：tensor, zeros, ones, rand, arange, linspace
✅ 张量操作：索引、切片、reshape、transpose
✅ 张量运算：基本运算、矩阵乘法、广播
✅ GPU加速：cuda()、to('cuda')、cpu()
✅ NumPy互操作：numpy()、from_numpy()、as_tensor()
✅ Autograd：requires_grad、backward()、grad

下节课预告：自动求导深入 + 神经网络构建
""")
