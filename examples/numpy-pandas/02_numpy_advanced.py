"""
NumPy第二课：进阶应用（线性代数、随机数、广播深入）
学习目标：
1. 掌握NumPy线性代数运算（矩阵运算、特征值、SVD）
2. 学会高级随机数生成
3. 深入理解广播机制
4. 学会数组的保存和加载

这是理解和实现深度学习算法的基础！
"""

import numpy as np

# ==================== 线性代数基础 ====================
print("=" * 60)
print("📐 线性代数基础")
print("=" * 60)

# 矩阵和向量
print("\n【矩阵和向量】")
A = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2矩阵
B = np.array([[1, 2, 3], [4, 5, 6]])    # 2x3矩阵
v = np.array([1, 2, 3])                  # 向量

print(f"矩阵A (3x2):\n{A}")
print(f"矩阵B (2x3):\n{B}")
print(f"向量v: {v}")

# 矩阵乘法
print("\n【矩阵乘法】")
C = A @ B  # 3x2 @ 2x3 = 3x3
print(f"A @ B (3x3):\n{C}")

# 矩阵与向量乘法
print("\n【矩阵与向量】")
D = np.array([[1, 2], [3, 4]])
result = D @ v[:2]  # 2x2 @ 2x1
print(f"D @ v[:2]: {result}")

# 转置
print("\n【转置】")
print(f"A的转置:\n{A.T}")
print(f"A.T的形状: {A.T.shape}")  # 2x3

# 逆矩阵（必须是方阵）
print("\n【逆矩阵】")
E = np.array([[1, 2], [3, 4]])
E_inv = np.linalg.inv(E)
print(f"矩阵E:\n{E}")
print(f"E的逆矩阵:\n{E_inv}")
print(f"验证 E @ E_inv:\n{E @ E_inv}")  # 接近单位矩阵


# ==================== 线性代数高级运算 ====================
print("\n" + "=" * 60)
print("🔬 线性代数高级运算")
print("=" * 60)

# 行列式
print("\n【行列式】")
det_E = np.linalg.det(E)
print(f"det(E) = {det_E:.4f}")

# 矩阵的秩
print("\n【矩阵的秩】")
rank = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank}")

# 迹（对角线元素之和）
print("\n【迹】")
trace = np.trace(E)
print(f"tr(E) = {trace}")

# 特征值和特征向量
print("\n【特征值和特征向量】")
F = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(F)
print(f"矩阵F:\n{F}")
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证：F @ v = λ * v
for i, (eigval, eigvec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    left = F @ eigvec
    right = eigval * eigvec
    print(f"\n特征向量{i+1}验证:")
    print(f"  F @ v = {left}")
    print(f"  λ * v = {right}")
    print(f"  是否相等: {np.allclose(left, right)}")


# SVD分解（奇异值分解）- 非常重要！
print("\n【SVD分解 - 奇异值分解】")
G = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, S, Vt = np.linalg.svd(G)

print(f"原矩阵G (3x3):\n{G}")
print(f"U (左奇异向量):\n{U}")
print(f"S (奇异值): {S}")
print(f"Vt (右奇异向量转置):\n{Vt}")

# 验证 SVD: G = U @ diag(S) @ Vt
S_diag = np.diag(S)
reconstructed = U @ S_diag @ Vt
print(f"\n重构验证:")
print(f"U @ diag(S) @ Vt:\n{reconstructed}")
print(f"是否接近原矩阵: {np.allclose(G, reconstructed)}")

# SVD的应用：降维
print("\n【SVD应用：降维】")
# 只保留前2个奇异值
k = 2
G_reduced = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
print(f"降维后 (k={k}):\n{G_reduced}")


# ==================== 方程组求解 ====================
print("\n" + "=" * 60)
print("🔢 线性方程组求解")
print("=" * 60)

# Ax = b
print("\n【求解 Ax = b】")
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])

print(f"系数矩阵A:\n{A}")
print(f"常数向量b: {b}")

# 方法1：使用inv（不推荐，数值不稳定）
x1 = np.linalg.inv(A) @ b
print(f"\n方法1 (inv): x = {x1}")

# 方法2：使用solve（推荐）
x2 = np.linalg.solve(A, b)
print(f"方法2 (solve): x = {x2}")

# 验证
print(f"\n验证 A @ x = {A @ x2}")

# 最小二乘解（超定方程组）
print("\n【最小二乘解】")
A_over = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])  # 4x2
b_over = np.array([2, 4, 5, 4])  # 4个方程，2个未知数

x_ls, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"最小二乘解: {x_ls}")
print(f"残差: {residuals}")


# ==================== 随机数进阶 ====================
print("\n" + "=" * 60)
print("🎲 随机数进阶")
print("=" * 60)

# 设置随机种子（可重复性）
np.random.seed(42)

# 各种分布的随机数
print("\n【各种分布】")

# 均匀分布 [a, b)
uniform = np.random.uniform(0, 10, 5)
print(f"均匀分布[0,10): {uniform}")

# 正态分布（高斯分布）
normal = np.random.normal(0, 1, 5)
print(f"标准正态分布: {normal}")

# 指定均值和标准差的正态分布
normal2 = np.random.normal(100, 15, 5)  # 均值100，标准差15
print(f"正态分布(μ=100, σ=15): {normal2}")

# 整数随机数
randint = np.random.randint(1, 100, 5)
print(f"整数随机数[1,100): {randint}")

# 泊松分布
poisson = np.random.poisson(5, 5)
print(f"泊松分布(λ=5): {poisson}")

# 指数分布
exponential = np.random.exponential(1, 5)
print(f"指数分布(λ=1): {exponential}")

# 二项分布
binomial = np.random.binomial(10, 0.5, 5)
print(f"二项分布(n=10,p=0.5): {binomial}")


# 随机采样
print("\n【随机采样】")
arr = np.arange(10)

# 随机打乱
shuffled = arr.copy()
np.random.shuffle(shuffled)
print(f"打乱后: {shuffled}")

# 随机选择（有放回）
choice_with = np.random.choice(arr, 5, replace=True)
print(f"有放回采样5个: {choice_with}")

# 随机选择（无放回）
choice_without = np.random.choice(arr, 5, replace=False)
print(f"无放回采样5个: {choice_without}")

# 按概率采样
probabilities = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
choice_prob = np.random.choice(arr, 5, p=probabilities)
print(f"按概率采样: {choice_prob}")


# ==================== 广播机制深入 ====================
print("\n" + "=" * 60)
print("📡 广播机制深入")
print("=" * 60)

print("""
广播三规则：
1. 维度从后向前对齐
2. 维度相等或其中一个为1时，可以广播
3. 否则报错
""")

# 示例1：标量广播（最简单）
print("\n【示例1：标量广播】")
arr = np.array([1, 2, 3])
print(f"{arr} + 10 = {arr + 10}")

# 示例2：不同形状数组广播
print("\n【示例2：不同形状广播】")
a = np.array([[1, 2, 3], [4, 5, 6]])      # 2x3
b = np.array([10, 20, 30])               # (3,)
print(f"a shape: {a.shape}")
print(f"b shape: {b.shape}")
print(f"a + b:\n{a + b}")  # b广播为2x3

# 示例3：需要添加维度
print("\n【示例3：添加维度后广播】")
c = np.array([[1], [2], [3]])  # 3x1
d = np.array([10, 20, 30])      # (3,)
print(f"c shape: {c.shape}")
print(f"d shape: {d.shape}")
# 需要对齐维度
d_col = d[:, np.newaxis]  # 3x1
print(f"d[:, np.newaxis] shape: {d_col.shape}")
print(f"c + d_col:\n{c + d_col}")


# 实际应用：数据标准化
print("\n【实际应用：数据标准化】")
data = np.random.randn(5, 3) * 10 + 50  # 5个样本，3个特征
print(f"原始数据 shape={data.shape}:\n{data}")

# 按特征标准化（每列减去均值，除以标准差）
mean = data.mean(axis=0)  # 每列的均值
std = data.std(axis=0)    # 每列的标准差

print(f"\n均值 (每列): {mean}")
print(f"标准差 (每列): {std}")

# 广播标准化
data_normalized = (data - mean) / std
print(f"\n标准化后:\n{data_normalized}")
print(f"标准化后均值: {data_normalized.mean(axis=0)}")  # 接近0
print(f"标准化后标准差: {data_normalized.std(axis=0)}")  # 接近1


# ==================== 数组保存和加载 ====================
print("\n" + "=" * 60)
print("💾 数组保存和加载")
print("=" * 60)

# 创建示例数据
arr_save = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"要保存的数组:\n{arr_save}")

# 保存为.npy文件（二进制，NumPy专用）
print("\n【保存为.npy文件】")
np.save("temp_array.npy", arr_save)
print("已保存到 temp_array.npy")

# 加载.npy文件
loaded = np.load("temp_array.npy")
print(f"加载的数组:\n{loaded}")

# 保存为.txt文件（文本，可读）
print("\n【保存为文本文件】")
np.savetxt("temp_array.txt", arr_save, fmt="%d", delimiter=",")
print("已保存到 temp_array.txt")

# 加载文本文件
loaded_txt = np.loadtxt("temp_array.txt", delimiter=",", dtype=int)
print(f"从文本加载:\n{loaded_txt}")

# 保存多个数组
print("\n【保存多个数组】")
arr1 = np.array([1, 2, 3])
arr2 = np.array([[4, 5], [6, 7]])
np.savez("temp_arrays.npz", first=arr1, second=arr2)
print("已保存到 temp_arrays.npz")

# 加载多个数组
data = np.load("temp_arrays.npz")
print(f"加载的数组 'first': {data['first']}")
print(f"加载的数组 'second':\n{data['second']}")

# 清理临时文件
import os
os.remove("temp_array.npy")
os.remove("temp_array.txt")
os.remove("temp_arrays.npz")
print("\n已清理临时文件")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：图像处理（矩阵运算）
print("\n【场景1：简单的图像处理】")
# 模拟3x3图像
image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"图像:\n{image}")

# 简单的图像操作：反转
image_flipped = np.flip(image, axis=0)
print(f"垂直翻转:\n{image_flipped}")

# 图像归一化到0-1
image_normalized = (image - image.min()) / (image.max() - image.min())
print(f"归一化到[0,1]:\n{image_normalized}")


# 场景2：数据降维（PCA简化版）
print("\n【场景2：PCA降维】")
# 生成5个样本，每个样本3维数据
np.random.seed(42)
X = np.random.randn(5, 3)
print(f"原始数据 (5x3):\n{X}")

# 1. 中心化
X_centered = X - X.mean(axis=0)

# 2. 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)
print(f"\n协方差矩阵:\n{cov_matrix}")

# 3. 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"\n特征值: {eigenvalues}")

# 4. 按特征值排序，取前k个
k = 2
idx = eigenvalues.argsort()[::-1][:k]
W = eigenvectors[:, idx]

# 5. 投影
X_reduced = X_centered @ W
print(f"\n降维后 (5x2):\n{X_reduced}")


# 场景3：批量数据处理
print("\n【场景3：批量距离计算】")
# 3个样本点
points = np.array([[0, 0], [3, 4], [6, 8]])
# 2个查询点
queries = np.array([[1, 1], [5, 5]])

print(f"样本点:\n{points}")
print(f"查询点:\n{queries}")

# 计算所有查询点到所有样本点的距离
# 利用广播：queries (2x1x2) - points (1x3x2) = (2x3x2)
diff = queries[:, np.newaxis, :] - points[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=2))

print(f"\n距离矩阵 (查询点 x 样本点):\n{distances}")

# 找出最近的样本点
nearest = distances.argmin(axis=1)
print(f"每个查询点最近的样本点索引: {nearest}")


# 场景4：随机初始化神经网络权重
print("\n【场景4：神经网络权重初始化】")

# Xavier初始化
input_size, output_size = 784, 256

# 均匀分布 Xavier
W_xavier = np.random.uniform(
    -np.sqrt(6 / (input_size + output_size)),
    np.sqrt(6 / (input_size + output_size)),
    (input_size, output_size)
)
print(f"Xavier初始化权重 shape: {W_xavier.shape}")
print(f"  均值: {W_xavier.mean():.6f}")
print(f"  标准差: {W_xavier.std():.6f}")

# He初始化（ReLU常用）
W_he = np.random.normal(0, np.sqrt(2 / input_size), (input_size, output_size))
print(f"\nHe初始化权重 shape: {W_he.shape}")
print(f"  均值: {W_he.mean():.6f}")
print(f"  标准差: {W_he.std():.6f}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：矩阵运算
给定矩阵：
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
计算：
- A的转置
- A的逆矩阵
- A @ B（矩阵乘法）
- A * B（逐元素乘法）

练习2：特征值分解
给定矩阵：
M = [[4, 2], [1, 3]]
- 计算特征值和特征向量
- 验证 M = V @ diag(λ) @ V^{-1}

练习3：数据标准化
给定数据矩阵（3个样本，4个特征）：
X = [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]]
对每列进行标准化：(x - mean) / std
要求使用广播，不使用循环

练习4：随机数应用
生成1000个服从正态分布N(50, 10)的随机数：
- 计算均值和标准差
- 计算落在[40, 60]范围内的比例
- 绘制直方图（如果有matplotlib）

练习5：线性回归
给定数据：
X = [[1, 2], [2, 3], [3, 4], [4, 5]]  (4个样本，2个特征)
y = [5, 7, 9, 11]  (目标值)
使用最小二乘法求解权重w：
y = X @ w
提示：使用 np.linalg.lstsq 或正规方程 w = (X^T @ X)^{-1} @ X^T @ y
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1
print("\n练习1：矩阵运算")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"A:\n{A}")
print(f"A的转置:\n{A.T}")
print(f"A的逆矩阵:\n{np.linalg.inv(A)}")
print(f"A @ B:\n{A @ B}")
print(f"A * B:\n{A * B}")

# 练习2
print("\n练习2：特征值分解")
M = np.array([[4, 2], [1, 3]])
eigvals, eigvecs = np.linalg.eig(M)
print(f"特征值: {eigvals}")
print(f"特征向量:\n{eigvecs}")

# 验证
V = eigvecs
D = np.diag(eigvals)
V_inv = np.linalg.inv(V)
reconstructed = V @ D @ V_inv
print(f"验证 V @ D @ V_inv:\n{reconstructed}")
print(f"是否接近M: {np.allclose(M, reconstructed)}")

# 练习3
print("\n练习3：数据标准化")
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]], dtype=float)

mean = X.mean(axis=0)
std = X.std(axis=0)
X_normalized = (X - mean) / std

print(f"原始数据:\n{X}")
print(f"标准化后:\n{X_normalized}")
print(f"标准化后均值: {X_normalized.mean(axis=0)}")
print(f"标准化后标准差: {X_normalized.std(axis=0)}")

# 练习4
print("\n练习4：随机数统计")
np.random.seed(42)
data = np.random.normal(50, 10, 1000)
print(f"理论值: μ=50, σ=10")
print(f"实际均值: {data.mean():.2f}")
print(f"实际标准差: {data.std():.2f}")

in_range = ((data >= 40) & (data <= 60)).sum()
print(f"落在[40,60]范围内的比例: {in_range/len(data)*100:.1f}%")
print(f"理论比例 (约1σ): 68.3%")

# 练习5
print("\n练习5：线性回归")
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([5, 7, 9, 11])

# 添加偏置项（可选）
X_with_bias = np.column_stack([np.ones(len(X)), X])

# 方法1：lstsq
w, _, _, _ = np.linalg.lstsq(X_with_bias, y, rcond=None)
print(f"最小二乘解 (含偏置): {w}")
print(f"即: y = {w[0]:.2f} + {w[1]:.2f}*x1 + {w[2]:.2f}*x2")

# 方法2：正规方程（使用pinv处理可能的奇异矩阵）
try:
    w_normal = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    print(f"正规方程解: {w_normal}")
except np.linalg.LinAlgError:
    print("正规方程：矩阵奇异，使用pinv（伪逆）")
    w_normal = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    print(f"伪逆解: {w_normal}")

# 预测
y_pred = X_with_bias @ w
print(f"预测值: {y_pred}")
print(f"实际值: {y}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ 线性代数：矩阵乘法、逆矩阵、行列式、秩、特征值、SVD
✅ 方程求解：solve、lstsq（最小二乘）
✅ 随机数：各种分布、采样、打乱
✅ 广播：维度对齐、自动扩展
✅ 文件IO：save/load npy，savetxt/loadtxt，savez

下节课预告：Pandas数据处理
""")
