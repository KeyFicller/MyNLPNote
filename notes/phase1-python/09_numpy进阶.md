# NumPy第二课：进阶应用

## 学习目标
- [ ] 掌握NumPy线性代数运算
- [ ] 学会高级随机数生成
- [ ] 深入理解广播机制
- [ ] 学会数组的保存和加载

## 线性代数

```python
import numpy as np

# 矩阵乘法
C = A @ B

# 逆矩阵
A_inv = np.linalg.inv(A)

# 行列式
det = np.linalg.det(A)

# 矩阵的秩
rank = np.linalg.matrix_rank(A)

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(A)

# SVD分解
U, S, Vt = np.linalg.svd(A)
```

## 方程组求解

```python
# 求解 Ax = b
x = np.linalg.solve(A, b)

# 最小二乘解
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

## 随机数进阶

```python
# 设置随机种子
np.random.seed(42)

# 各种分布
np.random.uniform(0, 10, 5)      # 均匀分布
np.random.normal(0, 1, 5)         # 正态分布
np.random.randint(1, 100, 5)      # 整数随机数
np.random.poisson(5, 5)           # 泊松分布
np.random.exponential(1, 5)       # 指数分布

# 随机采样
np.random.shuffle(arr)             # 打乱数组
np.random.choice(arr, 5)          # 随机选择
np.random.choice(arr, 5, replace=False)  # 无放回
```

## 广播机制深入

```python
# 广播规则
# 1. 维度从后向前对齐
# 2. 维度相等或其中一个为1时，可以广播

# 标量广播
arr + 10

# 不同形状广播
arr_2d + arr_1d  # (3,3) + (3,) = (3,3)

# 添加维度
d[:, np.newaxis]  # 列向量
```

## 文件IO

```python
# 保存单个数组
np.save("file.npy", arr)
arr = np.load("file.npy")

# 保存文本
np.savetxt("file.txt", arr, fmt="%d", delimiter=",")
arr = np.loadtxt("file.txt", delimiter=",", dtype=int)

# 保存多个数组
np.savez("file.npz", first=arr1, second=arr2)
data = np.load("file.npz")
arr1 = data["first"]
```

## 实际应用场景

| 场景 | 代码示例 |
|------|----------|
| PCA降维 | SVD分解 |
| 数据标准化 | `(data - mean) / std`（广播） |
| 批量距离计算 | 利用广播 `(queries[:, None] - points)` |
| 权重初始化 | `np.random.uniform()`, `np.random.normal()` |

## 遇到问题记录
（在这里记录学习过程中遇到的任何问题）

## 练习记录
- [ ] 完成 `02_numpy_advanced.py` 中的5个练习
- [ ] 尝试用NumPy实现简单的线性回归

## 下节课预告
- Pandas数据处理
