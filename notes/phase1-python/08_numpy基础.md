# NumPy第一课：数组基础

## 学习目标
- [ ] 理解NumPy数组与Python列表的区别
- [ ] 掌握数组的创建方法
- [ ] 学会数组的索引和切片
- [ ] 掌握数组的形状操作
- [ ] 理解广播机制

## 为什么用NumPy？

1. **速度快**：比Python列表快10-100倍
2. **内存省**：内存占用更少
3. **功能强**：向量化运算、广播、线性代数

## 创建数组

```python
import numpy as np

# 从列表创建
arr = np.array([1, 2, 3, 4, 5])

# 常用创建函数
np.zeros((3, 4))        # 3行4列的0
np.ones((2, 3))         # 2行3列的1
np.arange(10)           # 0-9
np.linspace(0, 1, 5)    # 0到1之间5个数
np.random.rand(3)       # 3个随机数
```

## 数组属性

| 属性 | 说明 | 示例 |
|------|------|------|
| `ndim` | 维度数 | `arr.ndim` → 2 |
| `shape` | 形状 | `arr.shape` → (3, 4) |
| `size` | 总元素数 | `arr.size` → 12 |
| `dtype` | 数据类型 | `arr.dtype` → int64 |

## 索引和切片

```python
# 一维
arr[0]        # 第一个元素
arr[2:5]      # 索引2到4
arr[::2]      # 每隔一个
arr[::-1]     # 反转

# 二维
arr[0, 0]     # 第0行第0列
arr[0, :]     # 第0行（整行）
arr[:, 0]     # 第0列（整列）
arr[0:2, 1:3] # 子数组

# 布尔索引
arr[arr > 5]  # 大于5的元素
```

## 形状操作

```python
arr.reshape(3, 4)     # 改变形状
arr.ravel()           # 展平
arr.T                 # 转置
arr[:, np.newaxis]    # 增加维度（列向量）
```

## 数组运算

```python
# 元素级运算
arr1 + arr2    # 对应元素相加
arr1 * arr2    # 对应元素相乘
arr ** 2       # 每个元素平方

# 矩阵乘法
arr1 @ arr2    # 矩阵乘法
arr1.dot(arr2) # 同上
```

## 广播机制

```python
# 标量广播
arr + 10       # 10广播到arr的每个元素

# 数组广播
arr_2d + arr_1d  # 1D数组广播到2D的每一行
```

## 统计函数

```python
arr.sum()          # 总和
arr.mean()         # 均值
arr.std()          # 标准差
arr.min()          # 最小值
arr.max()          # 最大值
arr.sum(axis=0)    # 按列求和
arr.mean(axis=1)   # 按行求平均
arr.argmax()       # 最大值索引
```

## 常用函数速查

| 函数 | 作用 |
|------|------|
| `np.eye(n)` | n×n单位矩阵 |
| `np.diag([1,2,3])` | 对角矩阵 |
| `np.vstack((a,b))` | 垂直拼接 |
| `np.hstack((a,b))` | 水平拼接 |
| `np.concatenate((a,b), axis=0)` | 指定轴拼接 |
| `np.split(arr, n, axis=0)` | 分割 |

## 遇到问题记录
（在这里记录学习过程中遇到的任何问题）

## 练习记录
- [ ] 完成 `01_numpy_basics.py` 中的5个练习
- [ ] 尝试创建不同形状的数组进行操作

## 下节课预告
- NumPy进阶：广播、线性代数、随机数
