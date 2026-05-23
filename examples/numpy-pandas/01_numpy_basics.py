"""
NumPy第一课：数组基础操作
学习目标：
1. 理解NumPy数组与Python列表的区别
2. 掌握数组的创建方法
3. 学会数组的索引和切片
4. 掌握数组的形状操作
5. 理解广播机制

NumPy是Python科学计算的基础，也是Pandas、PyTorch的基础！
"""

import numpy as np

# ==================== NumPy简介 ====================
print("=" * 60)
print("📊 NumPy简介")
print("=" * 60)

print("""
为什么用NumPy？
1. 速度快：NumPy数组比Python列表快10-100倍
2. 内存省：NumPy数组内存占用更少
3. 功能强：支持向量化运算、广播、线性代数等
""")

# 速度对比示例
import time

print("\n【速度对比】")
# Python列表
python_list = list(range(1000000))
start = time.time()
python_result = [x * 2 for x in python_list]
python_time = time.time() - start

# NumPy数组
numpy_array = np.array(range(1000000))
start = time.time()
numpy_result = numpy_array * 2
numpy_time = time.time() - start

print(f"Python列表用时: {python_time:.4f}秒")
print(f"NumPy数组用时: {numpy_time:.4f}秒")
print(f"NumPy快了近 {python_time/numpy_time:.0f} 倍！")


# ==================== 创建数组 ====================
print("\n" + "=" * 60)
print("📝 创建数组")
print("=" * 60)

# 从列表创建
print("\n【从列表创建】")
list_data = [1, 2, 3, 4, 5]
arr1 = np.array(list_data)
print(f"列表: {list_data}")
print(f"数组: {arr1}")
print(f"类型: {type(arr1)}")
print(f"数据类型: {arr1.dtype}")

# 创建二维数组
print("\n【二维数组】")
list_2d = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
arr_2d = np.array(list_2d)
print(f"二维数组:\n{arr_2d}")
print(f"形状: {arr_2d.shape}")  # (行数, 列数)
print(f"维度: {arr_2d.ndim}")  # 维数

# 指定数据类型
print("\n【指定数据类型】")
arr_float = np.array([1, 2, 3], dtype=np.float32)
arr_int = np.array([1.5, 2.7, 3.2], dtype=np.int32)
print(f"整数转浮点: {arr_float}, dtype={arr_float.dtype}")
print(f"浮点转整数: {arr_int}, dtype={arr_int.dtype}")  # 截断小数


# 常用创建函数
print("\n" + "=" * 60)
print("🔧 常用创建函数")
print("=" * 60)

# zeros, ones, empty
print("\n【zeros, ones, empty】")
zeros = np.zeros((3, 4))  # 3行4列的0数组
ones = np.ones((2, 3))    # 2行3列的1数组
empty = np.empty((2, 2))  # 2行2列的未初始化数组

print(f"zeros(3,4):\n{zeros}")
print(f"ones(2,3):\n{ones}")

# arange - 类似range
print("\n【arange】")
arr1 = np.arange(10)       # 0到9
arr2 = np.arange(1, 10, 2) # 1到9，步长2
print(f"arange(10): {arr1}")
print(f"arange(1,10,2): {arr2}")

# linspace - 等间隔数列
print("\n【linspace】")
arr3 = np.linspace(0, 1, 5)   # 0到1之间，5个数
arr4 = np.linspace(0, np.pi, 3)  # 0到π之间，3个数
print(f"linspace(0,1,5): {arr3}")
print(f"linspace(0,pi,3): {arr4}")

# random - 随机数
print("\n【random】")
np.random.seed(42)  # 设置随机种子，保证可重复
rand1 = np.random.rand(3)       # 0-1均匀分布
rand2 = np.random.randn(3)      # 标准正态分布
rand3 = np.random.randint(1, 10, 5)  # 1-9的随机整数，5个
print(f"rand(3): {rand1}")
print(f"randn(3): {rand2}")
print(f"randint(1,10,5): {rand3}")

# 特殊矩阵
print("\n【特殊矩阵】")
eye = np.eye(3)  # 3x3单位矩阵
diag = np.diag([1, 2, 3])  # 对角矩阵
print(f"单位矩阵:\n{eye}")
print(f"对角矩阵:\n{diag}")


# ==================== 数组属性 ====================
print("\n" + "=" * 60)
print("📏 数组属性")
print("=" * 60)

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"数组:\n{arr}")
print(f"ndim (维度数): {arr.ndim}")
print(f"shape (形状): {arr.shape}")
print(f"size (总元素数): {arr.size}")
print(f"dtype (数据类型): {arr.dtype}")
print(f"itemsize (每个元素字节): {arr.itemsize}")
print(f"nbytes (总字节): {arr.nbytes}")


# ==================== 索引和切片 ====================
print("\n" + "=" * 60)
print("🔍 索引和切片")
print("=" * 60)

arr = np.arange(10)
print(f"一维数组: {arr}")

# 一维索引
print("\n【一维索引】")
print(f"arr[0] = {arr[0]}")
print(f"arr[5] = {arr[5]}")
print(f"arr[-1] = {arr[-1]}")

# 一维切片 [start:end:step]
print("\n【一维切片】")
print(f"arr[2:5] = {arr[2:5]}")    # 索引2到4
print(f"arr[:4] = {arr[:4]}")      # 开始到3
print(f"arr[5:] = {arr[5:]}")      # 5到结束
print(f"arr[::2] = {arr[::2]}")    # 每隔一个
print(f"arr[::-1] = {arr[::-1]}")  # 反转

# 二维数组索引
print("\n【二维数组索引】")
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"二维数组:\n{arr2d}")

print(f"\narr2d[0, 0] = {arr2d[0, 0]}")    # 第0行第0列
print(f"arr2d[1, 2] = {arr2d[1, 2]}")    # 第1行第2列
print(f"arr2d[0] = {arr2d[0]}")          # 第0行（整行）
print(f"arr2d[:, 1] = {arr2d[:, 1]}")   # 第1列（整列）

# 二维切片
print("\n【二维切片】")
print(f"arr2d[0:2, 1:3]:\n{arr2d[0:2, 1:3]}")  # 0-1行，1-2列
print(f"arr2d[:, :2]:\n{arr2d[:, :2]}")         # 所有行，0-1列
print(f"arr2d[:2, :]:\n{arr2d[:2, :]}")         # 0-1行，所有列

# 布尔索引
print("\n【布尔索引】")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"原数组: {arr}")
print(f"大于5的元素: {arr[arr > 5]}")
print(f"偶数元素: {arr[arr % 2 == 0]}")
print(f"3到7之间的元素: {arr[(arr > 3) & (arr < 7)]}")


# ==================== 形状操作 ====================
print("\n" + "=" * 60)
print("🔧 形状操作")
print("=" * 60)

arr = np.arange(12)
print(f"原数组: {arr}, shape={arr.shape}")

# reshape
print("\n【reshape】")
arr_3x4 = arr.reshape(3, 4)
arr_2x6 = arr.reshape(2, 6)
arr_4x3 = arr.reshape(4, 3)

print(f"reshape(3,4):\n{arr_3x4}")
print(f"reshape(2,6):\n{arr_2x6}")
print(f"reshape(4,3):\n{arr_4x3}")

# -1自动计算
arr_2d = arr.reshape(3, -1)  # -1表示自动计算
print(f"reshape(3, -1): shape={arr_2d.shape}")

# ravel / flatten - 展平
print("\n【展平】")
arr_flat = arr_3x4.ravel()
print(f"ravel后: {arr_flat}")

# transpose - 转置
print("\n【转置】")
arr_t = arr_3x4.T
print(f"转置前:\n{arr_3x4}")
print(f"转置后:\n{arr_t}")

# newaxis - 增加维度
print("\n【增加维度】")
arr_1d = np.array([1, 2, 3])
arr_2d_col = arr_1d[:, np.newaxis]  # 变成列向量
arr_2d_row = arr_1d[np.newaxis, :]  # 变成行向量
print(f"一维: {arr_1d}, shape={arr_1d.shape}")
print(f"列向量:\n{arr_2d_col}, shape={arr_2d_col.shape}")
print(f"行向量: {arr_2d_row}, shape={arr_2d_row.shape}")


# ==================== 数组运算 ====================
print("\n" + "=" * 60)
print("➕ 数组运算")
print("=" * 60)

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([10, 20, 30, 40])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")

# 元素级运算
print("\n【元素级运算】")
print(f"arr1 + arr2 = {arr1 + arr2}")
print(f"arr1 - arr2 = {arr1 - arr2}")
print(f"arr1 * arr2 = {arr1 * arr2}")  # 逐元素相乘
print(f"arr1 / arr2 = {arr1 / arr2}")
print(f"arr1 ** 2 = {arr1 ** 2}")
print(f"arr1 + 10 = {arr1 + 10}")   # 广播

# 矩阵乘法
print("\n【矩阵乘法】")
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])
print(f"mat1:\n{mat1}")
print(f"mat2:\n{mat2}")
print(f"逐元素相乘 (mat1 * mat2):\n{mat1 * mat2}")
print(f"矩阵乘法 (mat1 @ mat2):\n{mat1 @ mat2}")
print(f"矩阵乘法 (mat1.dot(mat2)):\n{mat1.dot(mat2)}")

# 常用数学函数
print("\n【数学函数】")
arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"角度: {arr}")
print(f"sin: {np.sin(arr)}")
print(f"cos: {np.cos(arr)}")
print(f"exp: {np.exp([1, 2, 3])}")
print(f"log: {np.log([1, np.e, np.e**2])}")
print(f"sqrt: {np.sqrt([1, 4, 9, 16])}")


# ==================== 广播机制 ====================
print("\n" + "=" * 60)
print("📡 广播机制 (Broadcasting)")
print("=" * 60)

print("""
广播规则：
1. 维度从后向前对齐
2. 某个维度为1时，可以拉伸匹配
3. 维度不匹配且不为1时，报错
""")

# 示例1：标量广播
print("\n【示例1：标量广播】")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"数组:\n{arr}")
print(f"数组 + 10:\n{arr + 10}")  # 标量10广播到(2,3)

# 示例2：一维数组广播到二维
print("\n【示例2：行广播】")
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr_1d = np.array([10, 20, 30])
print(f"2D数组 shape={arr_2d.shape}:\n{arr_2d}")
print(f"1D数组 shape={arr_1d.shape}: {arr_1d}")
print(f"相加:\n{arr_2d + arr_1d}")  # [10,20,30]广播到每行

# 示例3：列广播
print("\n【示例3：列广播】")
arr_col = np.array([[1], [2], [3]])
print(f"列向量 shape={arr_col.shape}:\n{arr_col}")
print(f"2D数组 + 列向量:\n{arr_2d + arr_col}")


# ==================== 常用统计函数 ====================
print("\n" + "=" * 60)
print("📊 常用统计函数")
print("=" * 60)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"数组:\n{arr}")

print(f"\n【整体统计】")
print(f"总和: {arr.sum()}")
print(f"均值: {arr.mean()}")
print(f"标准差: {arr.std():.4f}")
print(f"方差: {arr.var():.4f}")
print(f"最小值: {arr.min()}")
print(f"最大值: {arr.max()}")

print(f"\n【按轴统计】")
print(f"按行求和 (axis=1): {arr.sum(axis=1)}")
print(f"按列求和 (axis=0): {arr.sum(axis=0)}")
print(f"每行均值 (axis=1): {arr.mean(axis=1)}")
print(f"每列最大值 (axis=0): {arr.max(axis=0)}")

# argmax, argmin
print(f"\n【最值索引】")
arr_1d = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"数组: {arr_1d}")
print(f"最大值索引: {arr_1d.argmax()}")
print(f"最小值索引: {arr_1d.argmin()}")


# ==================== 数组拼接和分割 ====================
print("\n" + "=" * 60)
print("🔗 数组拼接和分割")
print("=" * 60)

# 拼接
print("\n【拼接】")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
print(f"arr1:\n{arr1}")
print(f"arr2:\n{arr2}")
print(f"垂直拼接 vstack:\n{np.vstack((arr1, arr2))}")
print(f"水平拼接 hstack:\n{np.hstack((arr1, arr2))}")
print(f"concatenate axis=0:\n{np.concatenate((arr1, arr2), axis=0)}")
print(f"concatenate axis=1:\n{np.concatenate((arr1, arr2), axis=1)}")

# 分割
print("\n【分割】")
arr = np.arange(12).reshape(3, 4)
print(f"原数组:\n{arr}")
parts = np.split(arr, 2, axis=1)  # 按列分割成2份
print(f"按列分割:")
for i, p in enumerate(parts):
    print(f"  第{i+1}份:\n{p}")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：图像数据表示
print("\n【场景1：图像数据】")
# 模拟一张 4x4 的灰度图像
image = np.random.randint(0, 256, (4, 4))
print(f"图像数据 (4x4像素):\n{image}")
print(f"最亮像素值: {image.max()}")
print(f"最暗像素值: {image.min()}")
print(f"平均亮度: {image.mean():.1f}")

# 场景2：批量数据处理
print("\n【场景2：批量数据归一化】")
data = np.array([[100, 200, 300], [50, 150, 250], [25, 75, 125]], dtype=float)
print(f"原始数据:\n{data}")
# Min-Max归一化到0-1范围
data_norm = (data - data.min()) / (data.max() - data.min())
print(f"归一化后:\n{data_norm}")

# 场景3：数据筛选
print("\n【场景3：数据筛选】")
scores = np.array([85, 92, 78, 65, 95, 55, 88, 76])
print(f"所有成绩: {scores}")
print(f"及格成绩: {scores[scores >= 60]}")
print(f"优秀成绩(>=90): {scores[scores >= 90]}")

# 场景4：距离计算
print("\n【场景4：欧几里得距离】")
point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])
distance = np.sqrt(np.sum((point1 - point2)**2))
print(f"点A: {point1}")
print(f"点B: {point2}")
print(f"距离: {distance:.4f}")
# 或者用内置函数
print(f"距离(内置): {np.linalg.norm(point1 - point2):.4f}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：数组创建
创建一个 3x3 的数组，满足：
- 对角线元素为1
- 其余元素为0
（提示：使用np.eye或np.diag）

练习2：切片操作
给定数组 arr = np.arange(24).reshape(4, 6)
提取出：
- 第2行（索引1）
- 最后2列
- 左上角 2x2 的子数组

练习3：条件筛选
给定成绩数组 scores = np.array([65, 78, 82, 55, 91, 88, 76, 95, 60, 72])
- 统计及格人数
- 计算平均分
- 找出最高分和最低分的索引

练习4：矩阵运算
给定两个矩阵：
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
计算：
- 逐元素相乘
- 矩阵乘法
- A的转置乘以B

练习5：广播应用
给定一个形状为 (5, 3) 的数据矩阵，
对每一列进行标准化：(x - mean) / std
（提示：利用广播机制，不用循环）
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1
print("\n练习1：单位矩阵")
identity = np.eye(3)
print(f"np.eye(3):\n{identity}")

# 练习2
print("\n练习2：切片")
arr = np.arange(24).reshape(4, 6)
print(f"原数组:\n{arr}")
print(f"第2行: {arr[1]}")
print(f"最后2列:\n{arr[:, -2:]}")
print(f"左上2x2:\n{arr[:2, :2]}")

# 练习3
print("\n练习3：成绩统计")
scores = np.array([65, 78, 82, 55, 91, 88, 76, 95, 60, 72])
passed = np.sum(scores >= 60)
avg = scores.mean()
max_idx = scores.argmax()
min_idx = scores.argmin()
print(f"及格人数: {passed}")
print(f"平均分: {avg:.1f}")
print(f"最高分索引: {max_idx} (分数{scores[max_idx]})")
print(f"最低分索引: {min_idx} (分数{scores[min_idx]})")

# 练习4
print("\n练习4：矩阵运算")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"逐元素相乘 (A * B):\n{A * B}")
print(f"矩阵乘法 (A @ B):\n{A @ B}")
print(f"A.T @ B:\n{A.T @ B}")

# 练习5
print("\n练习5：列标准化")
data = np.random.randn(5, 3)  # 随机生成数据
print(f"原始数据 shape={data.shape}:")
means = data.mean(axis=0)
stds = data.std(axis=0)
normalized = (data - means) / stds
print(f"标准化后每列均值: {normalized.mean(axis=0)}")  # 接近0
print(f"标准化后每列标准差: {normalized.std(axis=0)}")  # 接近1


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ 数组创建：np.array(), zeros(), ones(), arange(), linspace(), random
✅ 数组属性：shape, ndim, size, dtype
✅ 索引切片：arr[i, j], arr[:, :], 布尔索引
✅ 形状操作：reshape(), ravel(), T, newaxis
✅ 数组运算：元素级运算 (+, -, *, /)，矩阵乘法 (@, dot)
✅ 广播机制：自动扩展维度进行运算
✅ 统计函数：sum, mean, std, max, min, argmax
✅ 拼接分割：vstack, hstack, concatenate, split

下节课预告：NumPy进阶 - 广播、线性代数、随机数
""")
