"""
Pandas第二课：数据清洗与处理（进阶）
学习目标：
1. 掌握缺失值的处理方法
2. 学会处理重复数据
3. 掌握数据类型转换
4. 学会数据合并与连接
5. 掌握数据透视表和分组聚合

数据清洗占数据分析项目的70%时间，是最重要的实战技能！
"""

import pandas as pd
import numpy as np

# ==================== 缺失值处理 ====================
print("=" * 60)
print("🔍 缺失值处理")
print("=" * 60)

# 创建带有缺失值的数据
print("\n【创建示例数据】")
df_missing = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '孙七'],
    '年龄': [25, np.nan, 30, 35, np.nan],
    '成绩': [85, 90, np.nan, 78, 92],
    '班级': ['A', 'A', 'B', 'B', 'A']
})
print(f"原始数据:\n{df_missing}")

# 检查缺失值
print("\n【检查缺失值】")
print(f"缺失值统计:\n{df_missing.isnull().sum()}")
print(f"\n缺失值位置:\n{df_missing.isnull()}")
print(f"\n哪些行有缺失值:\n{df_missing[df_missing.isnull().any(axis=1)]}")

# 删除缺失值
print("\n【删除缺失值】")
print(f"删除包含缺失值的行:\n{df_missing.dropna()}")
print(f"\n删除有缺失值的列:\n{df_missing.dropna(axis=1)}")
print(f"\n删除所有值都缺失的行:\n{df_missing.dropna(how='all')}")
print(f"\n删除至少有2个缺失值的行:\n{df_missing.dropna(thresh=3)}")

# 填充缺失值
print("\n【填充缺失值】")
df_filled = df_missing.copy()

# 用固定值填充
df_filled['年龄'] = df_filled['年龄'].fillna(0)
print(f"用0填充年龄:\n{df_filled}")

# 用均值填充
df_filled2 = df_missing.copy()
df_filled2['成绩'] = df_filled2['成绩'].fillna(df_filled2['成绩'].mean())
print(f"\n用均值填充成绩:\n{df_filled2}")

# 用前后值填充
df_filled3 = df_missing.copy()
df_filled3['年龄'] = df_filled3['年龄'].ffill()  # 前向填充
print(f"\n前向填充年龄:\n{df_filled3}")

# 按组填充
print("\n【按组填充】")
df_group_filled = df_missing.copy()
# 按班级分组，用组内均值填充
for col in ['年龄', '成绩']:
    df_group_filled[col] = df_group_filled.groupby('班级')[col].transform(
        lambda x: x.fillna(x.mean())
    )
print(f"按班级均值填充:\n{df_group_filled}")


# ==================== 重复值处理 ====================
print("\n" + "=" * 60)
print("🔄 重复值处理")
print("=" * 60)

# 创建有重复值的数据
df_dup = pd.DataFrame({
    '姓名': ['张三', '李四', '张三', '王五', '李四', '赵六'],
    '年龄': [25, 30, 25, 35, 30, 40],
    '成绩': [85, 90, 85, 78, 90, 88]
})
print(f"原始数据:\n{df_dup}")

# 检查重复值
print("\n【检查重复值】")
print(f"重复行:\n{df_dup[df_dup.duplicated()]}")
print(f"\n重复次数:\n{df_dup.duplicated().sum()}")

# 基于特定列检查重复
print(f"\n基于姓名检查重复:\n{df_dup[df_dup.duplicated(subset=['姓名'])]}")

# 删除重复值
print("\n【删除重复值】")
print(f"删除所有重复行（保留第一个）:\n{df_dup.drop_duplicates()}")
print(f"\n删除所有重复行（保留最后一个）:\n{df_dup.drop_duplicates(keep='last')}")
print(f"\n删除所有重复行（不保留任何）:\n{df_dup.drop_duplicates(keep=False)}")

# 基于特定列删除重复
print(f"\n基于姓名删除重复:\n{df_dup.drop_duplicates(subset=['姓名'])}")


# ==================== 数据类型转换 ====================
print("\n" + "=" * 60)
print("🔄 数据类型转换")
print("=" * 60)

df_types = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '年龄': ['25', '30', '35'],  # 字符串
    '成绩': [85.5, 90.2, 78.0],
    '日期': ['2024-01-01', '2024-01-02', '2024-01-03'],
    '是否及格': ['是', '否', '是']
})
print(f"原始数据:\n{df_types}")
print(f"\n数据类型:\n{df_types.dtypes}")

# 转换数据类型
print("\n【转换数据类型】")
df_types['年龄'] = df_types['年龄'].astype(int)
df_types['日期'] = pd.to_datetime(df_types['日期'])
df_types['是否及格'] = df_types['是否及格'].map({'是': True, '否': False})

print(f"转换后:\n{df_types}")
print(f"\n转换后数据类型:\n{df_types.dtypes}")

# 提取日期信息
print(f"\n日期信息:")
print(f"年份: {df_types['日期'].dt.year.tolist()}")
print(f"月份: {df_types['日期'].dt.month.tolist()}")
print(f"星期: {df_types['日期'].dt.dayofweek.tolist()}")


# ==================== 数据合并 ====================
print("\n" + "=" * 60)
print("🔗 数据合并")
print("=" * 60)

# 创建示例数据
df1 = pd.DataFrame({
    '学号': ['001', '002', '003', '004'],
    '姓名': ['张三', '李四', '王五', '赵六'],
    '班级': ['A', 'A', 'B', 'B']
})

df2 = pd.DataFrame({
    '学号': ['001', '002', '003', '005'],
    '数学': [85, 90, 78, 92],
    '英语': [88, 85, 92, 90]
})

print(f"学生信息:\n{df1}")
print(f"\n成绩信息:\n{df2}")

# merge连接（类似SQL的JOIN）
print("\n【merge连接】")
print(f"内连接 (inner join):\n{pd.merge(df1, df2, on='学号', how='inner')}")
print(f"\n左连接 (left join):\n{pd.merge(df1, df2, on='学号', how='left')}")
print(f"\n右连接 (right join):\n{pd.merge(df1, df2, on='学号', how='right')}")
print(f"\n外连接 (outer join):\n{pd.merge(df1, df2, on='学号', how='outer')}")

# 多键连接
df3 = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '班级': ['A', 'A', 'B'],
    '语文': [80, 85, 90]
})
print(f"\n多键连接:\n{pd.merge(df1, df3, on=['姓名', '班级'])}")

# concat拼接
print("\n【concat拼接】")
df_a = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df_b = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df_c = pd.DataFrame({'C': [9, 10]})

print(f"纵向拼接 (axis=0):\n{pd.concat([df_a, df_b], axis=0, ignore_index=True)}")
print(f"\n横向拼接 (axis=1):\n{pd.concat([df_a, df_c], axis=1)}")

# join（基于索引的连接）
print("\n【join连接】")
df_left = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df_right = pd.DataFrame({'B': [4, 5, 6]}, index=['a', 'b', 'd'])
print(f"基于索引join:\n{df_left.join(df_right, how='outer')}")


# ==================== 分组聚合 ====================
print("\n" + "=" * 60)
print("📊 分组聚合")
print("=" * 60)

df_sales = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=10),
    '产品': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B', 'A', 'C'],
    '地区': ['北', '北', '南', '南', '北', '北', '南', '南', '南', '北'],
    '销量': [100, 150, 120, 200, 130, 80, 90, 180, 110, 95],
    '单价': [10, 15, 10, 15, 10, 20, 20, 15, 10, 20]
})
df_sales['销售额'] = df_sales['销量'] * df_sales['单价']

print(f"销售数据:\n{df_sales}")

# 单列分组
print("\n【单列分组】")
print(f"按产品分组统计销量:\n{df_sales.groupby('产品')['销量'].sum()}")
print(f"\n按产品分组统计销售额:\n{df_sales.groupby('产品')['销售额'].sum()}")

# 多列分组
print("\n【多列分组】")
print(f"按产品和地区分组:\n{df_sales.groupby(['产品', '地区'])['销售额'].sum()}")

# 多统计量
print("\n【多统计量】")
print(f"按产品分组多统计:\n{df_sales.groupby('产品')['销售额'].agg(['sum', 'mean', 'max', 'min', 'count'])}")

# 自定义聚合
print("\n【自定义聚合】")
def profit_margin(x):
    return (x.sum() * 0.3)  # 假设30%利润率

result = df_sales.groupby('产品').agg({
    '销量': 'sum',
    '销售额': ['sum', 'mean', profit_margin]
})
print(f"自定义聚合:\n{result}")

# transform（保持原索引）
print("\n【transform】")
df_sales['产品平均销量'] = df_sales.groupby('产品')['销量'].transform('mean')
df_sales['与平均差'] = df_sales['销量'] - df_sales['产品平均销量']
print(f"与产品平均销量比较:\n{df_sales[['产品', '销量', '产品平均销量', '与平均差']]}")

# apply（自定义函数）
print("\n【apply】")
def top_2_sales(group):
    return group.nlargest(2, '销售额')

print(f"每个产品销售额最高的2条:\n{df_sales.groupby('产品', group_keys=False).apply(top_2_sales)}")


# ==================== 数据透视表 ====================
print("\n" + "=" * 60)
print("📊 数据透视表")
print("=" * 60)

print(f"销售数据:\n{df_sales[['产品', '地区', '销量', '销售额']]}")

# pivot_table
print("\n【数据透视表】")
pivot1 = df_sales.pivot_table(
    values='销售额',
    index='产品',
    columns='地区',
    aggfunc='sum',
    fill_value=0
)
print(f"产品-地区销售额透视表:\n{pivot1}")

# 多统计量透视表
print("\n【多统计量透视表】")
pivot2 = df_sales.pivot_table(
    values=['销量', '销售额'],
    index='产品',
    aggfunc={'销量': 'sum', '销售额': ['sum', 'mean']}
)
print(f"多统计量透视表:\n{pivot2}")

# 多层索引透视表
print("\n【多层索引透视表】")
df_sales['月份'] = df_sales['日期'].dt.month
pivot3 = df_sales.pivot_table(
    values='销售额',
    index=['月份', '产品'],
    columns='地区',
    aggfunc='sum',
    fill_value=0
)
print(f"月份-产品-地区透视表:\n{pivot3}")

# melt（透视表逆操作）
print("\n【melt宽转长】")
df_wide = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '语文': [85, 90, 78],
    '数学': [88, 85, 92],
    '英语': [82, 88, 85]
})
print(f"宽格式:\n{df_wide}")
df_long = df_wide.melt(id_vars=['姓名'], var_name='科目', value_name='成绩')
print(f"\n长格式:\n{df_long}")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：数据清洗流程
print("\n【场景1：完整数据清洗流程】")

# 原始脏数据
df_raw = pd.DataFrame({
    '姓名': ['张三', '李四', ' 王五 ', '张三', '赵六', '孙七'],
    '年龄': ['25', '30', None, '25', '35', 'abc'],
    '成绩': ['85.5', '90', '78.5', '85.5', None, '88'],
    '日期': ['2024-01-01', '2024/01/02', '2024-01-03', '2024-01-01', None, '2024-01-06'],
    '班级': ['A', 'A', 'B', 'a', 'B', 'A']
})
print(f"原始脏数据:\n{df_raw}")

# 清洗步骤
df_clean = df_raw.copy()

# 1. 去除空格
df_clean['姓名'] = df_clean['姓名'].str.strip()

# 2. 处理年龄（转数字，错误值设为NaN）
df_clean['年龄'] = pd.to_numeric(df_clean['年龄'], errors='coerce')

# 3. 处理成绩
df_clean['成绩'] = pd.to_numeric(df_clean['成绩'], errors='coerce')

# 4. 处理日期
df_clean['日期'] = pd.to_datetime(df_clean['日期'], errors='coerce')

# 5. 统一班级格式
df_clean['班级'] = df_clean['班级'].str.upper()

# 6. 删除重复值（基于姓名和班级）
df_clean = df_clean.drop_duplicates(subset=['姓名', '班级'], keep='first')

# 7. 填充缺失值
df_clean['年龄'] = df_clean['年龄'].fillna(df_clean['年龄'].median())
df_clean['成绩'] = df_clean['成绩'].fillna(df_clean['成绩'].mean())
df_clean['日期'] = df_clean['日期'].fillna(pd.Timestamp.now())

print(f"\n清洗后数据:\n{df_clean}")
print(f"\n清洗后数据类型:\n{df_clean.dtypes}")


# 场景2：销售数据分析
print("\n【场景2：销售数据分析】")

# 生成更多销售数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
products = ['A', 'B', 'C', 'D']
regions = ['北京', '上海', '广州', '深圳']

df_sales_large = pd.DataFrame({
    '日期': np.random.choice(dates, 100),
    '产品': np.random.choice(products, 100),
    '地区': np.random.choice(regions, 100),
    '销量': np.random.randint(50, 200, 100),
    '单价': np.random.choice([10, 15, 20, 25], 100)
})
df_sales_large['销售额'] = df_sales_large['销量'] * df_sales_large['单价']
df_sales_large = df_sales_large.sort_values('日期').reset_index(drop=True)

print(f"销售数据样例:\n{df_sales_large.head(10)}")

# 按月份统计
print(f"\n月度销售额统计:")
df_sales_large['月份'] = df_sales_large['日期'].dt.to_period('M')
monthly_sales = df_sales_large.groupby('月份')['销售额'].sum()
print(monthly_sales)

# 产品排行榜
print(f"\n产品销售排行:")
product_rank = df_sales_large.groupby('产品')['销售额'].sum().sort_values(ascending=False)
print(product_rank)

# 地区销售占比
print(f"\n地区销售占比:")
region_pct = df_sales_large.groupby('地区')['销售额'].sum()
region_pct = (region_pct / region_pct.sum() * 100).round(2)
print(region_pct)


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：缺失值处理
给定以下DataFrame：
df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, None, 30, None],
    '成绩': [85, 90, None, 78],
    '班级': ['A', 'A', 'B', 'B']
})
- 检查每列的缺失值数量
- 用班级平均年龄填充年龄缺失值
- 用全局平均成绩填充成绩缺失值

练习2：重复值处理
给定以下DataFrame：
df = pd.DataFrame({
    '姓名': ['张三', '李四', '张三', '王五', '李四'],
    '班级': ['A', 'A', 'A', 'B', 'A'],
    '成绩': [85, 90, 85, 78, 88]
})
- 找出基于姓名和班级的重复行
- 删除重复行，保留成绩最高的记录

练习3：数据合并
给定两个DataFrame：
df_students = pd.DataFrame({
    '学号': ['S001', 'S002', 'S003'],
    '姓名': ['张三', '李四', '王五']
})
df_scores = pd.DataFrame({
    '学号': ['S001', 'S002', 'S004'],
    '数学': [85, 90, 78],
    '英语': [88, 85, 92]
})
- 用内连接合并两个表
- 用左连接合并，保留所有学生信息

练习4：分组聚合
给定以下销售数据：
df = pd.DataFrame({
    '产品': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B'],
    '地区': ['北', '北', '南', '南', '北', '南', '北', '南'],
    '销量': [100, 150, 120, 200, 130, 80, 90, 180],
    '单价': [10, 15, 10, 15, 10, 20, 20, 15]
})
df['销售额'] = df['销量'] * df['单价']
- 计算每个产品的总销售额
- 计算每个地区每个产品的平均销量
- 用透视表展示产品-地区的销售额

练习5：数据清洗流程
给定以下脏数据：
df = pd.DataFrame({
    '姓名': [' 张三 ', '李四', '王五 ', '张三'],
    '年龄': ['25', '30', None, '25'],
    '成绩': ['85.5', '90', '78.5', '85.5'],
    '日期': ['2024-01-01', '2024/01/02', '2024-01-03', '2024-01-01']
})
编写完整的数据清洗流程，包括：
- 去除姓名空格
- 转换年龄为数值型，填充缺失值
- 转换成绩为数值型
- 统一日期格式
- 删除重复姓名
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1
print("\n练习1：缺失值处理")
df1 = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, None, 30, None],
    '成绩': [85, 90, None, 78],
    '班级': ['A', 'A', 'B', 'B']
})
print(f"缺失值数量:\n{df1.isnull().sum()}")
df1['年龄'] = df1.groupby('班级')['年龄'].transform(lambda x: x.fillna(x.mean()))
df1['成绩'] = df1['成绩'].fillna(df1['成绩'].mean())
print(f"处理后:\n{df1}")

# 练习2
print("\n练习2：重复值处理")
df2 = pd.DataFrame({
    '姓名': ['张三', '李四', '张三', '王五', '李四'],
    '班级': ['A', 'A', 'A', 'B', 'A'],
    '成绩': [85, 90, 85, 78, 88]
})
print(f"重复行:\n{df2[df2.duplicated(subset=['姓名', '班级'], keep=False)]}")
df2_clean = df2.sort_values('成绩', ascending=False).drop_duplicates(subset=['姓名', '班级'], keep='first')
print(f"保留最高成绩后:\n{df2_clean}")

# 练习3
print("\n练习3：数据合并")
df_students = pd.DataFrame({
    '学号': ['S001', 'S002', 'S003'],
    '姓名': ['张三', '李四', '王五']
})
df_scores = pd.DataFrame({
    '学号': ['S001', 'S002', 'S004'],
    '数学': [85, 90, 78],
    '英语': [88, 85, 92]
})
print(f"内连接:\n{pd.merge(df_students, df_scores, on='学号', how='inner')}")
print(f"\n左连接:\n{pd.merge(df_students, df_scores, on='学号', how='left')}")

# 练习4
print("\n练习4：分组聚合")
df4 = pd.DataFrame({
    '产品': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'B'],
    '地区': ['北', '北', '南', '南', '北', '南', '北', '南'],
    '销量': [100, 150, 120, 200, 130, 80, 90, 180],
    '单价': [10, 15, 10, 15, 10, 20, 20, 15]
})
df4['销售额'] = df4['销量'] * df4['单价']
print(f"各产品总销售额:\n{df4.groupby('产品')['销售额'].sum()}")
print(f"\n各地区各产品平均销量:\n{df4.groupby(['地区', '产品'])['销量'].mean()}")
print(f"\n产品-地区销售额透视表:\n{df4.pivot_table(values='销售额', index='产品', columns='地区', aggfunc='sum', fill_value=0)}")

# 练习5
print("\n练习5：数据清洗流程")
df5 = pd.DataFrame({
    '姓名': [' 张三 ', '李四', '王五 ', '张三'],
    '年龄': ['25', '30', None, '25'],
    '成绩': ['85.5', '90', '78.5', '85.5'],
    '日期': ['2024-01-01', '2024/01/02', '2024-01-03', '2024-01-01']
})
df5['姓名'] = df5['姓名'].str.strip()
df5['年龄'] = pd.to_numeric(df5['年龄'], errors='coerce')
df5['年龄'] = df5['年龄'].fillna(df5['年龄'].median())
df5['成绩'] = pd.to_numeric(df5['成绩'])
df5['日期'] = pd.to_datetime(df5['日期'], format='mixed')
df5 = df5.drop_duplicates(subset=['姓名'], keep='first')
print(f"清洗后:\n{df5}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ 缺失值处理：isnull检查、dropna删除、fillna填充（固定值/统计值/前后值）
✅ 重复值处理：duplicated检查、drop_duplicates删除
✅ 数据类型转换：astype、to_numeric、to_datetime
✅ 数据合并：merge（SQL风格）、concat（拼接）、join（索引）
✅ 分组聚合：groupby + agg/transform/apply
✅ 数据透视表：pivot_table、melt（宽转长）
✅ 实际应用：完整数据清洗流程

恭喜！你已经掌握了数据分析的核心技能！

【学习建议】
1. 在实际项目中练习这些操作
2. 数据清洗没有固定套路，需要根据数据特点灵活处理
3. 记住：数据分析80%时间花在清洗上！

【下一阶段】
可以考虑学习：
- Matplotlib/Seaborn 数据可视化
- Scikit-learn 机器学习入门
- 或者直接开始 PyTorch 深度学习！
""")
