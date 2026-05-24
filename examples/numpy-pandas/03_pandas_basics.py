"""
Pandas第一课：基础操作
学习目标：
1. 理解Pandas的数据结构：Series和DataFrame
2. 掌握数据的创建和基本操作
3. 学会数据的索引和切片
4. 掌握数据的读取和保存

Pandas是Python数据分析的利器，是处理表格数据的首选工具！
"""

import pandas as pd
import numpy as np

# ==================== Pandas简介 ====================
print("=" * 60)
print("📊 Pandas简介")
print("=" * 60)

print("""
Pandas是什么？
- 基于NumPy构建的数据分析库
- 专门处理表格数据（类似Excel）
- 提供高性能、易用的数据结构和分析工具
- 是数据科学、机器学习的必备工具

核心数据结构：
1. Series：一维数组，带索引
2. DataFrame：二维表格，带行索引和列名
""")


# ==================== Series（一维数据）====================
print("\n" + "=" * 60)
print("📈 Series - 一维数组")
print("=" * 60)

# 从列表创建
print("\n【从列表创建】")
s1 = pd.Series([1, 2, 3, 4, 5])
print(f"s1:\n{s1}")
print(f"类型: {type(s1)}")

# 自定义索引
print("\n【自定义索引】")
s2 = pd.Series([85, 92, 78, 88], index=['小明', '小红', '小刚', '小李'])
print(f"s2:\n{s2}")

# 从字典创建
print("\n【从字典创建】")
s3 = pd.Series({'语文': 85, '数学': 92, '英语': 78})
print(f"s3:\n{s3}")

# 访问数据
print("\n【访问数据】")
print(f"s2['小红'] = {s2['小红']}")
print(f"s2.iloc[0] = {s2.iloc[0]}")  # 用iloc按位置访问
print(f"s2[['小明', '小红']] = \n{s2[['小明', '小红']]}")

# Series属性
print("\n【Series属性】")
print(f"值: {s2.values}")
print(f"索引: {s2.index}")
print(f"数据类型: {s2.dtype}")
print(f"长度: {len(s2)}")

# 条件筛选
print("\n【条件筛选】")
print(f"成绩大于85分的: \n{s2[s2 > 85]}")


# ==================== DataFrame（二维表格）====================
print("\n" + "=" * 60)
print("📋 DataFrame - 二维表格")
print("=" * 60)

# 从字典创建
print("\n【从字典创建】")
data = {
    '姓名': ['小明', '小红', '小刚', '小李'],
    '年龄': [20, 21, 19, 20],
    '数学': [85, 92, 78, 88],
    '英语': [88, 85, 92, 78]
}
df1 = pd.DataFrame(data)
print(f"df1:\n{df1}")

# 从二维数组创建
print("\n【从数组创建】")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df2 = pd.DataFrame(arr, columns=['A', 'B', 'C'], index=['x', 'y', 'z'])
print(f"df2:\n{df2}")

# 从Series字典创建
print("\n【从Series字典创建】")
s_name = pd.Series(['小明', '小红', '小刚'])
s_age = pd.Series([20, 21, 19])
s_score = pd.Series([85, 92, 78])
df3 = pd.DataFrame({'姓名': s_name, '年龄': s_age, '成绩': s_score})
print(f"df3:\n{df3}")


# ==================== DataFrame基本操作 ====================
print("\n" + "=" * 60)
print("🔧 DataFrame基本操作")
print("=" * 60)

# 查看数据
print("\n【查看数据】")
print(f"df1:\n{df1}")
print(f"\n前2行:\n{df1.head(2)}")
print(f"\n后2行:\n{df1.tail(2)}")
print(f"\n数据形状: {df1.shape}")
print(f"\n列名: {df1.columns.tolist()}")
print(f"\n行索引: {df1.index.tolist()}")

# 基本信息
print("\n【基本信息】")
print(f"数据类型:\n{df1.dtypes}")
print(f"\n基本信息:")
print(df1.info())
print(f"\n统计信息:")
print(df1.describe())


# ==================== 数据选择 ====================
print("\n" + "=" * 60)
print("🔍 数据选择")
print("=" * 60)

# 选择列
print("\n【选择列】")
print(f"选择单列:\n{df1['姓名']}")
print(f"\n选择多列:\n{df1[['姓名', '数学']]}")

# 选择行（按位置）
print("\n【按位置选择行 - iloc】")
print(f"第1行:\n{df1.iloc[0]}")
print(f"\n前2行:\n{df1.iloc[0:2]}")
print(f"\n第1,3行:\n{df1.iloc[[0, 2]]}")

# 选择行（按标签）
print("\n【按标签选择行 - loc】")
df1_indexed = df1.set_index('姓名')
print(f"设置索引后:\n{df1_indexed}")
print(f"\n选择'小明':\n{df1_indexed.loc['小明']}")
print(f"\n选择'小明'到'小刚':\n{df1_indexed.loc['小明':'小刚']}")

# 同时选择行和列
print("\n【同时选择行列】")
print(f"df1.loc[0:2, ['姓名', '数学']]:")
print(df1.loc[0:2, ['姓名', '数学']])

print(f"\ndf1.iloc[0:2, 0:3]:")
print(df1.iloc[0:2, 0:3])


# ==================== 条件筛选 ====================
print("\n" + "=" * 60)
print("🎯 条件筛选")
print("=" * 60)

# 单条件筛选
print("\n【单条件】")
print(f"数学成绩大于85分:\n{df1[df1['数学'] > 85]}")

# 多条件筛选
print("\n【多条件】")
condition1 = df1['数学'] >= 85
condition2 = df1['英语'] >= 85
print(f"数学和英语都>=85:\n{df1[condition1 & condition2]}")

print(f"\n数学或英语>=90:\n{df1[(df1['数学'] >= 90) | (df1['英语'] >= 90)]}")

# 多条件组合
print("\n【复杂条件】")
condition = (df1['数学'] >= 80) & (df1['英语'] >= 80) & (df1['年龄'] >= 20)
print(f"双科>=80且年龄>=20:\n{df1[condition]}")


# ==================== 数据修改 ====================
print("\n" + "=" * 60)
print("✏️ 数据修改")
print("=" * 60)

# 创建副本进行修改
df_edit = df1.copy()

# 修改列
print("\n【修改列】")
df_edit['数学'] = df_edit['数学'] + 5
print(f"数学成绩加5分:\n{df_edit}")

# 添加新列
print("\n【添加新列】")
df_edit['总分'] = df_edit['数学'] + df_edit['英语']
df_edit['平均分'] = (df_edit['数学'] + df_edit['英语']) / 2
print(f"添加总分和平均分:\n{df_edit}")

# 删除列
print("\n【删除列】")
df_edit = df_edit.drop(['总分', '平均分'], axis=1)
print(f"删除总分和平均分:\n{df_edit}")

# 修改特定值
print("\n【修改特定值】")
df_edit.loc[0, '数学'] = 95
print(f"修改小明的数学成绩为95:\n{df_edit}")


# ==================== 数据排序 ====================
print("\n" + "=" * 60)
print("📊 数据排序")
print("=" * 60)

# 按单列排序
print("\n【按单列排序】")
print(f"按数学成绩排序:\n{df1.sort_values('数学', ascending=False)}")

# 按多列排序
print("\n【按多列排序】")
print(f"先按年龄，再按数学:\n{df1.sort_values(['年龄', '数学'], ascending=[True, False])}")

# 按索引排序
print("\n【按索引排序】")
df_shuffled = df1.sample(frac=1)  # 打乱顺序
print(f"打乱后:\n{df_shuffled}")
print(f"按索引排序:\n{df_shuffled.sort_index()}")


# ==================== 数据读取和保存 ====================
print("\n" + "=" * 60)
print("💾 数据读取和保存")
print("=" * 60)

# 保存为CSV
print("\n【保存为CSV】")
df1.to_csv('temp_data.csv', index=False, encoding='utf-8')
print("已保存到 temp_data.csv")

# 从CSV读取
df_loaded = pd.read_csv('temp_data.csv')
print(f"从CSV加载:\n{df_loaded}")

# 保存为Excel（需要安装openpyxl）
# df1.to_excel('temp_data.xlsx', index=False)
# df_excel = pd.read_excel('temp_data.xlsx')

# 保存为JSON
df1.to_json('temp_data.json', orient='records', force_ascii=False)
print("\n已保存到 temp_data.json")

df_json = pd.read_json('temp_data.json')
print(f"从JSON加载:\n{df_json}")

# 清理临时文件
import os
os.remove('temp_data.csv')
os.remove('temp_data.json')
print("\n已清理临时文件")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：成绩分析
print("\n【场景1：成绩分析】")
students = pd.DataFrame({
    '姓名': ['小明', '小红', '小刚', '小李', '小王', '小张'],
    '班级': ['A', 'A', 'B', 'B', 'A', 'B'],
    '语文': [85, 92, 78, 88, 95, 82],
    '数学': [88, 85, 92, 78, 90, 85],
    '英语': [82, 88, 85, 92, 85, 90]
})

print(f"学生成绩表:\n{students}")

# 计算各科统计
cols = ['语文', '数学', '英语']
stats = students[cols].agg(['mean', 'max', 'min', 'std'])
print(f"\n各科统计:\n{stats}")

# 计算总分和排名
students['总分'] = students[cols].sum(axis=1)
students['排名'] = students['总分'].rank(ascending=False)
students_sorted = students.sort_values('总分', ascending=False)
print(f"\n按总分排序:\n{students_sorted}")

# 班级统计
print(f"\n各班平均分:\n{students.groupby('班级')[cols].mean()}")


# 场景2：销售数据分析
print("\n【场景2：销售数据分析】")
sales = pd.DataFrame({
    '日期': pd.date_range('2024-01-01', periods=10, freq='D'),
    '产品': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'C'],
    '销量': [100, 150, 120, 80, 200, 110, 90, 180, 130, 100],
    '单价': [10, 15, 10, 20, 15, 10, 20, 15, 10, 20]
})
sales['销售额'] = sales['销量'] * sales['单价']

print(f"销售数据:\n{sales}")

# 产品汇总
print(f"\n产品汇总:\n{sales.groupby('产品').agg({
    '销量': 'sum',
    '销售额': 'sum',
    '日期': 'count'
}).rename(columns={'日期': '销售天数'})}")

# 按日期统计
print(f"\n每日销售:\n{sales.groupby('日期')['销售额'].sum()}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：创建DataFrame
创建一个包含以下信息的学生DataFrame：
- 姓名：张三、李四、王五、赵六
- 年龄：20、21、19、20
- 性别：男、女、男、女
- 成绩：85、92、78、88

练习2：数据筛选
在上面的DataFrame中：
- 筛选出所有男生
- 筛选出成绩大于80分的学生
- 筛选出成绩大于85分的女生

练习3：添加新列
在上面DataFrame中添加：
- 等级列：成绩>=90为"优秀"，>=80为"良好"，否则为"及格"
- 是否成年列：年龄>=20为True，否则为False

练习4：排序
- 按成绩从高到低排序
- 先按性别，再按年龄排序

练习5：数据分析
给定以下数据，计算：
- 每个人的总分和平均分
- 各科目平均分
- 找出总分最高的学生
data = {
    '姓名': ['张三', '李四', '王五'],
    '语文': [85, 92, 78],
    '数学': [88, 85, 92],
    '英语': [82, 88, 85]
}
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1
print("\n练习1：创建DataFrame")
df_ex1 = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [20, 21, 19, 20],
    '性别': ['男', '女', '男', '女'],
    '成绩': [85, 92, 78, 88]
})
print(df_ex1)

# 练习2
print("\n练习2：数据筛选")
print("男生:\n", df_ex1[df_ex1['性别'] == '男'])
print("\n成绩>80:\n", df_ex1[df_ex1['成绩'] > 80])
print("\n成绩>85的女生:\n", df_ex1[(df_ex1['性别'] == '女') & (df_ex1['成绩'] > 85)])

# 练习3
print("\n练习3：添加新列")
df_ex1['等级'] = df_ex1['成绩'].apply(lambda x: '优秀' if x >= 90 else '良好' if x >= 80 else '及格')
df_ex1['是否成年'] = df_ex1['年龄'] >= 20
print(df_ex1)

# 练习4
print("\n练习4：排序")
print("按成绩排序:\n", df_ex1.sort_values('成绩', ascending=False))
print("\n先按性别，再按年龄:\n", df_ex1.sort_values(['性别', '年龄']))

# 练习5
print("\n练习5：数据分析")
data_ex5 = {
    '姓名': ['张三', '李四', '王五'],
    '语文': [85, 92, 78],
    '数学': [88, 85, 92],
    '英语': [82, 88, 85]
}
df_ex5 = pd.DataFrame(data_ex5)
df_ex5['总分'] = df_ex5[['语文', '数学', '英语']].sum(axis=1)
df_ex5['平均分'] = df_ex5[['语文', '数学', '英语']].mean(axis=1)
print(df_ex5)

print(f"\n各科平均分:\n{df_ex5[['语文', '数学', '英语']].mean()}")
print(f"\n总分最高: {df_ex5.loc[df_ex5['总分'].idxmax(), '姓名']}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ Series：一维数据，带索引
✅ DataFrame：二维表格，带行列索引
✅ 数据创建：从列表、字典、数组创建
✅ 数据选择：loc（标签）、iloc（位置）
✅ 条件筛选：单条件、多条件、复杂条件
✅ 数据修改：修改值、添加列、删除列
✅ 数据排序：按值、按索引、多列排序
✅ 读写数据：CSV、JSON、Excel

下节课预告：Pandas进阶 - 数据清洗和处理
""")
