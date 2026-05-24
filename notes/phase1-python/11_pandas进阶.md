# Pandas第二课：数据清洗与处理

## 学习目标
- [ ] 掌握缺失值的处理方法
- [ ] 学会处理重复数据
- [ ] 掌握数据类型转换
- [ ] 学会数据合并与连接
- [ ] 掌握数据透视表和分组聚合

## 缺失值处理

```python
# 检查缺失值
df.isnull().sum()           # 每列缺失值数量
df.isnull().any(axis=1)   # 哪些行有缺失值

# 删除缺失值
df.dropna()                 # 删除包含缺失值的行
df.dropna(axis=1)          # 删除包含缺失值的列
df.dropna(thresh=3)        # 删除少于3个非缺失值的行

# 填充缺失值
df.fillna(0)                # 用0填充
df.fillna(df.mean())       # 用均值填充
df.fillna(method='ffill')   # 前向填充
df.fillna(method='bfill')   # 后向填充

# 按组填充
df.groupby('班级')['成绩'].transform(lambda x: x.fillna(x.mean()))
```

## 重复值处理

```python
# 检查重复值
df.duplicated()                       # 检查所有列
df.duplicated(subset=['姓名'])      # 检查特定列
df.duplicated().sum()                # 重复行数量

# 删除重复值
df.drop_duplicates()                 # 删除所有重复行
df.drop_duplicates(keep='last')      # 保留最后一个
df.drop_duplicates(keep=False)       # 不保留任何重复行
df.drop_duplicates(subset=['姓名'])  # 基于特定列删除
```

## 数据类型转换

```python
# 基本类型转换
df['年龄'] = df['年龄'].astype(int)
df['成绩'] = df['成绩'].astype(float)

# 特殊转换
df['年龄'] = pd.to_numeric(df['年龄'], errors='coerce')  # 错误值设为NaN
df['日期'] = pd.to_datetime(df['日期'])
df['是否'] = df['是否'].map({'是': True, '否': False})

# 提取日期信息
df['年'] = df['日期'].dt.year
df['月'] = df['日期'].dt.month
df['星期'] = df['日期'].dt.dayofweek
```

## 数据合并

```python
# merge（基于列）
pd.merge(df1, df2, on='学号', how='inner')    # 内连接
pd.merge(df1, df2, on='学号', how='left')     # 左连接
pd.merge(df1, df2, on='学号', how='right')    # 右连接
pd.merge(df1, df2, on='学号', how='outer')    # 外连接
pd.merge(df1, df2, on=['姓名', '班级'])       # 多键连接

# concat（拼接）
pd.concat([df1, df2], axis=0)   # 纵向拼接
pd.concat([df1, df2], axis=1)   # 横向拼接

# join（基于索引）
df1.join(df2, how='outer')
```

## 分组聚合

```python
# 基本分组
df.groupby('产品')['销售额'].sum()
df.groupby(['产品', '地区'])['销售额'].sum()

# 多统计量
df.groupby('产品')['销售额'].agg(['sum', 'mean', 'max', 'min', 'count'])

# 自定义聚合
df.groupby('产品').agg({
    '销量': 'sum',
    '销售额': ['sum', 'mean']
})

# transform（保持原索引）
df['组均值'] = df.groupby('产品')['销量'].transform('mean')

# apply（自定义函数）
df.groupby('产品').apply(lambda x: x.nlargest(2, '销售额'))
```

## 数据透视表

```python
# 基本透视表
df.pivot_table(
    values='销售额',
    index='产品',
    columns='地区',
    aggfunc='sum',
    fill_value=0
)

# 多统计量透视表
df.pivot_table(
    values=['销量', '销售额'],
    index='产品',
    aggfunc={'销量': 'sum', '销售额': ['sum', 'mean']}
)

# melt（宽转长）
df.melt(id_vars=['姓名'], var_name='科目', value_name='成绩')
```

## 完整数据清洗流程

```python
# 1. 去除空格
df['姓名'] = df['姓名'].str.strip()

# 2. 转换数据类型
df['年龄'] = pd.to_numeric(df['年龄'], errors='coerce')
df['日期'] = pd.to_datetime(df['日期'])

# 3. 处理缺失值
df['年龄'] = df['年龄'].fillna(df['年龄'].median())

# 4. 删除重复值
df = df.drop_duplicates(subset=['姓名', '班级'], keep='first')

# 5. 统一格式
df['班级'] = df['班级'].str.upper()
```

## 遇到问题记录
（在这里记录学习过程中遇到的任何问题）

## 练习记录
- [ ] 完成 `04_pandas_advanced.py` 中的5个练习
- [ ] 尝试用自己的数据进行清洗练习

## 下一阶段
恭喜完成数据分析基础！可以选择：
1. Matplotlib/Seaborn 数据可视化
2. Scikit-learn 机器学习入门
3. PyTorch 深度学习
