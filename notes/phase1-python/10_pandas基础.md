# Pandas第一课：基础操作

## 学习目标
- [ ] 理解Pandas的数据结构：Series和DataFrame
- [ ] 掌握数据的创建和基本操作
- [ ] 学会数据的索引和切片
- [ ] 掌握数据的读取和保存

## 数据结构

### Series（一维）
```python
import pandas as pd

# 从列表创建
s = pd.Series([1, 2, 3, 4, 5])

# 自定义索引
s = pd.Series([85, 92, 78], index=['小明', '小红', '小刚'])

# 从字典创建
s = pd.Series({'语文': 85, '数学': 92})
```

### DataFrame（二维）
```python
# 从字典创建
df = pd.DataFrame({
    '姓名': ['小明', '小红'],
    '年龄': [20, 21],
    '成绩': [85, 92]
})

# 从数组创建
df = pd.DataFrame(
    np.array([[1, 2], [3, 4]]),
    columns=['A', 'B'],
    index=['x', 'y']
)
```

## 数据选择

| 方法 | 说明 | 示例 |
|------|------|------|
| `df['列名']` | 选择单列 | `df['姓名']` |
| `df[['列1', '列2']]` | 选择多列 | `df[['姓名', '成绩']]` |
| `df.loc[行, 列]` | 按标签选择 | `df.loc[0:2, ['姓名']]` |
| `df.iloc[行, 列]` | 按位置选择 | `df.iloc[0:2, 0:3]` |

## 条件筛选

```python
# 单条件
df[df['成绩'] > 85]

# 多条件（与）
df[(df['成绩'] >= 80) & (df['年龄'] >= 20)]

# 多条件（或）
df[(df['数学'] >= 90) | (df['英语'] >= 90)]
```

## 数据修改

```python
# 修改列
df['成绩'] = df['成绩'] + 5

# 添加新列
df['总分'] = df['数学'] + df['英语']

# 删除列
df = df.drop(['总分'], axis=1)

# 修改特定值
df.loc[0, '成绩'] = 95
```

## 数据排序

```python
# 按单列排序
df.sort_values('成绩', ascending=False)

# 按多列排序
df.sort_values(['年龄', '成绩'], ascending=[True, False])

# 按索引排序
df.sort_index()
```

## 读写数据

```python
# CSV
df.to_csv('data.csv', index=False)
df = pd.read_csv('data.csv')

# JSON
df.to_json('data.json', orient='records')
df = pd.read_json('data.json')

# Excel（需要openpyxl）
df.to_excel('data.xlsx', index=False)
df = pd.read_excel('data.xlsx')
```

## 常用属性

| 属性 | 说明 |
|------|------|
| `df.shape` | 形状（行数, 列数） |
| `df.columns` | 列名 |
| `df.index` | 行索引 |
| `df.dtypes` | 数据类型 |
| `df.info()` | 详细信息 |
| `df.describe()` | 统计信息 |

## 遇到问题记录
（在这里记录学习过程中遇到的任何问题）

## 练习记录
- [ ] 完成 `03_pandas_basics.py` 中的5个练习
- [ ] 尝试用Pandas处理自己的数据

## 下节课预告
- Pandas进阶：数据清洗和处理
