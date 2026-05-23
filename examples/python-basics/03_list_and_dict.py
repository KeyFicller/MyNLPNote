"""
第三课：列表和字典（Python核心数据结构）
学习目标：
1. 掌握列表（List）的创建和操作
2. 掌握字典（Dict）的创建和操作
3. 理解元组（Tuple）和集合（Set）
4. 学会遍历数据结构

这些是未来学习NumPy、Pandas、PyTorch的基础！
"""

# ==================== 列表（List） ====================
print("=" * 60)
print("📋 列表（List）- 有序、可变的集合")
print("=" * 60)

# 创建列表
fruits = ["苹果", "香蕉", "橙子", "葡萄"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True, None]  # 可以混合类型

print(f"水果列表: {fruits}")
print(f"数字列表: {numbers}")
print(f"混合列表: {mixed}")

# 访问元素（索引从0开始）
print(f"\n【索引访问】")
print(f"第一个水果: {fruits[0]}")
print(f"第二个水果: {fruits[1]}")
print(f"最后一个水果: {fruits[-1]}")  # 负数索引从末尾开始
print(f"倒数第二个: {fruits[-2]}")

# 切片操作 [start:end]（左闭右开）
print(f"\n【切片操作】")
print(f"前3个: {fruits[0:3]}")   # 索引0,1,2
print(f"后2个: {fruits[2:4]}")   # 索引2,3
print(f"步长2: {numbers[::2]}")  # 每隔一个取一个
print(f"反转: {numbers[::-1]}")   # 倒序

# 列表操作
print(f"\n【常用操作】")

# 添加元素
fruits.append("西瓜")          # 末尾添加
print(f"append后: {fruits}")

fruits.insert(1, "草莓")      # 指定位置插入
print(f"insert后: {fruits}")

fruits.extend(["梨", "桃"])    # 批量添加
print(f"extend后: {fruits}")

# 删除元素
fruits.remove("草莓")          # 删除指定值（第一个匹配）
print(f"remove后: {fruits}")

popped = fruits.pop()          # 删除并返回最后一个
print(f"pop后: {fruits}, 弹出了: {popped}")

# 查找和统计
print(f"\n【查找统计】")
print(f"列表长度: {len(fruits)}")
print(f"苹果出现次数: {fruits.count('苹果')}")
print(f"香蕉的索引: {fruits.index('香蕉')}")
print(f"西瓜在列表中? {'西瓜' in fruits}")

# 排序
scores = [85, 92, 78, 90, 88]
print(f"\n原成绩: {scores}")
scores.sort()                  # 原地排序（升序）
print(f"升序排序: {scores}")
scores.sort(reverse=True)      # 降序
print(f"降序排序: {scores}")

# sorted返回新列表，不改变原列表
original = [3, 1, 4, 1, 5]
sorted_new = sorted(original, reverse=True)
print(f"原列表: {original}")
print(f"新排序: {sorted_new}")


# ==================== 字典（Dictionary） ====================
print("\n" + "=" * 60)
print("📖 字典（Dict）- 键值对存储")
print("=" * 60)

# 创建字典
student = {
    "name": "小明",
    "age": 20,
    "major": "计算机科学",
    "grades": [85, 90, 88, 92]
}

print(f"学生信息: {student}")

# 访问值
print(f"\n【访问值】")
print(f"姓名: {student['name']}")
print(f"年龄: {student['age']}")
print(f"专业: {student.get('major')}")  # get方法更安全的访问
print(f"成绩: {student.get('grades', [])}")
print(f"不存在键: {student.get('hobby', '未设置')}")  # 提供默认值

# 添加和修改
print(f"\n【添加修改】")
student["hobby"] = "编程"          # 添加新键值对
student["age"] = 21                # 修改已有值
print(f"更新后: {student}")

# 删除
removed = student.pop("hobby")       # 删除并返回值
print(f"删除了hobby: {removed}")
print(f"当前信息: {student}")

# 字典常用方法
print(f"\n【字典方法】")
print(f"所有键: {list(student.keys())}")
print(f"所有值: {list(student.values())}")
print(f"所有键值对: {list(student.items())}")

# 判断键是否存在
print(f"'name'在字典中? {'name' in student}")
print(f"'hobby'在字典中? {'hobby' in student}")


# ==================== 列表和字典的组合使用 ====================
print("\n" + "=" * 60)
print("🔄 列表和字典的组合（实际开发常用）")
print("=" * 60)

# 列表套字典 - 学生成绩表
students = [
    {"name": "小明", "math": 90, "english": 85},
    {"name": "小红", "math": 88, "english": 92},
    {"name": "小刚", "math": 95, "english": 78}
]

print("学生成绩表:")
for s in students:
    avg = (s["math"] + s["english"]) / 2
    print(f"  {s['name']}: 数学{s['math']}, 英语{s['english']}, 平均{avg:.1f}")

# 字典套列表 - 按科目分类
grades_by_subject = {
    "math": [90, 88, 95, 92, 85],
    "english": [85, 92, 78, 88, 90]
}

print(f"\n按科目统计:")
for subject, scores in grades_by_subject.items():
    avg = sum(scores) / len(scores)
    print(f"  {subject}: 平均分{avg:.1f}, 最高分{max(scores)}")


# ==================== 元组（Tuple）和 集合（Set） ====================
print("\n" + "=" * 60)
print("📌 元组和集合（了解即可）")
print("=" * 60)

# 元组 - 不可变的有序集合
coordinates = (120.5, 30.2)  # 坐标
rgb = (255, 128, 0)          # 颜色

print(f"坐标: {coordinates}")
print(f"红色值: {rgb[0]}")
# rgb[0] = 100  # 报错！元组不可修改

# 解包 - 元组的实用技巧
x, y = coordinates
print(f"x={x}, y={y}")

# 集合 - 无序的唯一元素集合
unique_numbers = {1, 2, 3, 3, 3, 3, 4}  # 自动去重
print(f"集合（自动去重）: {unique_numbers}")

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
print(f"交集: {a & b}")      # {3, 4}
print(f"并集: {a | b}")      # {1, 2, 3, 4, 5, 6}
print(f"差集: {a - b}")      # {1, 2}


# ==================== 遍历技巧 ====================
print("\n" + "=" * 60)
print("🔄 遍历技巧（for循环）")
print("=" * 60)

# 遍历列表
print("\n【遍历列表】")
fruits = ["苹果", "香蕉", "橙子"]

# 方式1：直接遍历
for fruit in fruits:
    print(f"  水果: {fruit}")

# 方式2：带索引（enumerate）
for index, fruit in enumerate(fruits):
    print(f"  索引{index}: {fruit}")

# 遍历字典
print("\n【遍历字典】")
student = {"name": "小明", "age": 20, "major": "CS"}

# 方式1：遍历键
for key in student:
    print(f"  {key}: {student[key]}")

# 方式2：遍历键值对（推荐）
for key, value in student.items():
    print(f"  {key} = {value}")

# 列表推导式 - Python特色（初学者先了解）
print("\n【列表推导式】")
squares = [x**2 for x in range(5)]
print(f"0-4的平方: {squares}")

evens = [x for x in range(10) if x % 2 == 0]
print(f"0-9的偶数: {evens}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：列表操作
创建一个包含5个数字的列表，完成：
- 添加一个数字到末尾
- 在第2个位置插入一个数字
- 删除最后一个元素
- 计算列表中所有数字的和

练习2：字典操作
创建一个字典存储一本书的信息：
- 书名、作者、价格、出版年份
- 添加一个"评分"键
- 修改价格
- 输出所有信息

练习3：综合应用
有一个学生列表，每个学生是一个字典：
students = [
    {"name": "小明", "scores": [80, 90, 85]},
    {"name": "小红", "scores": [95, 88, 92]}
]
计算每个学生的平均分，并找出平均分最高的学生。

练习4：数据统计
给定一个数字列表 numbers = [12, 5, 8, 12, 3, 5, 12, 8]
统计每个数字出现的次数，用字典存储结果。
""")

# region My Submission 
list = [1, 3, 5, 6, 7]
list.append(9)
list.insert(2, 23)
list.pop()
print(f"list is {list}, sum is {sum(list)}")

dict = {
    "name": "Hello",
    "arthor": "KeyFicller",
    "price": "99",
    "year": "2026",
    "score": "99"
}
dict["score"] = 20
print(dict)

students = [
    {"name": "小明", "scores": [80, 90, 85]},
    {"name": "小红", "scores": [95, 88, 92]}
]
best = None
best_score = None
for i, s in enumerate( students):
    average = sum(s["scores"]) / len(s["scores"])
    if best is None:
        best = i
        best_score = average
    elif average > best:
        best = i
        best_score = average

print(f"最高分: {students[best].get("name")} ({best_score:.1f}分)")

numbers = [12, 5, 8, 12, 3, 5, 12, 8]
dict = {}
for u in numbers:
    dict[u] = dict.get(u, 0) + 1
print(dict)

# endregion

# 答案参考
print("\n--- 答案参考 ---")

# 练习1答案
nums = [1, 2, 3, 4, 5]
nums.append(6)
nums.insert(2, 99)
nums.pop()
total = sum(nums)
print(f"练习1: {nums}, 总和={total}")

# 练习2答案
book = {"title": "Python入门", "author": "张三", "price": 59.9, "year": 2024}
book["rating"] = 4.5
book["price"] = 49.9
print(f"\n练习2: 书名《{book['title']}》, 作者{book['author']}, 价格{book['price']}元")

# 练习3答案
students = [
    {"name": "小明", "scores": [80, 90, 85]},
    {"name": "小红", "scores": [95, 88, 92]}
]
best_student = None
best_avg = 0
for s in students:
    avg = sum(s["scores"]) / len(s["scores"])
    print(f"  {s['name']}: 平均{avg:.1f}分")
    if avg > best_avg:
        best_avg = avg
        best_student = s["name"]
print(f"最高分: {best_student} ({best_avg:.1f}分)")

# 练习4答案
numbers = [12, 5, 8, 12, 3, 5, 12, 8]
count_dict = {}
for n in numbers:
    if n in count_dict:
        count_dict[n] += 1
    else:
        count_dict[n] = 1
print(f"\n练习4: 统计结果 {count_dict}")

# 更简洁的写法（使用get）
count_dict2 = {}
for n in numbers:
    count_dict2[n] = count_dict2.get(n, 0) + 1
print(f"简洁写法: {count_dict2}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ 列表 List: 有序、可变、用[]，常用方法：append, insert, pop, sort
✅ 字典 Dict: 键值对、用{}，常用方法：get, keys, values, items
✅ 元组 Tuple: 不可变、用()，用于固定数据（如坐标）
✅ 集合 Set: 无序唯一、自动去重，用于交集并集运算
✅ 遍历：for循环 + enumerate（带索引）/ items()（字典键值对）

下节课预告：条件判断和循环（if/else, for, while）
""")
