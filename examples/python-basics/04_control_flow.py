"""
第四课：条件判断与循环（程序的控制流程）
学习目标：
1. 掌握 if/else/elif 条件判断
2. 掌握 for 循环遍历
3. 掌握 while 循环
4. 理解 break、continue、pass
5. 学会列表推导式

这是编写逻辑复杂的AI数据处理代码的基础！
"""

# ==================== if/else/elif 条件判断 ====================
print("=" * 60)
print("🔀 条件判断 if/else/elif")
print("=" * 60)

# 基础 if
print("\n【基础 if 语句】")
age = 18

if age >= 18:
    print(f"{age}岁，已成年")

# if-else
print("\n【if-else】")
score = 75

if score >= 60:
    print("及格！")
else:
    print("不及格，需要补考")

# if-elif-else（多条件判断）
print("\n【if-elif-else 成绩分级】")
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"成绩{score}分，等级：{grade}")

# 嵌套 if（尽量少用，可以用逻辑运算符简化）
print("\n【嵌套判断】")
is_member = True
purchase_amount = 200

if is_member:
    if purchase_amount >= 100:
        discount = 0.8
    else:
        discount = 0.9
else:
    discount = 1.0

print(f"会员：{is_member}, 消费：{purchase_amount}元, 折扣：{discount}")

# 简化为逻辑运算符
if is_member and purchase_amount >= 100:
    discount = 0.8
elif is_member:
    discount = 0.9
else:
    discount = 1.0

# 三元表达式（简洁的条件赋值）
print("\n【三元表达式】")
age = 20
status = "成年" if age >= 18 else "未成年"
print(f"年龄{age}岁，状态：{status}")


# ==================== for 循环 ====================
print("\n" + "=" * 60)
print("🔄 for 循环 - 遍历序列")
print("=" * 60)

# 遍历列表
print("\n【遍历列表】")
fruits = ["苹果", "香蕉", "橙子", "葡萄"]

for fruit in fruits:
    print(f"  我喜欢吃{fruit}")

# 使用 range()
print("\n【range() 函数】")
print("range(5):", list(range(5)))          # 0,1,2,3,4
print("range(1,6):", list(range(1, 6)))   # 1,2,3,4,5
print("range(0,10,2):", list(range(0, 10, 2)))  # 0,2,4,6,8

# range 配合 for
print("\n【for + range】")
for i in range(1, 6):
    print(f"  第{i}次循环")

# enumerate - 同时获取索引和值
print("\n【enumerate - 带索引遍历】")
for index, fruit in enumerate(fruits):
    print(f"  索引{index}: {fruit}")

# 可以指定起始索引
for index, fruit in enumerate(fruits, start=1):
    print(f"  第{index}个水果: {fruit}")

# zip - 并行遍历多个序列
print("\n【zip - 并行遍历】")
names = ["小明", "小红", "小刚"]
scores = [85, 92, 78]

for name, score in zip(names, scores):
    print(f"  {name}: {score}分")

# 遍历字典
print("\n【遍历字典】")
student = {"name": "小明", "age": 20, "major": "CS"}

# 方式1：遍历键
for key in student:
    print(f"  {key}: {student[key]}")

# 方式2：遍历键值对（推荐）
print()
for key, value in student.items():
    print(f"  {key} = {value}")

# 方式3：只遍历值
print()
for value in student.values():
    print(f"  值: {value}")


# ==================== while 循环 ====================
print("\n" + "=" * 60)
print("🔄 while 循环 - 条件循环")
print("=" * 60)

# 基础 while
print("\n【基础 while】")
count = 0
while count < 5:
    print(f"  计数: {count}")
    count += 1  # 必须手动增加，否则会无限循环！

# while 读取用户输入（实际场景）
print("\n【while 实际应用】")
# 模拟用户输入正确密码
password = "123456"
attempts = 0
max_attempts = 3

# 取消注释可运行交互版本
# while attempts < max_attempts:
#     user_input = input("请输入密码: ")
#     if user_input == password:
#         print("密码正确！")
#         break
#     else:
#         attempts += 1
#         print(f"密码错误，还剩{max_attempts - attempts}次机会")
# else:
#     print("次数用完，账户锁定")

# while-else（循环正常结束才执行else）
print("\n【while-else】")
n = 5
while n > 0:
    print(f"  n = {n}")
    n -= 1
else:
    print("  循环正常结束（没有被break）")


# ==================== 循环控制：break / continue / pass ====================
print("\n" + "=" * 60)
print("🛑 循环控制 break / continue / pass")
print("=" * 60)

# break - 跳出整个循环
print("\n【break - 跳出循环】")
for i in range(10):
    if i == 5:
        print(f"  遇到{i}，跳出循环")
        break
    print(f"  处理: {i}")

# continue - 跳过当前迭代，继续下一次
print("\n【continue - 跳过当前】")
for i in range(5):
    if i == 2:
        print(f"  跳过{i}")
        continue
    print(f"  处理: {i}")

# pass - 占位符，什么都不做
print("\n【pass - 占位符】")
for i in range(3):
    if i == 1:
        pass  # 暂时没想好怎么处理，先占位
        print(f"  i={i}，用了pass占位")
    else:
        print(f"  处理: {i}")

# 实际应用：找列表中的第一个偶数
print("\n【实际应用：break】")
numbers = [1, 3, 5, 8, 10, 12]
for num in numbers:
    if num % 2 == 0:
        print(f"找到第一个偶数: {num}")
        break

# 跳过None值处理数据
print("\n【实际应用：continue】")
data = [10, None, 20, None, 30]
for value in data:
    if value is None:
        continue  # 跳过无效数据
    print(f"处理数据: {value}")


# ==================== 嵌套循环 ====================
print("\n" + "=" * 60)
print("🔄 嵌套循环")
print("=" * 60)

# 打印乘法表
print("\n【九九乘法表】")
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j:<2}", end="  ")
    print()  # 换行

# 遍历二维列表
print("\n【遍历二维列表】")
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

for row in matrix:
    for num in row:
        print(f"{num:2}", end=" ")
    print()


# ==================== 列表推导式（Python特色） ====================
print("\n" + "=" * 60)
print("✨ 列表推导式 - 简洁高效")
print("=" * 60)

# 基础推导式
print("\n【基础推导式】")
squares = [x**2 for x in range(5)]
print(f"0-4的平方: {squares}")

# 带条件的推导式
print("\n【带条件的推导式】")
evens = [x for x in range(10) if x % 2 == 0]
print(f"0-9的偶数: {evens}")

# 多条件
print("\n【多条件推导式】")
numbers = [x for x in range(100) if x % 3 == 0 if x % 5 == 0]
print(f"100以内能被3和5整除的数: {numbers}")

# if-else 在推导式中
print("\n【if-else 在推导式中】")
labels = ["偶数" if x % 2 == 0 else "奇数" for x in range(5)]
print(f"0-4的奇偶标签: {labels}")

# 字典推导式
print("\n【字典推导式】")
fruits = ["苹果", "香蕉", "橙子"]
fruit_lengths = {fruit: len(fruit) for fruit in fruits}
print(f"水果名字长度: {fruit_lengths}")

# 集合推导式
print("\n【集合推导式】")
unique_lengths = {len(fruit) for fruit in ["苹果", "香蕉", "橙子", "梨"]}
print(f"名字长度集合: {unique_lengths}")

# 嵌套推导式（了解即可）
print("\n【嵌套推导式】")
matrix = [[i * j for j in range(1, 4)] for i in range(1, 4)]
print(f"3x3乘法矩阵: {matrix}")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：数据过滤
print("\n【场景1：过滤数据】")
raw_scores = [55, 88, 92, 45, 76, 85, None, 90, 30]
# 过滤掉None和不及格分数
valid_scores = [s for s in raw_scores if s is not None and s >= 60]
print(f"原始成绩: {raw_scores}")
print(f"有效及格成绩: {valid_scores}")

# 场景2：数据处理
print("\n【场景2：批量处理】")
students = [
    {"name": "小明", "scores": [80, 90, 85]},
    {"name": "小红", "scores": [95, 88, 92]},
    {"name": "小刚", "scores": [70, 75, 80]}
]

results = []
for student in students:
    avg = sum(student["scores"]) / len(student["scores"])
    status = "优秀" if avg >= 90 else "良好" if avg >= 80 else "及格" if avg >= 60 else "不及格"
    results.append({
        "name": student["name"],
        "average": round(avg, 1),
        "status": status
    })

print("学生成绩统计:")
for r in results:
    print(f"  {r['name']}: 平均{r['average']}分, 评级{r['status']}")

# 场景3：查找数据
print("\n【场景3：查找数据】")
users = [
    {"id": 1, "name": "张三", "active": True},
    {"id": 2, "name": "李四", "active": False},
    {"id": 3, "name": "王五", "active": True}
]

target_id = 2
found_user = None
for user in users:
    if user["id"] == target_id:
        found_user = user
        break

if found_user:
    print(f"找到用户: {found_user['name']}")
else:
    print("用户不存在")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：判断闰年
输入一个年份，判断是否是闰年：
- 能被4整除但不能被100整除，或
- 能被400整除

练习2：打印三角形
使用嵌套循环打印如下图案：
*
**
***
****
*****

练习3：求素数
找出2-100之间的所有素数（只能被1和自身整除的数）

练习4：列表推导式应用
给定一个数字列表，用一行代码完成：
- 提取所有偶数
- 将每个数平方
- 结果按从大到小排序

练习5：字典统计
统计一段文字中每个字符出现的次数，忽略空格和大小写。
例如："Hello World" → {'h':1, 'e':1, 'l':3, 'o':2, 'w':1, 'r':1, 'd':1}
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1：判断闰年
print("\n练习1：闰年判断")
def is_leap_year(year):
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

for year in [2000, 1900, 2024, 2025]:
    result = "是" if is_leap_year(year) else "不是"
    print(f"{year}年{result}闰年")

# 练习2：打印三角形
print("\n练习2：三角形")
for i in range(1, 6):
    print("*" * i)

# 练习3：求素数
print("\n练习3：素数")
primes = []
for num in range(2, 101):
    is_prime = True
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            is_prime = False
            break
    if is_prime:
        primes.append(num)
print(f"2-100之间的素数共{len(primes)}个: {primes[:10]}...")

# 练习4：列表推导式
print("\n练习4：列表推导式")
nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = sorted([x**2 for x in nums if x % 2 == 0], reverse=True)
print(f"偶数平方降序: {result}")

# 练习5：字符统计
print("\n练习5：字符统计")
text = "Hello World"
counts = {}
for char in text.lower():
    if char != " ":
        counts[char] = counts.get(char, 0) + 1
print(f"字符统计: {counts}")
# 字典推导式版本
char_count = {c: text.lower().count(c) for c in set(text.lower()) if c != " "}
print(f"推导式版本: {char_count}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ 条件判断：if/elif/else，三元表达式
✅ for循环：遍历序列，range()，enumerate()，zip()
✅ while循环：条件循环，注意避免无限循环
✅ 循环控制：break跳出，continue跳过，pass占位
✅ 嵌套循环：处理二维数据
✅ 推导式：[x for x in list if condition]，简洁高效

下节课预告：函数定义（def）
""")
