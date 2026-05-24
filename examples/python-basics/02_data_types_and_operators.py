"""
第二课：数据类型与运算符
学习目标：
1. 掌握Python基本数据类型
2. 学会使用各类运算符
3. 理解类型转换
"""

# ==================== 数据类型详解 ====================
print("=" * 50)
print("📊 Python基本数据类型")
print("=" * 50)

# 1. 数字（Numbers）
print("\n【数字类型】")

# 整数（int）
age = 25
year = 2026
big_number = 10_000_000  # 可以用下划线分隔，方便阅读

print(f"整数: {age}, {year}, {big_number}")
print(f"类型: {type(age)}")

# 浮点数（float）
height = 1.75
price = 99.99
scientific = 3.14e10  # 科学计数法

print(f"\n浮点数: {height}, {price}, {scientific}")
print(f"类型: {type(height)}")

# 复数（complex）- 了解即可
complex_num = 3 + 4j
print(f"\n复数: {complex_num}")


# 2. 字符串（String）
print("\n\n【字符串类型】")

name = "Python"
multiline = """这是一个
多行字符串"""

print(f"字符串: {name}")
print(f"多行字符串:\n{multiline}")

# 字符串常用操作
print("\n字符串操作：")
text = "  Hello Python  "
print(f"原字符串: '{text}'")
print(f"去掉空格: '{text.strip()}'")
print(f"转大写: '{name.upper()}'")
print(f"转小写: '{name.lower()}'")
print(f"长度: {len(name)}")
print(f"切片[0:3]: {name[0:3]}")  # 前3个字符


# 3. 布尔值（Boolean）
print("\n\n【布尔类型】")

is_student = True
is_working = False

print(f"is_student: {is_student}")
print(f"is_working: {is_working}")
print(f"布尔运算: {True and False}")  # False
print(f"布尔运算: {True or False}")   # True
print(f"布尔运算: {not True}")          # False


# 4. 空值（None）
print("\n\n【空值类型】")
result = None
print(f"result: {result}")
print(f"类型: {type(result)}")


# ==================== 运算符 ====================
print("\n" + "=" * 50)
print("🔢 Python运算符")
print("=" * 50)

a, b = 10, 3

# 1. 算术运算符
print("\n【算术运算符】")
print(f"a = {a}, b = {b}")
print(f"加法 a + b = {a + b}")
print(f"减法 a - b = {a - b}")
print(f"乘法 a * b = {a * b}")
print(f"除法 a / b = {a / b}")       # 浮点除法
print(f"整除 a // b = {a // b}")    # 取整数部分
print(f"取余 a % b = {a % b}")      # 取余数
print(f"幂 a ** b = {a ** b}")      # 10的3次方


# 2. 比较运算符
print("\n【比较运算符】")
print(f"a == b: {a == b}")   # 等于
print(f"a != b: {a != b}")   # 不等于
print(f"a > b: {a > b}")    # 大于
print(f"a < b: {a < b}")    # 小于
print(f"a >= b: {a >= b}")  # 大于等于
print(f"a <= b: {a <= b}")  # 小于等于


# 3. 赋值运算符
print("\n【赋值运算符】")
x = 10
print(f"x = {x}")
x += 5   # x = x + 5
print(f"x += 5 → x = {x}")
x -= 3   # x = x - 3
print(f"x -= 3 → x = {x}")
x *= 2   # x = x * 2
print(f"x *= 2 → x = {x}")
x /= 4   # x = x / 4
print(f"x /= 4 → x = {x}")


# 4. 逻辑运算符
print("\n【逻辑运算符】")
p, q = True, False
print(f"p = {p}, q = {q}")
print(f"p and q: {p and q}")  # 两边都True才True
print(f"p or q: {p or q}")    # 有一边True就True
print(f"not p: {not p}")      # 取反

# 实际应用：年龄判断
age = 20
is_adult = age >= 18
has_id = True
can_enter = is_adult and has_id
print(f"\n年龄{age}岁，成年且带证件能进入: {can_enter}")


# ==================== 类型转换 ====================
print("\n" + "=" * 50)
print("🔄 类型转换")
print("=" * 50)

# 字符串转数字
str_num = "100"
int_num = int(str_num)
float_num = float(str_num)
print(f"字符串'{str_num}' → 整数{int_num} → 浮点数{float_num}")

# 数字转字符串
num = 42
str_from_num = str(num)
print(f"数字{num} → 字符串'{str_from_num}'")

# 注意：不是所有转换都能成功
# int("hello")  # 会报错！

# 用户输入（都是字符串）
# user_input = input("请输入一个数字: ")  # 运行时取消注释
# number = int(user_input)
# print(f"你输入的数字是: {number}")


# ==================== 动手练习 ====================
print("\n" + "=" * 50)
print("✏️ 动手练习")
print("=" * 50)

print("""
请完成以下练习：

练习1：变量交换
a = 5, b = 10，如何交换它们的值？
（提示：可以用临时变量，也可以用Python特有的方式）

练习2：温度转换
摄氏度转华氏度：F = C × 9/5 + 32
编写代码，将25°C转换为华氏度

练习3：判断闰年
年份能被4整除但不能被100整除，或者能被400整除
判断2026年是不是闰年

练习4：字符串拼接
name = "小明"
age = 20
用3种不同方式输出："我叫小明，今年20岁"
（+号拼接、f-string、format方法）
""")

print("="*50)

# 练习答案（先尝试自己做，再看答案）
print("\n--- 答案参考 ---")

# 练习1答案
a, b = 5, 10
a, b = b, a  # Python特有的交换方式
print(f"交换后: a={a}, b={b}")

# 练习2答案
celsius = 25
fahrenheit = celsius * 9/5 + 32
print(f"{celsius}°C = {fahrenheit}°F")

# 练习3答案
year = 2026
is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
print(f"{year}年是闰年: {is_leap}")

# 练习4答案
name, age = "小明", 20
print("方式1: 我叫" + name + "，今年" + str(age) + "岁")
print(f"方式2: 我叫{name}，今年{age}岁")
print("方式3: 我叫{}，今年{}岁".format(name, age))


print("\n" + "=" * 50)
print("📚 本课总结")
print("=" * 50)
print("""
✅ 数据类型：int, float, str, bool, None
✅ 运算符：算术、比较、赋值、逻辑
✅ 类型转换：int(), float(), str()

下节课预告：列表和字典（最重要的数据结构）
""")
