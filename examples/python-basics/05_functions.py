"""
第五课：函数定义（def）
学习目标：
1. 掌握函数的定义和调用
2. 理解参数和返回值
3. 学会使用默认参数、可变参数
4. 理解作用域（局部变量和全局变量）
5. 掌握lambda表达式

函数是代码复用的核心，也是构建AI模型和数据处理流程的基础！
"""

# ==================== 函数基础 ====================
print("=" * 60)
print("🔧 函数基础")
print("=" * 60)

# 定义函数
def greet():
    """这是一个简单的问候函数"""
    print("你好！欢迎使用Python函数。")

# 调用函数
print("【调用无参数函数】")
greet()


# 带参数的函数
def greet_person(name):
    """带参数的问候函数"""
    print(f"你好，{name}！")

print("\n【调用带参数函数】")
greet_person("小明")
greet_person("小红")


# 带返回值的函数
def add(a, b):
    """计算两个数的和"""
    result = a + b
    return result

print("\n【调用带返回值的函数】")
sum_result = add(3, 5)
print(f"3 + 5 = {sum_result}")

# 可以简写
print(f"10 + 20 = {add(10, 20)}")


# 多返回值（实际是返回元组）
def get_min_max(numbers):
    """返回列表中的最小值和最大值"""
    return min(numbers), max(numbers)

print("\n【多返回值】")
nums = [3, 1, 4, 1, 5, 9, 2, 6]
minimum, maximum = get_min_max(nums)
print(f"列表 {nums}")
print(f"最小值: {minimum}, 最大值: {maximum}")


# ==================== 参数详解 ====================
print("\n" + "=" * 60)
print("📦 参数详解")
print("=" * 60)

# 位置参数
def describe_pet(animal, name):
    """位置参数"""
    print(f"我有一只{animal}，名字叫{name}。")

print("\n【位置参数】")
describe_pet("狗", "旺财")  # 按位置对应
describe_pet("猫", "咪咪")


# 关键字参数（更清晰的调用）
print("\n【关键字参数】")
describe_pet(animal="兔子", name="小白")
describe_pet(name="大黄", animal="狗")  # 顺序不重要


# 默认参数值
def greet_with_style(name, greeting="你好", punctuation="！"):
    """带默认值的参数"""
    print(f"{greeting}，{name}{punctuation}")

print("\n【默认参数】")
greet_with_style("小明")  # 使用所有默认值
greet_with_style("小红", "早上好")  # 修改第一个默认参数
greet_with_style("小刚", "晚上好", "。")  # 修改所有默认参数
greet_with_style("小李", punctuation="~")  # 只修改第二个默认参数


# 可变参数 *args（接收任意数量的位置参数）
def sum_all(*numbers):
    """计算所有参数的和"""
    total = 0
    for n in numbers:
        total += n
    return total

print("\n【可变参数 *args】")
print(f"sum_all() = {sum_all()}")
print(f"sum_all(1, 2, 3) = {sum_all(1, 2, 3)}")
print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")

# *args 内部是一个元组
def show_args(*args):
    print(f"接收到的参数: {args}")
    print(f"类型: {type(args)}")

print("\n【*args 内部结构】")
show_args(1, 2, 3, "hello")


# 可变参数 **kwargs（接收任意数量的关键字参数）
def build_profile(**kwargs):
    """构建用户资料"""
    profile = {}
    for key, value in kwargs.items():
        profile[key] = value
    return profile

print("\n【可变参数 **kwargs】")
user1 = build_profile(name="小明", age=20, city="北京")
user2 = build_profile(name="小红", age=22, city="上海", hobby="编程")
print(f"用户1: {user1}")
print(f"用户2: {user2}")

# **kwargs 内部是一个字典
def show_kwargs(**kwargs):
    print(f"接收到的参数: {kwargs}")
    print(f"类型: {type(kwargs)}")

print("\n【**kwargs 内部结构】")
show_kwargs(a=1, b=2, name="test")


# 组合使用所有参数类型
def complex_function(pos1, pos2, default="默认值", *args, **kwargs):
    """展示所有参数类型的组合"""
    print(f"位置参数: {pos1}, {pos2}")
    print(f"默认参数: {default}")
    print(f"可变参数 *args: {args}")
    print(f"关键字参数 **kwargs: {kwargs}")

print("\n【所有参数类型组合】")
complex_function("a", "b")
print("-" * 40)
complex_function("a", "b", "新默认值")
print("-" * 40)
complex_function("a", "b", "新默认值", 1, 2, 3, x=10, y=20)


# ==================== 作用域 ====================
print("\n" + "=" * 60)
print("🌐 作用域（变量的可见范围）")
print("=" * 60)

# 局部变量
def local_example():
    local_var = "我是局部变量"
    print(f"函数内部: {local_var}")

print("\n【局部变量】")
local_example()
# print(local_var)  # 错误！函数外部访问不到


# 全局变量
global_var = "我是全局变量"

def use_global():
    print(f"函数内部访问全局: {global_var}")

print("\n【全局变量】")
use_global()
print(f"函数外部: {global_var}")


# 在函数内修改全局变量（需要 global 关键字）
counter = 0

def increment():
    global counter  # 声明使用全局变量
    counter += 1
    print(f"计数器: {counter}")

print("\n【修改全局变量】")
increment()
increment()
increment()


# 更好的做法：避免使用 global，通过参数和返回值
def better_increment(count):
    """更好的方式：通过参数传递和返回值"""
    return count + 1

print("\n【推荐：避免使用global】")
count = 0
count = better_increment(count)
count = better_increment(count)
print(f"最终计数: {count}")


# ==================== 函数作为参数（高阶函数） ====================
print("\n" + "=" * 60)
print("🔄 函数作为参数（高阶函数）")
print("=" * 60)

# 定义一个接受函数作为参数的函数
def apply_operation(numbers, operation):
    """对列表中的每个数字应用操作"""
    results = []
    for n in numbers:
        results.append(operation(n))
    return results

# 定义一些简单的操作函数
def square(x):
    return x ** 2

def double(x):
    return x * 2

def absolute(x):
    return abs(x)

print("\n【函数作为参数】")
numbers = [1, -2, 3, -4, 5]
print(f"原始列表: {numbers}")
print(f"平方: {apply_operation(numbers, square)}")
print(f"双倍: {apply_operation(numbers, double)}")
print(f"绝对值: {apply_operation(numbers, absolute)}")


# ==================== Lambda 表达式 ====================
print("\n" + "=" * 60)
print("⚡ Lambda 表达式（匿名函数）")
print("=" * 60)

# 普通函数
def add_normal(x, y):
    return x + y

# 等价的lambda
add_lambda = lambda x, y: x + y

print("\n【lambda基础】")
print(f"普通函数: {add_normal(2, 3)}")
print(f"lambda: {add_lambda(2, 3)}")

# lambda 与高阶函数结合
print("\n【lambda与map】")
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(f"原始: {numbers}")
print(f"平方: {squared}")

print("\n【lambda与filter】")
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(f"偶数: {evens}")

print("\n【lambda与sorted】")
students = [("小明", 85), ("小红", 92), ("小刚", 78)]
sorted_by_score = sorted(students, key=lambda x: x[1], reverse=True)
print(f"按分数排序: {sorted_by_score}")


# ==================== 递归函数 ====================
print("\n" + "=" * 60)
print("🔄 递归函数（了解即可）")
print("=" * 60)

# 阶乘：n! = n * (n-1) * (n-2) * ... * 1
def factorial(n):
    """计算阶乘"""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print("\n【递归求阶乘】")
for i in range(1, 6):
    print(f"{i}! = {factorial(i)}")


# 斐波那契数列
def fibonacci(n):
    """计算斐波那契数列第n项"""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print("\n【斐波那契数列】")
fib_sequence = [fibonacci(i) for i in range(10)]
print(f"前10项: {fib_sequence}")

# 注意：递归要注意终止条件，否则会导致无限递归！


# ==================== 文档字符串和类型提示 ====================
print("\n" + "=" * 60)
print("📝 文档字符串和类型提示（现代Python推荐）")
print("=" * 60)

def calculate_area(length: float, width: float) -> float:
    """
    计算矩形的面积。
    
    Args:
        length: 矩形的长度
        width: 矩形的宽度
    
    Returns:
        矩形的面积
    
    Example:
        >>> calculate_area(5.0, 3.0)
        15.0
    """
    return length * width

print("\n【类型提示和文档字符串】")
area = calculate_area(5.0, 3.0)
print(f"5×3的矩形面积: {area}")
print(f"\n函数文档:\n{calculate_area.__doc__}")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：数据处理管道
def clean_data(data):
    """清洗数据：去除None值"""
    return [x for x in data if x is not None]

def normalize(data):
    """归一化数据到0-1范围"""
    min_val, max_val = min(data), max(data)
    if max_val == min_val:
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def batch_process(data_list, processors):
    """批量处理数据"""
    for processor in processors:
        data_list = processor(data_list)
    return data_list

print("\n【场景1：数据处理管道】")
raw_data = [10, None, 20, None, 30, 40, None, 50]
processed = batch_process(raw_data, [clean_data, normalize])
print(f"原始数据: {raw_data}")
print(f"处理后: {[round(x, 2) for x in processed]}")


# 场景2：配置生成器
def create_model_config(model_type, **kwargs):
    """创建模型配置"""
    base_configs = {
        "transformer": {"layers": 12, "heads": 8, "hidden_size": 768},
        "cnn": {"layers": 5, "filters": [64, 128, 256, 512, 512]},
        "rnn": {"layers": 2, "hidden_size": 256, "bidirectional": True}
    }
    
    config = base_configs.get(model_type, {}).copy()
    config.update(kwargs)
    config["type"] = model_type
    return config

print("\n【场景2：模型配置生成】")
transformer_config = create_model_config("transformer", dropout=0.1, learning_rate=1e-4)
print(f"Transformer配置: {transformer_config}")


# 场景3：验证器
def create_validator(min_val=None, max_val=None, allow_none=False):
    """创建验证函数"""
    def validator(value):
        if value is None:
            return allow_none
        if min_val is not None and value < min_val:
            return False
        if max_val is not None and value > max_val:
            return False
        return True
    return validator

print("\n【场景3：动态验证器】")
age_check = create_validator(min_val=0, max_val=150)
scores = [25, -5, 200, 85]
for score in scores:
    print(f"  {score}岁: {'有效' if age_check(score) else '无效'}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：温度转换函数
编写一个函数 celsius_to_fahrenheit(celsius)，将摄氏度转为华氏度。
公式：F = C × 9/5 + 32
要求：添加类型提示和文档字符串

练习2：列表处理函数
编写一个函数 process_list(numbers, operation)，根据operation参数对列表进行不同操作：
- "sum": 求和
- "avg": 求平均
- "max": 求最大值
- "min": 求最小值

练习3：使用 *args 和 **kwargs
编写一个函数 log_message(level, *args, **kwargs)，格式化输出日志：
- level: 日志级别（如"INFO", "ERROR"）
- *args: 要输出的内容
- **kwargs: 额外的上下文信息（如user_id, timestamp等）

练习4：装饰器（进阶）
编写一个简单的计时装饰器，计算函数执行时间。
（提示：使用time模块和嵌套函数）

练习5：lambda应用
使用lambda和sorted，对一个学生列表按多个条件排序：
先按年级（grade）升序，再按分数（score）降序。
students = [
    {"name": "小明", "grade": 3, "score": 85},
    {"name": "小红", "grade": 2, "score": 92},
    {"name": "小刚", "grade": 3, "score": 78},
    {"name": "小李", "grade": 2, "score": 88}
]
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1
def celsius_to_fahrenheit(celsius: float) -> float:
    """将摄氏度转换为华氏度"""
    return celsius * 9 / 5 + 32

print("\n练习1：温度转换")
print(f"25°C = {celsius_to_fahrenheit(25):.1f}°F")
print(f"0°C = {celsius_to_fahrenheit(0):.1f}°F")


# 练习2
def process_list(numbers, operation):
    """根据操作类型处理列表"""
    operations = {
        "sum": sum,
        "avg": lambda x: sum(x) / len(x),
        "max": max,
        "min": min
    }
    func = operations.get(operation)
    return func(numbers) if func else None

print("\n练习2：列表处理")
nums = [1, 2, 3, 4, 5]
print(f"列表: {nums}")
print(f"求和: {process_list(nums, 'sum')}")
print(f"平均: {process_list(nums, 'avg')}")
print(f"最大: {process_list(nums, 'max')}")


# 练习3
from datetime import datetime

def log_message(level, *args, **kwargs):
    """格式化输出日志"""
    timestamp = kwargs.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    message = " ".join(str(arg) for arg in args)
    extra = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != "timestamp")
    
    log_str = f"[{timestamp}] [{level}] {message}"
    if extra:
        log_str += f" | {extra}"
    print(log_str)

print("\n练习3：日志函数")
log_message("INFO", "用户登录成功", user_id=12345)
log_message("ERROR", "数据库连接失败", retry_count=3, max_retries=5)


# 练习4
import time

def timer_decorator(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 执行时间: {elapsed:.4f}秒")
        return result
    return wrapper

@timer_decorator
def slow_function():
    """模拟耗时操作"""
    time.sleep(0.1)
    return "完成"

print("\n练习4：计时装饰器")
slow_function()


# 练习5
print("\n练习5：多条件排序")
students = [
    {"name": "小明", "grade": 3, "score": 85},
    {"name": "小红", "grade": 2, "score": 92},
    {"name": "小刚", "grade": 3, "score": 78},
    {"name": "小李", "grade": 2, "score": 88}
]

# 先按年级升序，再按分数降序
sorted_students = sorted(students, key=lambda x: (x["grade"], -x["score"]))
print("排序结果:")
for s in sorted_students:
    print(f"  {s['name']}: 年级{s['grade']}, 分数{s['score']}")


print("\n" + "=" * 60)
print("📚 本课总结")
print("=" * 60)
print("""
✅ 函数定义：def 函数名(参数): 函数体
✅ 参数类型：位置参数、关键字参数、默认参数、*args、**kwargs
✅ 返回值：return 可以返回单个值或多个值（元组）
✅ 作用域：局部变量、全局变量（global关键字）
✅ 高阶函数：函数作为参数传递
✅ Lambda：简洁的匿名函数，常与map/filter/sorted配合
✅ 递归：函数调用自身，注意终止条件
✅ 类型提示：def func(x: int) -> str（现代Python推荐）

下节课预告：模块和包（import、文件组织）
""")
