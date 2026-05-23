"""
第六课：模块和包
学习目标：
1. 理解模块的概念和使用方法
2. 掌握import的各种方式
3. 学会创建和使用自定义模块
4. 了解Python标准库中的常用模块
5. 掌握包的组织结构

这是组织大型AI项目代码的基础！
"""

# ==================== 模块基础 ====================
print("=" * 60)
print("📦 模块基础")
print("=" * 60)

# 模块就是一个包含Python代码的 .py 文件
# 使用 import 关键字导入模块

# 方式1：导入整个模块
import math

print("\n【导入整个模块】")
print(f"math.pi = {math.pi}")
print(f"math.sqrt(16) = {math.sqrt(16)}")


# 方式2：从模块导入特定函数/变量
from math import pow, floor, ceil

print("\n【导入特定内容】")
print(f"pow(2, 3) = {pow(2, 3)}")      # 2的3次方
print(f"floor(3.7) = {floor(3.7)}")    # 向下取整
print(f"ceil(3.2) = {ceil(3.2)}")      # 向上取整


# 方式3：使用别名（简化长名称）
import random as rd
from datetime import datetime as dt

print("\n【使用别名】")
print(f"随机数: {rd.randint(1, 100)}")
print(f"当前时间: {dt.now()}")


# 方式4：导入所有（不推荐，容易命名冲突）
# from math import *  # 不推荐！


# ==================== 常用标准库模块 ====================
print("\n" + "=" * 60)
print("📚 常用标准库模块")
print("=" * 60)

# 1. os 模块 - 操作系统相关
import os

print("\n【os模块 - 文件路径】")
print(f"当前工作目录: {os.getcwd()}")
print(f"文件分隔符: {os.sep}")
print(f"路径拼接: {os.path.join('folder', 'file.txt')}")
print(f"绝对路径: {os.path.abspath('.')}")


# 2. sys 模块 - 系统相关
import sys

print("\n【sys模块 - 系统信息】")
print(f"Python版本: {sys.version}")
print(f"平台: {sys.platform}")
print(f"命令行参数: {sys.argv}")  # 运行脚本时传入的参数


# 3. random 模块 - 随机数
import random

print("\n【random模块 - 随机数】")
print(f"随机整数(1-100): {random.randint(1, 100)}")
print(f"随机小数(0-1): {random.random():.4f}")
fruits = ["苹果", "香蕉", "橙子", "葡萄"]
print(f"随机选择: {random.choice(fruits)}")
random.shuffle(fruits)
print(f"打乱顺序: {fruits}")


# 4. datetime 模块 - 日期时间
from datetime import datetime, timedelta, date

print("\n【datetime模块 - 日期时间】")
now = datetime.now()
print(f"当前时间: {now}")
print(f"格式化: {now.strftime('%Y-%m-%d %H:%M:%S')}")

future = now + timedelta(days=7)
print(f"7天后: {future.strftime('%Y-%m-%d')}")

birthday = date(1995, 5, 20)
today = date.today()
age = today.year - birthday.year
print(f"年龄: {age}岁")


# 5. json 模块 - JSON数据处理
import json

print("\n【json模块 - JSON处理】")
data = {
    "name": "小明",
    "age": 20,
    "courses": ["数学", "物理", "化学"],
    "is_student": True
}

# Python字典 → JSON字符串
json_str = json.dumps(data, ensure_ascii=False, indent=2)
print(f"JSON字符串:\n{json_str}")

# JSON字符串 → Python字典
parsed = json.loads(json_str)
print(f"解析后类型: {type(parsed)}")
print(f"姓名: {parsed['name']}")


# 6. re 模块 - 正则表达式（了解即可）
import re

print("\n【re模块 - 正则表达式】")
text = "我的邮箱是 example@email.com，电话是 13812345678"
# 查找邮箱
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
print(f"找到邮箱: {emails}")

# 查找手机号
phones = re.findall(r'1[3-9]\d{9}', text)
print(f"找到手机号: {phones}")


# 7. collections 模块 - 高级数据结构
from collections import Counter, defaultdict

print("\n【collections模块】")
# Counter - 计数器
words = ["苹果", "香蕉", "苹果", "橙子", "苹果", "香蕉"]
word_count = Counter(words)
print(f"词频统计: {dict(word_count)}")
print(f"最常见的: {word_count.most_common(2)}")

# defaultdict - 带默认值的字典
grouped = defaultdict(list)
for word in words:
    grouped[word[0]].append(word)  # 按首字母分组
print(f"按首字母分组: {dict(grouped)}")


# ==================== 自定义模块 ====================
print("\n" + "=" * 60)
print("📝 自定义模块")
print("=" * 60)

print("""
【创建自定义模块】

1. 创建一个 .py 文件（例如 my_module.py）
2. 在里面写函数和变量
3. 在其他文件用 import 导入使用

示例文件结构：
my_project/
├── main.py
└── my_utils.py  ← 自定义模块

my_utils.py 内容：
```python
# 文件名: my_utils.py

def greet(name):
    return f"你好，{name}！"

def calculate_average(scores):
    return sum(scores) / len(scores)

PI = 3.14159
```

main.py 中使用：
```python
import my_utils

print(my_utils.greet("小明"))
print(my_utils.PI)
```
""")

# 演示：将本课的函数保存为模块
# 实际项目中，你可以把常用的函数放到单独的 .py 文件中


# ==================== 包（Package）====================
print("\n" + "=" * 60)
print("📁 包（Package）- 模块的集合")
print("=" * 60)

print("""
【包的结构】

包是一个包含 __init__.py 文件的文件夹

my_package/
├── __init__.py      ← 必须有这个文件（可以为空）
├── module1.py
└── module2.py

使用方式：
import my_package.module1
from my_package import module2
from my_package.module1 import some_function

【实际项目结构示例】
ai_project/
├── main.py
├── requirements.txt
├── README.md
├── data/              ← 数据文件夹
│   ├── raw/
│   └── processed/
├── models/            ← 模型定义（包）
│   ├── __init__.py
│   ├── transformer.py
│   └── cnn.py
├── utils/             ← 工具函数（包）
│   ├── __init__.py
│   ├── data_loader.py
│   └── preprocess.py
└── configs/           ← 配置文件
    └── config.yaml
""")

# 演示导入（使用本项目的utils包）
print("\n【尝试导入项目中的模块】")
try:
    # 假设我们有一个 utils 模块
    from utils import math_utils
    print(f"3的平方: {math_utils.square(3)}")
except ImportError as e:
    print(f"提示: {e}")
    print("(这只是一个示例，实际需要你创建对应的文件)")


# ==================== 模块搜索路径 ====================
print("\n" + "=" * 60)
print("🔍 模块搜索路径")
print("=" * 60)

import sys

print("Python会在以下路径中查找模块：")
for i, path in enumerate(sys.path[:5], 1):  # 只显示前5个
    print(f"  {i}. {path}")
print("  ...")

print("\n你可以添加自定义路径：")
print("sys.path.append('/your/custom/path')")


# ==================== __name__ 和程序入口 ====================
print("\n" + "=" * 60)
print("🚪 __name__ 和程序入口")
print("=" * 60)

print(f"\n当前模块名: {__name__}")

print("""
【__name__ 的作用】

每个Python模块都有一个 __name__ 属性：
- 直接运行文件时，__name__ = "__main__"
- 作为模块被导入时，__name__ = 模块名

这就是为什么经常看到这样的代码：

```python
def main():
    # 主程序逻辑
    print("程序开始")
    
if __name__ == "__main__":
    main()
```

好处：
1. 被导入时，main() 不会自动执行
2. 直接运行时，main() 会执行
3. 可以清晰区分"被导入的代码"和"可执行代码"
""")


# ==================== 实际应用场景 ====================
print("\n" + "=" * 60)
print("💡 实际应用场景")
print("=" * 60)

# 场景1：配置文件管理
print("\n【场景1：配置文件管理】")
import json

config = {
    "model": {
        "name": "transformer",
        "layers": 12,
        "hidden_size": 768
    },
    "training": {
        "batch_size": 32,
        "learning_rate": 0.0001,
        "epochs": 10
    },
    "data": {
        "path": "./data/train.csv",
        "validation_split": 0.2
    }
}

# 保存配置
config_json = json.dumps(config, indent=2, ensure_ascii=False)
print("生成的配置：")
print(config_json)

# 场景2：日志记录
print("\n【场景2：日志记录】")
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("这是一条信息日志")
logging.warning("这是一条警告日志")

# 场景3：路径处理
print("\n【场景3：跨平台路径处理】")
from pathlib import Path

# 创建路径对象（自动处理 Windows/ macOS/ Linux 差异）
data_dir = Path("data") / "raw" / "train.csv"
print(f"路径: {data_dir}")
print(f"父目录: {data_dir.parent}")
print(f"文件名: {data_dir.name}")
print(f"后缀: {data_dir.suffix}")

# 检查文件是否存在
# print(f"文件存在: {data_dir.exists()}")


# ==================== 动手练习 ====================
print("\n" + "=" * 60)
print("✏️ 动手练习")
print("=" * 60)

print("""
练习1：随机密码生成器
使用 random 和 string 模块，生成一个8位随机密码，
包含大小写字母和数字。

练习2：日期计算
使用 datetime 模块，计算：
- 今天是今年的第几天
- 距离2026年春节还有多少天（假设春节是2026-02-17）

练习3：JSON文件读写
创建一个包含学生信息的字典，保存到 students.json 文件，
然后读取并打印。

练习4：统计词频
给定一段文本，使用 Counter 统计每个单词出现的次数，
找出出现频率最高的前5个单词。

练习5：创建自己的工具模块
创建一个 my_tools.py 文件，包含以下函数：
- calculate_bmi(height, weight) - 计算BMI指数
- is_valid_email(email) - 验证邮箱格式（使用re）
- save_to_json(data, filename) - 保存数据到JSON文件
然后在当前文件中导入并使用这些函数。
""")


# 答案参考
print("\n--- 答案参考 ---")

# 练习1：随机密码
import random
import string

def generate_password(length=8):
    """生成随机密码"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

print(f"\n练习1：随机密码 = {generate_password()}")

# 练习2：日期计算
from datetime import datetime, date

print("\n练习2：日期计算")
today = date.today()
start_of_year = date(today.year, 1, 1)
day_of_year = (today - start_of_year).days + 1
print(f"今天是今年的第 {day_of_year} 天")

spring_festival = date(2026, 2, 17)
days_until = (spring_festival - today).days
print(f"距离2026年春节还有 {days_until} 天")

# 练习3：JSON读写
print("\n练习3：JSON文件读写")
students = [
    {"name": "小明", "age": 20, "grade": "A"},
    {"name": "小红", "age": 19, "grade": "B"}
]

# 写入（实际会创建文件）
# with open("students.json", "w", encoding="utf-8") as f:
#     json.dump(students, f, ensure_ascii=False, indent=2)

# 读取
# with open("students.json", "r", encoding="utf-8") as f:
#     loaded = json.load(f)

print("代码已提供，可自行测试文件读写")

# 练习4：词频统计
print("\n练习4：词频统计")
text = "Python is great and Python is easy to learn and Python is powerful"
words = text.lower().split()
word_counter = Counter(words)
print(f"词频统计: {dict(word_counter)}")
print(f"Top 5: {word_counter.most_common(5)}")

# 练习5：自定义模块（示例代码）
print("\n练习5：自定义模块示例")
print("""
my_tools.py 内容：

import json
import re

def calculate_bmi(height_m, weight_kg):
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        status = "偏瘦"
    elif bmi < 24:
        status = "正常"
    elif bmi < 28:
        status = "偏胖"
    else:
        status = "肥胖"
    return {"bmi": round(bmi, 2), "status": status}

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def save_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到 {filename}")

使用方式：
from my_tools import calculate_bmi, is_valid_email
print(calculate_bmi(1.75, 70))
print(is_valid_email("test@example.com"))
""")


print("\n" + "=" * 60)
print("📚 Python基础阶段总结")
print("=" * 60)
print("""
✅ 第1课：Python初体验 - 环境、变量、print
✅ 第2课：数据类型与运算符 - int/str/float/bool, 各类运算符
✅ 第3课：列表和字典 - List/Dict/Tuple/Set, 遍历技巧
✅ 第4课：条件判断与循环 - if/for/while, break/continue, 推导式
✅ 第5课：函数定义 - def, 参数类型, lambda, 作用域
✅ 第6课：模块和包 - import, 标准库, 自定义模块

你已经掌握了Python编程的核心基础！

【下一阶段预告】
NumPy 和 Pandas - 数据科学基础
""")
