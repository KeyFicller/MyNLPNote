"""
第一课：Python初体验
学习目标：
1. 确认Python环境正常
2. 理解print()函数
3. 认识字符串和变量
"""

# ==================== 第一个程序 ====================
# print() 函数用来在屏幕上输出内容
print("Hello, Python!")
print("你好，生成式AI学习者！")

# ==================== 变量基础 ====================
# 变量就像一个盒子，用来存储数据
# 不需要声明类型，直接赋值即可

name = "小明"           # 字符串（文字）
age = 25               # 整数
temperature = 36.5      # 浮点数（小数）

print("\n--- 变量示例 ---")
print("姓名:", name)
print("年龄:", age)
print("体温:", temperature)

# ==================== 字符串操作 ====================
print("\n--- 字符串操作 ---")

# 字符串可以用单引号或双引号
message1 = '单引号字符串'
message2 = "双引号字符串"
print(message1)
print(message2)

# 字符串拼接（连接）
greeting = "你好，" + name + "！"
print(greeting)

# f-string 格式化（推荐用法）
info = f"我叫{name}，今年{age}岁"
print(info)

# ==================== 小练习 ====================
print("\n--- 动手练习 ---")
print("请修改下面的代码，填入你自己的信息：")
print("# 修改这里 ↓↓↓")
my_name = "KeyFicller"
my_age = 28
my_hobby = "Movie"
my_hometown = "China"

print(f"\n我是{my_name}，{my_age}岁，喜欢{my_hobby}，家乡{my_hometown}")
print("\n修改完成后，运行这个文件看看效果！")

# ==================== 今日任务 ====================
print("\n" + "="*50)
print("📚 今日学习任务：")
print("1. 运行这个文件，确认环境正常")
print("2. 修改上面的练习，填入真实信息")
print("3. 尝试添加一个新变量，比如 hometown = '你的家乡'")
print("4. 用print()输出你的家乡信息")
print("="*50)
