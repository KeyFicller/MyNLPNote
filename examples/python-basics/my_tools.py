"""
自定义工具模块 - 供练习使用

这个模块展示了如何创建和使用自定义模块。
可以在其他Python文件中通过 import my_tools 来使用。
"""

import json
import re


def calculate_bmi(height_m, weight_kg):
    """
    计算BMI指数并返回状态
    
    Args:
        height_m: 身高（米）
        weight_kg: 体重（公斤）
    
    Returns:
        包含BMI值和状态的字典
    """
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
    """
    验证邮箱格式是否正确
    
    Args:
        email: 邮箱地址
    
    Returns:
        bool: 是否有效
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def save_to_json(data, filename):
    """
    将数据保存为JSON文件
    
    Args:
        data: 要保存的数据（字典或列表）
        filename: 文件名
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 数据已保存到 {filename}")


def load_from_json(filename):
    """
    从JSON文件读取数据
    
    Args:
        filename: 文件名
    
    Returns:
        读取的数据
    """
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_password(length=8):
    """
    生成随机密码
    
    Args:
        length: 密码长度
    
    Returns:
        随机密码字符串
    """
    import random
    import string
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choice(chars) for _ in range(length))


# 测试代码（只有直接运行此文件时才会执行）
if __name__ == "__main__":
    print("=== 测试 my_tools 模块 ===\n")
    
    # 测试 BMI 计算
    print("1. BMI计算测试")
    result = calculate_bmi(1.75, 70)
    print(f"   身高1.75m，体重70kg: {result}")
    
    # 测试邮箱验证
    print("\n2. 邮箱验证测试")
    test_emails = ["test@example.com", "invalid.email", "user@domain.org"]
    for email in test_emails:
        valid = "✓" if is_valid_email(email) else "✗"
        print(f"   {email}: {valid}")
    
    # 测试密码生成
    print("\n3. 随机密码生成")
    print(f"   8位密码: {generate_password(8)}")
    print(f"   12位密码: {generate_password(12)}")
    
    print("\n=== 测试完成 ===")
