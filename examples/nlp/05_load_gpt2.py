#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
加载 GPT-2 模型（使用镜像解决网络问题）
"""

import os
import sys

# 设置 Hugging Face 镜像（中国大陆推荐）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = './hf_cache'  # 本地缓存目录

print("=" * 70)
print("加载 GPT-2 模型（使用 hf-mirror.com 镜像）")
print("=" * 70)

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
    print("✅ transformers 已安装")
except ImportError:
    print("❌ 请先安装 transformers: pip install transformers")
    sys.exit(1)

print("\n正在下载 GPT-2 模型（约 500MB）...")
print("使用镜像: https://hf-mirror.com")
print("-" * 70)

try:
    # 加载 Tokenizer
    print("\n1. 加载 Tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(
        'gpt2',
        cache_dir='./hf_cache'
    )
    print("✅ Tokenizer 加载成功!")
    print(f"   词表大小: {len(tokenizer)}")
    
    # 加载模型
    print("\n2. 加载 GPT-2 模型...")
    model = GPT2LMHeadModel.from_pretrained(
        'gpt2',
        cache_dir='./hf_cache'
    )
    print("✅ GPT-2 模型加载成功!")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 测试生成
    print("\n3. 测试文本生成...")
    print("-" * 70)
    
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device='cpu'  # 使用CPU，如有GPU可改为0
    )
    
    # 生成示例
    prompt = "Once upon a time"
    print(f"\n输入提示: '{prompt}'")
    print("\n生成结果:")
    
    result = generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True
    )
    
    generated_text = result[0]['generated_text']
    print(generated_text)
    
    print("\n" + "=" * 70)
    print("✅ GPT-2 模型加载和测试完成!")
    print("=" * 70)
    print("\n模型已缓存到: ./hf_cache/")
    print("下次加载会直接从本地读取，无需重新下载")
    
except Exception as e:
    print(f"\n❌ 错误: {e}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接")
    print("2. 尝试其他镜像: https://hf-mirror.com")
    print("3. 手动下载模型文件到 ./hf_cache/")
    sys.exit(1)
