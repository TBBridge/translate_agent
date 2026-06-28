#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON词典加载功能测试脚本
"""

import os
import sys
import json
from main import TranslationAgent

def test_json_dictionary_loading():
    """测试JSON词典加载功能"""
    print("开始测试JSON词典加载功能...")
    
    # 创建测试用的TranslationAgent实例
    agent = TranslationAgent(
        dictionary_file="./dictionary.json",
        target_language="zh_cn"  # 测试简体中文
    )
    
    # 测试词典加载
    dictionary = agent._load_dictionary_from_file()
    
    print(f"加载的词典条目数: {len(dictionary)}")
    
    # 显示前5个词典条目作为示例
    print("\n前5个词典条目示例:")
    count = 0
    for key, value in dictionary.items():
        if count >= 5:
            break
        print(f"  {key} = {value}")
        count += 1
    
    # 测试不同目标语言的字段映射
    print("\n测试不同目标语言的字段映射:")
    
    # 测试英文
    agent.target_language = "en"
    en_dictionary = agent._load_dictionary_from_file()
    print(f"英文词典条目数: {len(en_dictionary)}")
    
    # 测试繁体中文
    agent.target_language = "zh_tw"
    tw_dictionary = agent._load_dictionary_from_file()
    print(f"繁体中文词典条目数: {len(tw_dictionary)}")
    
    # 测试简体中文
    agent.target_language = "zh_cn"
    cn_dictionary = agent._load_dictionary_from_file()
    print(f"简体中文词典条目数: {len(cn_dictionary)}")
    
    # 验证词典不为空
    if dictionary:
        print("\n✅ JSON词典加载测试成功!")
        return True
    else:
        print("\n❌ JSON词典加载测试失败!")
        return False

def test_txt_dictionary_fallback():
    """测试TXT词典回退功能"""
    print("\n开始测试TXT词典回退功能...")
    
    # 创建测试用的TranslationAgent实例，使用不存在的JSON文件
    agent = TranslationAgent(
        dictionary_file="./dictionary.txt",  # 使用TXT文件
        target_language="zh_cn"
    )
    
    # 测试TXT词典加载
    dictionary = agent._load_dictionary_from_file()
    
    print(f"TXT词典条目数: {len(dictionary)}")
    
    if dictionary:
        print("✅ TXT词典回退测试成功!")
        return True
    else:
        print("❌ TXT词典回退测试失败!")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n开始测试错误处理...")
    
    # 测试不存在的文件
    agent = TranslationAgent(
        dictionary_file="./nonexistent.json",
        target_language="zh_cn"
    )
    
    dictionary = agent._load_dictionary_from_file()
    
    if len(dictionary) == 0:
        print("✅ 不存在文件的错误处理测试成功!")
        return True
    else:
        print("❌ 不存在文件的错误处理测试失败!")
        return False

def test_ollama_model_mapping():
    """测试 Ollama 模型映射规则"""
    print("\n开始测试 Ollama 模型映射规则...")

    agent = TranslationAgent(
        dictionary_file="./dictionary.json",
        target_language="en"
    )

    assert agent._select_ollama_model("gpt-5.1", "translate") == "gpt-oss:120b"
    assert agent._select_ollama_model("gpt-3.5-turbo", "translate") == "gpt-oss:20b"
    assert agent._select_ollama_model("qwen3-max", "translate") == "qwen3:30b"
    assert agent._select_ollama_model("qwen-max", "translate") == "qwen3:30b"
    assert agent._select_ollama_model("deepseek-chat", "translate") == "deepseek-r1:14b"

    print("✅ Ollama 模型映射测试通过!")
    return True

if __name__ == "__main__":
    print("JSON词典功能测试")
    print("=" * 50)
    
    # 检查dictionary.json文件是否存在
    if not os.path.exists("./dictionary.json"):
        print("❌ dictionary.json文件不存在，请确保文件存在后再运行测试")
        sys.exit(1)
    
    # 运行测试
    test1_result = test_json_dictionary_loading()
    test2_result = test_txt_dictionary_fallback()
    test3_result = test_error_handling()
    test4_result = test_ollama_model_mapping()
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    print(f"JSON词典加载: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"TXT词典回退: {'✅ 通过' if test2_result else '❌ 失败'}")
    print(f"错误处理: {'✅ 通过' if test3_result else '❌ 失败'}")
    print(f"Ollama模型映射: {'✅ 通过' if test4_result else '❌ 失败'}")
    
    if all([test1_result, test2_result, test3_result, test4_result]):
        print("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("\n⚠️ 部分测试失败!")
        sys.exit(1)


