#!/usr/bin/env python3
"""
测试context-only数据加载方法
展示如何只加载context字段，减少内存使用
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.utils.dual_language_loader import DualLanguageLoader

def test_context_only_loading():
    """测试context-only数据加载方法"""
    print("=== 测试context-only数据加载 ===")
    
    # 初始化数据加载器
    loader = DualLanguageLoader()
    
    # 方法1: 加载完整的QCA数据（包含query, answer等）
    print("\n1. 加载完整的QCA数据...")
    try:
        full_docs = loader.load_jsonl_data("evaluate_mrr/tatqa_train_qc.jsonl", "english")
        print(f"完整QCA数据: {len(full_docs)} 个文档")
        if full_docs:
            print(f"第一个文档元数据: {full_docs[0].metadata}")
            print(f"第一个文档内容长度: {len(full_docs[0].content)}")
    except Exception as e:
        print(f"加载完整QCA数据失败: {e}")
    
    # 方法2: 只加载context字段（优化版本）
    print("\n2. 只加载context字段...")
    try:
        context_only_docs = loader.load_context_only_data("evaluate_mrr/tatqa_train_qc.jsonl", "english")
        print(f"Context-only数据: {len(context_only_docs)} 个文档")
        if context_only_docs:
            print(f"第一个文档元数据: {context_only_docs[0].metadata}")
            print(f"第一个文档内容长度: {len(context_only_docs[0].content)}")
    except Exception as e:
        print(f"加载context-only数据失败: {e}")
    
    # 比较内存使用
    print("\n3. 内存使用比较:")
    print("- 完整QCA数据: 包含query, answer, context等字段")
    print("- Context-only数据: 只包含context字段")
    print("- 优势: 减少内存使用，提高加载速度，避免序列化问题")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_context_only_loading() 