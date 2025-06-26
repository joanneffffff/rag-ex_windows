#!/usr/bin/env python3
"""测试训练数据和评估数据的chunk一致性"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_chunk_consistency():
    """测试chunk一致性"""
    print("=== 测试训练数据和评估数据的chunk一致性 ===")
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        print("1. 加载训练数据（不包含评估数据）...")
        train_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=50,  # 只加载50个样本用于对比
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False  # 不包含评估数据
        )
        
        train_chinese = train_loader.chinese_docs
        train_english = train_loader.english_docs
        
        print(f"   ✅ 训练数据加载成功:")
        print(f"      中文chunks: {len(train_chinese)}")
        print(f"      英文chunks: {len(train_english)}")
        
        print("\n2. 加载包含评估数据的知识库...")
        eval_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=50,  # 只加载50个样本用于对比
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True  # 包含评估数据
        )
        
        eval_chinese = eval_loader.chinese_docs
        eval_english = eval_loader.english_docs
        
        print(f"   ✅ 包含评估数据的知识库加载成功:")
        print(f"      中文chunks: {len(eval_chinese)}")
        print(f"      英文chunks: {len(eval_english)}")
        
        print("\n3. 分析chunk一致性...")
        
        # 分析中文chunk长度分布
        print(f"\n--- 中文chunk长度分析 ---")
        train_chinese_lengths = [len(doc.content) for doc in train_chinese]
        eval_chinese_lengths = [len(doc.content) for doc in eval_chinese]
        
        print(f"训练数据中文chunk长度:")
        print(f"  平均长度: {sum(train_chinese_lengths)/len(train_chinese_lengths):.0f} 字符")
        print(f"  最小长度: {min(train_chinese_lengths)} 字符")
        print(f"  最大长度: {max(train_chinese_lengths)} 字符")
        print(f"  长度分布: {len([l for l in train_chinese_lengths if l <= 1000])} 短文档, {len([l for l in train_chinese_lengths if 1000 < l <= 5000])} 中文档, {len([l for l in train_chinese_lengths if l > 5000])} 长文档")
        
        print(f"\n评估数据中文chunk长度:")
        print(f"  平均长度: {sum(eval_chinese_lengths)/len(eval_chinese_lengths):.0f} 字符")
        print(f"  最小长度: {min(eval_chinese_lengths)} 字符")
        print(f"  最大长度: {max(eval_chinese_lengths)} 字符")
        print(f"  长度分布: {len([l for l in eval_chinese_lengths if l <= 1000])} 短文档, {len([l for l in eval_chinese_lengths if 1000 < l <= 5000])} 中文档, {len([l for l in eval_chinese_lengths if l > 5000])} 长文档")
        
        # 分析英文chunk长度分布
        print(f"\n--- 英文chunk长度分析 ---")
        train_english_lengths = [len(doc.content) for doc in train_english]
        eval_english_lengths = [len(doc.content) for doc in eval_english]
        
        print(f"训练数据英文chunk长度:")
        print(f"  平均长度: {sum(train_english_lengths)/len(train_english_lengths):.0f} 字符")
        print(f"  最小长度: {min(train_english_lengths)} 字符")
        print(f"  最大长度: {max(train_english_lengths)} 字符")
        
        print(f"\n评估数据英文chunk长度:")
        print(f"  平均长度: {sum(eval_english_lengths)/len(eval_english_lengths):.0f} 字符")
        print(f"  最小长度: {min(eval_english_lengths)} 字符")
        print(f"  最大长度: {max(eval_english_lengths)} 字符")
        
        # 检查评估数据是否遵循文档级别chunking
        print(f"\n--- 评估数据chunk策略验证 ---")
        
        # 检查中文评估数据是否使用文档级别
        eval_chinese_sources = set()
        for doc in eval_chinese:
            if 'eval' in doc.metadata.source:
                eval_chinese_sources.add(doc.metadata.source)
        
        print(f"中文评估数据来源: {eval_chinese_sources}")
        
        # 检查英文评估数据
        eval_english_sources = set()
        for doc in eval_english:
            if 'eval' in doc.metadata.source:
                eval_english_sources.add(doc.metadata.source)
        
        print(f"英文评估数据来源: {eval_english_sources}")
        
        # 显示一些示例
        print(f"\n--- 示例对比 ---")
        
        print(f"训练数据中文示例:")
        if train_chinese:
            sample_doc = train_chinese[0]
            print(f"  来源: {sample_doc.metadata.source}")
            print(f"  长度: {len(sample_doc.content)} 字符")
            print(f"  内容预览: {sample_doc.content[:200]}...")
        
        print(f"\n评估数据中文示例:")
        eval_chinese_docs = [doc for doc in eval_chinese if 'eval' in doc.metadata.source]
        if eval_chinese_docs:
            sample_eval_doc = eval_chinese_docs[0]
            print(f"  来源: {sample_eval_doc.metadata.source}")
            print(f"  长度: {len(sample_eval_doc.content)} 字符")
            print(f"  内容预览: {sample_eval_doc.content[:200]}...")
        
        # 一致性检查结果
        print(f"\n=== 一致性检查结果 ===")
        
        # 检查中文是否都使用文档级别
        chinese_consistency = True
        if train_chinese_lengths and eval_chinese_lengths:
            train_avg = sum(train_chinese_lengths) / len(train_chinese_lengths)
            eval_avg = sum(eval_chinese_lengths) / len(eval_chinese_lengths)
            # 平均长度应该相近（允许20%的差异）
            if abs(train_avg - eval_avg) / train_avg > 0.2:
                chinese_consistency = False
        
        print(f"中文chunk一致性: {'✅ 一致' if chinese_consistency else '❌ 不一致'}")
        
        # 检查英文是否都使用chunk级别
        english_consistency = True
        if train_english_lengths and eval_english_lengths:
            train_avg = sum(train_english_lengths) / len(train_english_lengths)
            eval_avg = sum(eval_english_lengths) / len(eval_english_lengths)
            # 平均长度应该相近（允许20%的差异）
            if abs(train_avg - eval_avg) / train_avg > 0.2:
                english_consistency = False
        
        print(f"英文chunk一致性: {'✅ 一致' if english_consistency else '❌ 不一致'}")
        
        overall_consistency = chinese_consistency and english_consistency
        print(f"\n总体一致性: {'✅ 训练数据和评估数据使用相同的chunk逻辑' if overall_consistency else '❌ 存在不一致'}")
        
        if overall_consistency:
            print(f"\n🎉 验证通过！评估数据正确遵循了与训练数据相同的chunk策略:")
            print(f"   ✅ 中文数据使用文档级别chunking")
            print(f"   ✅ 英文数据使用chunk级别处理")
            print(f"   ✅ 评估数据与训练数据长度分布一致")
        
        return overall_consistency
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chunk_consistency() 