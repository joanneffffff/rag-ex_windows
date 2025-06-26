#!/usr/bin/env python3
"""测试英文chunk修复效果"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_english_chunk_fix():
    """测试英文chunk修复效果"""
    print("=== 测试英文chunk修复效果 ===")
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        print("1. 加载训练数据（英文）...")
        train_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,  # 加载100个样本
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False  # 不包含评估数据
        )
        
        train_english = train_loader.english_docs
        
        print(f"   ✅ 英文训练数据加载成功: {len(train_english)} 个chunks")
        
        # 分析训练数据长度
        train_lengths = [len(doc.content) for doc in train_english]
        train_avg = sum(train_lengths) / len(train_lengths) if train_lengths else 0
        print(f"  训练数据平均长度: {train_avg:.0f} 字符")
        print(f"  训练数据长度范围: {min(train_lengths)} - {max(train_lengths)} 字符")
        
        print("\n2. 加载包含评估数据的知识库...")
        eval_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,  # 加载100个样本
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True  # 包含评估数据
        )
        
        eval_english = eval_loader.english_docs
        
        print(f"   ✅ 包含评估数据的知识库加载成功: {len(eval_english)} 个chunks")
        
        # 分离训练数据和评估数据
        train_docs = [doc for doc in eval_english if 'eval' not in doc.metadata.source]
        eval_docs = [doc for doc in eval_english if 'eval' in doc.metadata.source]
        
        print(f"  训练数据chunks: {len(train_docs)}")
        print(f"  评估数据chunks: {len(eval_docs)}")
        
        # 分析评估数据长度
        if eval_docs:
            eval_lengths = [len(doc.content) for doc in eval_docs]
            eval_avg = sum(eval_lengths) / len(eval_lengths)
            print(f"  评估数据平均长度: {eval_avg:.0f} 字符")
            print(f"  评估数据长度范围: {min(eval_lengths)} - {max(eval_lengths)} 字符")
            
            # 检查一致性
            length_diff = abs(train_avg - eval_avg) / train_avg
            print(f"  长度差异: {length_diff:.2%}")
            
            if length_diff < 0.3:  # 允许30%的差异
                print("  ✅ 英文chunk长度一致性良好")
            else:
                print("  ❌ 英文chunk长度差异较大")
        
        # 显示一些示例
        print(f"\n3. 示例对比...")
        
        print(f"训练数据英文示例:")
        if train_docs:
            sample_train = train_docs[0]
            print(f"  来源: {sample_train.metadata.source}")
            print(f"  长度: {len(sample_train.content)} 字符")
            print(f"  内容预览: {sample_train.content[:200]}...")
        
        print(f"\n评估数据英文示例:")
        if eval_docs:
            sample_eval = eval_docs[0]
            print(f"  来源: {sample_eval.metadata.source}")
            print(f"  长度: {len(sample_eval.content)} 字符")
            print(f"  内容预览: {sample_eval.content[:200]}...")
        
        # 检查chunk策略
        print(f"\n4. Chunk策略验证...")
        
        eval_sources = set()
        for doc in eval_docs:
            eval_sources.add(doc.metadata.source)
        
        print(f"评估数据来源: {eval_sources}")
        
        # 检查是否有chunk分割
        chunked_docs = [doc for doc in eval_docs if '_chunk_' in doc.metadata.source]
        print(f"分割的chunks: {len(chunked_docs)}")
        
        if chunked_docs:
            print("  ✅ 英文评估数据正确使用了chunk分割策略")
        else:
            print("  ⚠️  英文评估数据未进行chunk分割")
        
        print(f"\n🎉 英文chunk修复测试完成！")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_english_chunk_fix() 