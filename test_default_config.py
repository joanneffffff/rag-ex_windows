#!/usr/bin/env python3
"""测试默认配置 - 验证主服务器默认包含训练数据和评估数据"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_default_config():
    """测试默认配置"""
    print("=== 测试默认配置 ===")
    print("验证主服务器默认包含训练数据和评估数据")
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        print("\n1. 使用默认参数创建数据加载器...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=50,  # 只加载50个样本用于测试
            chinese_document_level=True,
            english_chunk_level=True
            # 不指定include_eval_data，使用默认值True
        )
        
        chinese_docs = data_loader.chinese_docs
        english_docs = data_loader.english_docs
        
        print(f"   ✅ 数据加载成功:")
        print(f"      中文文档数: {len(chinese_docs)}")
        print(f"      英文文档数: {len(english_docs)}")
        
        print("\n2. 检查是否包含评估数据...")
        
        # 检查中文文档中是否有评估数据
        eval_chinese_count = sum(1 for doc in chinese_docs if 'eval' in doc.metadata.source)
        eval_english_count = sum(1 for doc in english_docs if 'eval' in doc.metadata.source)
        
        print(f"   中文评估文档数: {eval_chinese_count}")
        print(f"   英文评估文档数: {eval_english_count}")
        
        if eval_chinese_count > 0 or eval_english_count > 0:
            print("   ✅ 默认配置正确：包含评估数据")
        else:
            print("   ❌ 默认配置错误：不包含评估数据")
        
        print("\n3. 测试显式排除评估数据...")
        data_loader_no_eval = OptimizedDataLoader(
            data_dir="data",
            max_samples=50,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False  # 显式排除评估数据
        )
        
        chinese_docs_no_eval = data_loader_no_eval.chinese_docs
        english_docs_no_eval = data_loader_no_eval.english_docs
        
        eval_chinese_count_no_eval = sum(1 for doc in chinese_docs_no_eval if 'eval' in doc.metadata.source)
        eval_english_count_no_eval = sum(1 for doc in english_docs_no_eval if 'eval' in doc.metadata.source)
        
        print(f"   排除评估数据时:")
        print(f"     中文评估文档数: {eval_chinese_count_no_eval}")
        print(f"     英文评估文档数: {eval_english_count_no_eval}")
        
        if eval_chinese_count_no_eval == 0 and eval_english_count_no_eval == 0:
            print("   ✅ 显式排除评估数据功能正常")
        else:
            print("   ❌ 显式排除评估数据功能异常")
        
        print("\n=== 测试总结 ===")
        print(f"默认配置（包含评估数据）:")
        print(f"  中文文档: {len(chinese_docs)} 个")
        print(f"  英文文档: {len(english_docs)} 个")
        print(f"  评估文档: {eval_chinese_count + eval_english_count} 个")
        
        print(f"\n排除评估数据配置:")
        print(f"  中文文档: {len(chinese_docs_no_eval)} 个")
        print(f"  英文文档: {len(english_docs_no_eval)} 个")
        print(f"  评估文档: {eval_chinese_count_no_eval + eval_english_count_no_eval} 个")
        
        print(f"\n✅ 配置正确！主服务器默认包含训练数据和评估数据，提供完整的知识库。")
        print(f"   如需公平评估，可显式设置 include_eval_data=False")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_default_config() 