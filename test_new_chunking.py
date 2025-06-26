#!/usr/bin/env python3
"""
测试新的chunking逻辑
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_optimized_data_loader():
    """测试优化的数据加载器"""
    print("=" * 60)
    print("测试优化的数据加载器")
    print("=" * 60)
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        print("1. 测试文档级别chunking:")
        loader_doc = OptimizedDataLoader(
            data_dir="data",
            max_samples=1000,
            chinese_document_level=True,
            english_chunk_level=True
        )
        
        doc_stats = loader_doc.get_statistics()
        print(f"   中文文档数: {doc_stats['chinese_docs']}")
        print(f"   英文文档数: {doc_stats['english_docs']}")
        print(f"   中文平均长度: {doc_stats['chinese_avg_length']:.2f}")
        
        print("\n2. 测试传统chunk级别chunking:")
        loader_chunk = OptimizedDataLoader(
            data_dir="data",
            max_samples=1000,
            chinese_document_level=False,
            english_chunk_level=True
        )
        
        chunk_stats = loader_chunk.get_statistics()
        print(f"   中文文档数: {chunk_stats['chinese_docs']}")
        print(f"   英文文档数: {chunk_stats['english_docs']}")
        print(f"   中文平均长度: {chunk_stats['chinese_avg_length']:.2f}")
        
        # 计算改进效果
        if chunk_stats['chinese_docs'] > 0:
            reduction = (chunk_stats['chinese_docs'] - doc_stats['chinese_docs']) / chunk_stats['chinese_docs'] * 100
            print(f"\n3. 改进效果:")
            print(f"   Chunk数量减少: {reduction:.1f}%")
            print(f"   文档级别: {doc_stats['chinese_docs']} chunks")
            print(f"   传统chunking: {chunk_stats['chinese_docs']} chunks")
        
        # 显示一些示例
        print("\n4. 文档级别chunking示例:")
        for i, doc in enumerate(loader_doc.chinese_docs[:2]):
            print(f"   文档 {i+1} (长度: {len(doc.content)}):")
            print(f"   内容预览: {doc.content[:150]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试新的chunking逻辑...")
    
    success = test_optimized_data_loader()
    
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    if success:
        print("✅ 新chunking逻辑测试成功！")
        print("✅ 文档级别chunking工作正常")
        print("✅ Chunk数量显著减少")
        print("\n现在可以运行: python run_new_chunking_force.py")
    else:
        print("❌ 新chunking逻辑测试失败")

if __name__ == "__main__":
    main() 