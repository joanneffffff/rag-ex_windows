#!/usr/bin/env python3
"""测试中文chunk优化效果"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_chinese_chunk_optimization():
    """测试中文chunk优化效果"""
    print("=== 测试中文chunk优化效果 ===")
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        print("1. 加载优化后的中文数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,  # 只加载100个样本
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False  # 不包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        # 分析中文chunk长度分布
        chinese_lengths = [len(doc.content) for doc in chinese_chunks]
        english_lengths = [len(doc.content) for doc in english_chunks]
        
        print(f"\n2. 中文chunk长度分析:")
        print(f"   平均长度: {sum(chinese_lengths)/len(chinese_lengths):.0f} 字符")
        print(f"   最小长度: {min(chinese_lengths)} 字符")
        print(f"   最大长度: {max(chinese_lengths)} 字符")
        print(f"   长度分布:")
        print(f"     短文档 (≤1000字符): {len([l for l in chinese_lengths if l <= 1000])}")
        print(f"     中文档 (1000-5000字符): {len([l for l in chinese_lengths if 1000 < l <= 5000])}")
        print(f"     长文档 (>5000字符): {len([l for l in chinese_lengths if l > 5000])}")
        
        print(f"\n3. 英文chunk长度分析:")
        print(f"   平均长度: {sum(english_lengths)/len(english_lengths):.0f} 字符")
        print(f"   最小长度: {min(english_lengths)} 字符")
        print(f"   最大长度: {max(english_lengths)} 字符")
        
        # 计算chunk比例
        chinese_english_ratio = len(chinese_chunks) / len(english_chunks)
        print(f"\n4. Chunk比例分析:")
        print(f"   中文/英文chunk比例: {chinese_english_ratio:.2f}")
        
        if chinese_english_ratio > 5:
            print("   ⚠️  中文chunk仍然过多，可能需要进一步优化")
        elif chinese_english_ratio > 2:
            print("   ⚠️  中文chunk较多，但可以接受")
        else:
            print("   ✅ 中文chunk数量合理")
        
        # 显示一些示例
        print(f"\n5. 中文chunk示例:")
        if chinese_chunks:
            sample_doc = chinese_chunks[0]
            print(f"   来源: {sample_doc.metadata.source}")
            print(f"   长度: {len(sample_doc.content)} 字符")
            print(f"   内容预览: {sample_doc.content[:200]}...")
            
            # 检查是否包含JSON格式
            if '{' in sample_doc.content and '}' in sample_doc.content:
                print("   ✅ 保持原始JSON格式")
            else:
                print("   ⚠️  可能被转换了")
        
        # 优化建议
        print(f"\n6. 优化建议:")
        
        if chinese_english_ratio > 5:
            print("   🔧 建议进一步优化:")
            print("      - 增加文档长度阈值（如从8192增加到16384）")
            print("      - 减少文档分割频率")
            print("      - 考虑使用更大的chunk单位")
        else:
            print("   ✅ 当前chunk策略合理")
        
        # 与之前结果对比
        print(f"\n7. 与之前结果对比:")
        print(f"   之前中文chunks: 6259个")
        print(f"   现在中文chunks: {len(chinese_chunks)}个")
        print(f"   改进比例: {(6259 - len(chinese_chunks)) / 6259 * 100:.1f}%")
        
        if len(chinese_chunks) < 6259:
            print("   🎉 成功减少了中文chunk数量！")
        else:
            print("   ⚠️  需要进一步优化")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_chinese_chunk_optimization() 