#!/usr/bin/env python3
"""
简化诊断脚本
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def main():
    """主诊断函数"""
    print("=" * 60)
    print("简化诊断检索问题")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        config = Config()
        
        print(f"1. 检查数据路径:")
        print(f"   中文数据路径: {config.data.chinese_data_path}")
        print(f"   英文数据路径: {config.data.english_data_path}")
        
        # 检查文件是否存在
        chinese_path = config.data.chinese_data_path
        english_path = config.data.english_data_path
        
        print(f"\n2. 检查文件存在性:")
        print(f"   中文文件存在: {os.path.exists(chinese_path)}")
        print(f"   英文文件存在: {os.path.exists(english_path)}")
        
        if os.path.exists(chinese_path):
            size = os.path.getsize(chinese_path) / (1024 * 1024)
            print(f"   中文文件大小: {size:.2f} MB")
        
        if os.path.exists(english_path):
            size = os.path.getsize(english_path) / (1024 * 1024)
            print(f"   英文文件大小: {size:.2f} MB")
        
        # 测试数据加载
        print(f"\n3. 测试数据加载:")
        try:
            from xlm.utils.optimized_data_loader import OptimizedDataLoader
            
            loader = OptimizedDataLoader(
                data_dir="data",
                max_samples=100,
                chinese_document_level=True,
                english_chunk_level=True
            )
            
            stats = loader.get_statistics()
            print(f"   ✅ 优化数据加载器成功:")
            print(f"      中文文档数: {stats['chinese_docs']}")
            print(f"      英文文档数: {stats['english_docs']}")
            print(f"      中文平均长度: {stats['chinese_avg_length']:.2f}")
            print(f"      英文平均长度: {stats['english_avg_length']:.2f}")
            
            # 显示一些示例
            print(f"\n4. 中文文档示例:")
            for i, doc in enumerate(loader.chinese_docs[:2]):
                print(f"   文档 {i+1}: {doc.content[:100]}...")
            
            print(f"\n✅ 数据加载正常！")
            print(f"问题可能在于:")
            print(f"1. 编码器模型问题")
            print(f"2. 检索参数设置")
            print(f"3. 查询与文档内容不匹配")
            
        except Exception as e:
            print(f"   ❌ 优化数据加载器失败: {e}")
            print(f"   错误详情:")
            import traceback
            traceback.print_exc()
            
            # 尝试传统加载器
            print(f"\n5. 尝试传统数据加载器:")
            try:
                from xlm.utils.dual_language_loader import DualLanguageLoader
                
                loader = DualLanguageLoader()
                chinese_docs, english_docs = loader.load_dual_language_data(
                    chinese_data_path=config.data.chinese_data_path,
                    english_data_path=config.data.english_data_path
                )
                
                print(f"   ✅ 传统数据加载器成功:")
                print(f"      中文文档数: {len(chinese_docs)}")
                print(f"      英文文档数: {len(english_docs)}")
                
            except Exception as e2:
                print(f"   ❌ 传统数据加载器也失败: {e2}")
                print(f"   错误详情:")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 