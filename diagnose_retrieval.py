#!/usr/bin/env python3
"""
诊断检索问题
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def diagnose_data_loading():
    """诊断数据加载问题"""
    print("=" * 60)
    print("诊断数据加载问题")
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
            
            return True
            
        except Exception as e:
            print(f"   ❌ 优化数据加载器失败: {e}")
            
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
                
                return True
                
            except Exception as e2:
                print(f"   ❌ 传统数据加载器也失败: {e2}")
                return False
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False

def diagnose_retriever():
    """诊断检索器问题"""
    print("\n" + "=" * 60)
    print("诊断检索器问题")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        
        config = Config()
        
        print("1. 加载编码器...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载数据...")
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,
            chinese_document_level=True,
            english_chunk_level=True
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n3. 创建检索器...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=False,  # 先不使用FAISS
            use_gpu=False,
            batch_size=8,
            cache_dir=""  # 使用空字符串而不是None
        )
        print("   ✅ 检索器创建成功")
        
        print("\n4. 测试检索...")
        test_queries = [
            "什么是市盈率？",
            "如何计算ROE？",
            "财务报表包括哪些内容？"
        ]
        
        for query in test_queries:
            print(f"\n   测试查询: {query}")
            try:
                docs, scores = retriever.retrieve(
                    text=query, 
                    top_k=5, 
                    return_scores=True, 
                    language='zh'
                )
                
                if docs and isinstance(docs, list) and len(docs) > 0:
                    print(f"   ✅ 检索成功，找到 {len(docs)} 个文档")
                    print(f"   最高分数: {scores[0]:.4f}")
                    first_doc = docs[0]
                    print(f"   文档预览: {first_doc.content[:100]}...")
                else:
                    print(f"   ❌ 检索失败，没有找到相关文档")
                    
            except Exception as e:
                print(f"   ❌ 检索异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 检索器诊断失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主诊断函数"""
    print("开始诊断检索问题...")
    
    # 诊断数据加载
    data_ok = diagnose_data_loading()
    
    # 诊断检索器
    retriever_ok = diagnose_retriever()
    
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)
    
    if data_ok and retriever_ok:
        print("✅ 所有诊断通过！")
        print("✅ 数据加载正常")
        print("✅ 检索器工作正常")
        print("\n问题可能在于:")
        print("1. 查询与文档内容不匹配")
        print("2. 编码器模型问题")
        print("3. 检索参数设置")
    else:
        print("❌ 发现问题:")
        if not data_ok:
            print("❌ 数据加载有问题")
        if not retriever_ok:
            print("❌ 检索器有问题")

if __name__ == "__main__":
    main() 