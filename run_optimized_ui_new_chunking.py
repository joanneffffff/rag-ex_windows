#!/usr/bin/env python3
"""
启动使用新Chunk逻辑的Gradio RAG UI系统
测试文档级别chunking对中文数据的效果
"""

import sys
import os
from pathlib import Path
from config.parameters import Config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """启动使用新chunk逻辑的Gradio UI"""
    try:
        # 检查gradio是否安装
        try:
            import gradio as gr
        except ImportError:
            print("❌ Gradio未安装，正在安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
            print("✅ Gradio安装完成")
        
        print("=" * 60)
        print("启动新Chunk逻辑的RAG UI系统")
        print("=" * 60)
        print("Step 1. 使用文档级别chunking处理中文数据...")
        print("Step 2. 保持传统chunk级别处理英文数据...")
        print("Step 3. 启动Gradio UI...")
        print("Access URL: http://localhost:7860")
        print("Press Ctrl+C to stop server")
        print("=" * 60)
        
        # 使用config中的平台感知配置
        config = Config()
        
        # 首先测试新的数据加载器
        print("\n测试新chunk逻辑的数据加载...")
        try:
            from xlm.utils.optimized_data_loader import OptimizedDataLoader
            
            # 测试文档级别chunking
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
            
            # 测试传统chunk级别chunking
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
                print("✅ 新chunk逻辑测试成功！")
            
        except Exception as e:
            print(f"❌ 新chunk逻辑测试失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 导入并启动UI
        print("\n启动Gradio UI...")
        from xlm.ui.optimized_rag_ui import OptimizedRagUI
        
        # 创建UI实例，使用专门测试新chunk逻辑的示例问题
        ui = OptimizedRagUI(
            cache_dir=config.cache_dir,
            use_faiss=True,
            enable_reranker=True,
            window_title="Financial RAG System (New Chunking Logic)",
            title="Financial RAG System (New Chunking Logic)",
            examples=[
                # 中文问题 - 测试文档级别chunking效果
                ["什么是市盈率？"],
                ["如何计算ROE？"],
                ["财务报表包括哪些内容？"],
                ["什么是资产负债表？"],
                ["现金流量表的作用是什么？"],
                ["市盈率的计算公式是什么？"],
                ["ROE和ROA有什么区别？"],
                ["如何分析公司的财务状况？"],
                ["什么是流动比率？"],
                ["净利润和毛利润的区别是什么？"],
                # 英文问题 - 保持原有逻辑
                ["How was internally developed software capitalised?"],
                ["Why did the Operating revenues decreased from 2018 to 2019?"],
                ["Why did the Operating costs decreased from 2018 to 2019?"],
                ["What is the difference between revenue and profit?"],
                ["How to calculate return on investment?"]
            ]
        )
        
        print("\n" + "=" * 60)
        print("UI启动成功！")
        print("=" * 60)
        print("测试建议:")
        print("1. 尝试中文问题，观察检索结果的质量和连贯性")
        print("2. 对比英文问题的处理效果")
        print("3. 注意检索速度的提升")
        print("4. 观察答案的完整性和准确性")
        print("=" * 60)
        
        # 启动UI
        ui.launch(share=False)
        
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()

def test_chunking_comparison():
    """测试chunking策略对比"""
    print("\n" + "=" * 60)
    print("Chunking策略对比测试")
    print("=" * 60)
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        # 测试不同策略
        strategies = [
            ("文档级别chunking", True, True),
            ("传统chunk级别chunking", False, True),
        ]
        
        results = {}
        
        for name, chinese_doc_level, english_chunk_level in strategies:
            print(f"\n测试 {name}:")
            loader = OptimizedDataLoader(
                data_dir="data",
                max_samples=500,
                chinese_document_level=chinese_doc_level,
                english_chunk_level=english_chunk_level
            )
            
            stats = loader.get_statistics()
            results[name] = stats
            
            print(f"   中文文档数: {stats['chinese_docs']}")
            print(f"   英文文档数: {stats['english_docs']}")
            print(f"   中文平均长度: {stats['chinese_avg_length']:.2f}")
            print(f"   英文平均长度: {stats['english_avg_length']:.2f}")
        
        # 计算改进效果
        if len(results) >= 2:
            doc_level = results["文档级别chunking"]
            chunk_level = results["传统chunk级别chunking"]
            
            if chunk_level['chinese_docs'] > 0:
                reduction = (chunk_level['chinese_docs'] - doc_level['chinese_docs']) / chunk_level['chinese_docs'] * 100
                print(f"\n改进效果:")
                print(f"   Chunk数量减少: {reduction:.1f}%")
                print(f"   文档级别: {doc_level['chinese_docs']} chunks")
                print(f"   传统chunking: {chunk_level['chinese_docs']} chunks")
        
    except Exception as e:
        print(f"对比测试失败: {e}")

if __name__ == "__main__":
    # 先进行对比测试
    test_chunking_comparison()
    
    # 然后启动UI
    main() 