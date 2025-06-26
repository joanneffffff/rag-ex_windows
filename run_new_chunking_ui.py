#!/usr/bin/env python3
"""
启动使用新Chunk逻辑的RAG UI系统
"""

import sys
import os
from pathlib import Path
from config.parameters import Config

sys.path.append(str(Path(__file__).parent))

def main():
    """启动使用新chunk逻辑的UI"""
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
    
    # 测试新chunk逻辑
    print("测试新chunk逻辑...")
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        # 文档级别chunking
        loader_doc = OptimizedDataLoader(
            data_dir="data",
            max_samples=1000,
            chinese_document_level=True,
            english_chunk_level=True
        )
        
        doc_stats = loader_doc.get_statistics()
        print(f"文档级别chunking: {doc_stats['chinese_docs']} 中文chunks")
        
        # 传统chunk级别chunking
        loader_chunk = OptimizedDataLoader(
            data_dir="data",
            max_samples=1000,
            chinese_document_level=False,
            english_chunk_level=True
        )
        
        chunk_stats = loader_chunk.get_statistics()
        print(f"传统chunk级别chunking: {chunk_stats['chinese_docs']} 中文chunks")
        
        # 计算改进效果
        if chunk_stats['chinese_docs'] > 0:
            reduction = (chunk_stats['chinese_docs'] - doc_stats['chinese_docs']) / chunk_stats['chinese_docs'] * 100
            print(f"Chunk数量减少: {reduction:.1f}%")
        
        print("✅ 新chunk逻辑测试成功！")
        
    except Exception as e:
        print(f"❌ 新chunk逻辑测试失败: {e}")
        return
    
    # 启动UI
    print("\n启动Gradio UI...")
    config = Config()
    
    from xlm.ui.optimized_rag_ui import OptimizedRagUI
    
    ui = OptimizedRagUI(
        cache_dir=config.cache_dir,
        use_faiss=True,
        enable_reranker=True,
        window_title="Financial RAG System (New Chunking)",
        title="Financial RAG System (New Chunking)",
        examples=[
            ["什么是市盈率？"],
            ["如何计算ROE？"],
            ["财务报表包括哪些内容？"],
            ["什么是资产负债表？"],
            ["现金流量表的作用是什么？"],
            ["How was internally developed software capitalised?"],
            ["Why did the Operating revenues decreased from 2018 to 2019?"]
        ]
    )
    
    print("Access URL: http://localhost:7860")
    print("Press Ctrl+C to stop server")
    
    ui.launch(share=False)
    
if __name__ == "__main__":
    main() 