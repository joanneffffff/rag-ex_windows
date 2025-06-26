#!/usr/bin/env python3
"""
强制使用新Chunk逻辑的RAG UI系统
强制重新计算embedding，使用文档级别chunking
"""

import sys
import os
from pathlib import Path
from config.parameters import Config

sys.path.append(str(Path(__file__).parent))

def main():
    """启动强制使用新chunk逻辑的UI"""
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio未安装，正在安装...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
        print("✅ Gradio安装完成")
    
    print("=" * 60)
    print("强制使用新Chunk逻辑的RAG UI系统")
    print("=" * 60)
    print("⚠️  强制重新计算embedding，使用文档级别chunking")
    print("⚠️  这将删除旧的embedding缓存")
    print("=" * 60)
    
    # 删除旧的embedding缓存
    print("\n清理旧的embedding缓存...")
    config = Config()
    cache_dir = config.encoder.cache_dir
    
    # 查找并删除旧的embedding文件
    import glob
    old_embeddings = glob.glob(os.path.join(cache_dir, "*finetuned_alphafin_zh*.npy"))
    old_faiss = glob.glob(os.path.join(cache_dir, "*finetuned_alphafin_zh*.faiss"))
    
    for file in old_embeddings + old_faiss:
        try:
            os.remove(file)
            print(f"删除: {file}")
        except Exception as e:
            print(f"删除失败: {file} - {e}")
    
    print("✅ 旧缓存清理完成")
    
    # 启动UI
    print("\n启动Gradio UI...")
    from xlm.ui.optimized_rag_ui import OptimizedRagUI
    
    ui = OptimizedRagUI(
        cache_dir=config.cache_dir,
        use_faiss=True,
        enable_reranker=True,
        use_existing_embedding_index=False,  # 强制重新计算
        window_title="Financial RAG System (Force New Chunking)",
        title="Financial RAG System (Force New Chunking)",
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