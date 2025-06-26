#!/usr/bin/env python3
"""
启动Gradio RAG UI系统
"""

import sys
import os
from pathlib import Path
from config.parameters import Config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """启动Gradio UI"""
    try:
        # 检查gradio是否安装
        try:
            import gradio as gr
        except ImportError:
            print("❌ Gradio未安装，正在安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
            print("✅ Gradio安装完成")
        
        print("🚀 启动Gradio RAG UI系统...")
        print("🌐 访问地址: http://localhost:7860")
        print("按 Ctrl+C 停止服务器")
        
        # 导入并启动UI
        from xlm.ui.optimized_rag_ui import OptimizedRagUI
        
        # 创建UI实例
        cache_dir = Config().cache_dir
        ui = OptimizedRagUI(
            encoder_model_name="paraphrase-multilingual-MiniLM-L12-v2",
            # generator_model_name 现在从config中读取
            cache_dir=cache_dir,
            # data_path 现在从config中读取
            use_faiss=True,
            enable_reranker=True,  # 启用reranker (将使用Qwen3-0.6B)
            window_title="Enhanced RAG Financial System",
            title="Enhanced RAG Financial System"
        )
        
        # 启动UI
        ui.launch(share=False)
        
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 