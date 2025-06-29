#!/usr/bin/env python3
"""
启动集成RAG UI系统 - 结合多阶段检索和传统RAG
中文查询：使用AlphaFin多阶段检索
英文查询：使用传统RAG系统
"""

import sys
import os
from pathlib import Path
from config.parameters import Config

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """启动集成RAG UI系统"""
    try:
        # 检查gradio是否安装
        try:
            import gradio as gr
        except ImportError:
            print("❌ Gradio未安装，正在安装...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio"])
            print("✅ Gradio安装完成")
        
        print("🚀 启动集成RAG UI系统...")
        print("Access URL: http://localhost:7860")
        print("Press Ctrl+C to stop server")
        
        # 使用config中的平台感知配置
        config = Config()
        
        # 导入并启动集成UI
        from xlm.ui.optimized_rag_ui import OptimizedRagUI
        
        # 创建UI实例，使用集成版本
        ui = OptimizedRagUI(
            cache_dir=config.cache_dir,
            use_faiss=True,
            enable_reranker=True,
            window_title="Financial Explainable RAG System",
            title="Financial Explainable RAG System",
            examples=[
                ["德赛电池(000049)的下一季度收益预测如何？"],
                ["用友网络2019年的每股经营活动产生的现金流量净额是多少？"],
                ["首钢股份的业绩表现如何？"],
                ["钢铁行业发展趋势"],
                ["How was internally developed software capitalised?"],
                ["Why did the Operating revenues decreased from 2018 to 2019?"],
                ["What are the key financial ratios for investment analysis?"]
            ]
        )
        
        # 启动UI
        ui.launch(share=False)
        
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Startup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 