import os
import sys
import warnings
from pathlib import Path
from config.parameters import config

# 初始化NLTK数据
print("正在检查NLTK数据...")
from init_nltk import download_nltk_data
download_nltk_data()

from xlm.registry.generator import load_generator
from xlm.registry.rag_system import load_rag_system
from xlm.registry.retriever import load_retriever
from xlm.ui.rag_explainer_ui import RagExplainerUI
from xlm.utils.visualizer import Visualizer

# 忽略警告
warnings.filterwarnings("ignore")

def load_visualizer() -> Visualizer:
    """加载可视化工具"""
    visualizer = Visualizer(show_mid_features=True, show_low_features=True)
    return visualizer

def ensure_directories():
    """确保必要的目录存在"""
    dirs = [
        # "D:/AI/huggingface",
        "M:/huggingface",
        "data",
        "xlm/ui/images",
        "xlm/ui/css"
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def check_files():
    """检查必要的文件是否存在"""
    required_files = {
        "data/rise_of_ai.txt": "知识库文件",
        "xlm/ui/images/iais.svg": "Logo文件",
        "xlm/ui/css/demo.css": "CSS样式文件"
    }
    
    missing_files = []
    for file_path, description in required_files.items():
        if not os.path.exists(file_path):
            missing_files.append(f"- {description} ({file_path})")
    
    if missing_files:
        print("\n警告：以下必要文件缺失：")
        print("\n".join(missing_files))
        return False
    return True

if __name__ == "__main__":
    try:
        # 确保目录存在
        ensure_directories()
        
        # 检查文件
        if not check_files():
            print("\n请确保所有必要文件存在后再运行程序。")
            sys.exit(1)
        
        # 配置参数
        encoder_model_name = config.encoder.model_name
        generator_model_name = config.generator.model_name
        data_path = config.data.data_path if hasattr(config.data, 'data_path') else "data/rise_of_ai.txt"
        prompt_template = "Context: {context}\nQuestion: {question}\n\nAnswer:"
        cache_dir = config.generator.cache_dir
        
        print("正在启动 RAG-Ex 系统...")
        print(f"- 使用编码器: {encoder_model_name}")
        print(f"- 使用生成器: {generator_model_name}")
        print(f"- 数据文件: {data_path}")
        print(f"- 模型缓存目录: {cache_dir}")
        
        # 加载检索器
        print("\n1. 加载检索器...")
        retriever = load_retriever(
            encoder_model_name=encoder_model_name,
            data_path=data_path,
        )
        
        # 加载生成器(使用本地LLM)
        print("\n2. 加载生成器...")
        generator = load_generator(
            generator_model_name=generator_model_name,
            split_lines=False,
            use_local_llm=True,  # 启用本地LLM
            cache_dir=cache_dir
        )
        
        # 加载RAG系统
        print("\n3. 初始化RAG系统...")
        rag_system = load_rag_system(
            retriever=retriever,
            generator=generator,
            prompt_template=prompt_template
        )

        # 创建UI界面
        print("\n4. 创建Web界面...")
        interface = RagExplainerUI(
            logo_path="xlm/ui/images/iais.svg",
            css_path="xlm/ui/css/demo.css",
            visualizer=load_visualizer(),
            window_title="RAG-Ex 2.0 (Local LLM)",
            title="✳️ RAG-Ex 2.0: Towards Generic Explainability (Local LLM)",
            rag_system=rag_system,
        )
        
        # 启动应用
        app = interface.build_app()
        app.queue()

        platform = sys.platform
        print(f"\n5. 启动服务 (平台: {platform})...")
        host = "127.0.0.1" if "win" in platform else "0.0.0.0"
        port = 9985
        print(f"访问地址: http://{host}:{port}")
        app.launch(server_name=host, server_port=port)
        
    except Exception as e:
        print(f"\n错误：{str(e)}")
        print("\n如果是模块导入错误，请确保已安装所有依赖：")
        print("pip install -r requirements.txt")
        print("\n如果是其他错误，请检查：")
        print("1. 是否有足够的磁盘空间")
        print("2. 是否有写入权限到 D:/AI/huggingface 目录")
        sys.exit(1) 