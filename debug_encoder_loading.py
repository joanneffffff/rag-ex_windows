#!/usr/bin/env python3
"""
调试编码器加载问题
检查为什么微调模型路径没有正确加载
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def debug_encoder_loading():
    """调试编码器加载问题"""
    print("=" * 60)
    print("调试编码器加载问题")
    print("=" * 60)
    
    # 1. 检查配置文件
    print("\n1. 检查配置文件:")
    print("-" * 40)
    try:
        from xlm.config.config import Config
        config = Config()
        print(f"✓ 配置文件加载成功")
        print(f"  中文模型路径: {config.encoder.chinese_model_path}")
        print(f"  英文模型路径: {config.encoder.english_model_path}")
        print(f"  缓存目录: {config.encoder.cache_dir}")
    except Exception as e:
        print(f"✗ 配置文件加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 检查模型路径是否存在
    print("\n2. 检查模型路径:")
    print("-" * 40)
    chinese_path = Path(config.encoder.chinese_model_path)
    english_path = Path(config.encoder.english_model_path)
    
    print(f"中文模型路径: {chinese_path}")
    print(f"  绝对路径: {chinese_path.absolute()}")
    print(f"  存在: {chinese_path.exists()}")
    if chinese_path.exists():
        print(f"  内容: {list(chinese_path.iterdir())}")
    
    print(f"英文模型路径: {english_path}")
    print(f"  绝对路径: {english_path.absolute()}")
    print(f"  存在: {english_path.exists()}")
    if english_path.exists():
        print(f"  内容: {list(english_path.iterdir())}")
    
    # 3. 尝试直接加载微调模型
    print("\n3. 尝试直接加载微调模型:")
    print("-" * 40)
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"尝试加载中文微调模型: {chinese_path}")
        chinese_model = SentenceTransformer(str(chinese_path.absolute()), cache_folder=config.encoder.cache_dir)
        print(f"✓ 中文微调模型加载成功")
        
        # 测试编码
        test_text = "这是一个测试文本"
        embedding = chinese_model.encode(test_text)
        print(f"✓ 编码测试成功，维度: {embedding.shape}")
        
    except Exception as e:
        print(f"✗ 中文微调模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 检查SentenceTransformer是否支持本地路径
    print("\n4. 检查SentenceTransformer支持:")
    print("-" * 40)
    try:
        from sentence_transformers import SentenceTransformer
        
        # 测试HuggingFace模型
        print("测试HuggingFace模型加载...")
        test_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✓ HuggingFace模型加载成功")
        
        # 测试本地路径格式
        print(f"测试本地路径格式: {chinese_path}")
        if chinese_path.exists():
            # 检查是否有必要的文件
            required_files = ["config.json", "pytorch_model.bin", "sentence_bert_config.json"]
            for file in required_files:
                file_path = chinese_path / file
                print(f"  {file}: {file_path.exists()}")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_encoder_loading() 