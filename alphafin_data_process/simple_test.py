#!/usr/bin/env python3
"""
简单的多阶段检索系统测试
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def simple_test():
    """简单测试"""
    print("🚀 简单测试多阶段检索系统")
    print("=" * 50)
    
    try:
        # 检查配置
        from config.parameters import Config, DEFAULT_CACHE_DIR
        config = Config()
        
        print(f"✅ 配置加载成功")
        print(f"  默认缓存目录: {DEFAULT_CACHE_DIR}")
        print(f"  中文编码器: {config.encoder.chinese_model_path}")
        print(f"  英文编码器: {config.encoder.english_model_path}")
        print(f"  重排序器: {config.reranker.model_name}")
        print()
        
        # 检查数据文件
        data_path = Path("../data/alphafin/alphafin_merged_generated_qa.json")
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            return
        
        print(f"✅ 数据文件存在: {data_path}")
        
        # 尝试导入多阶段检索系统
        try:
            from multi_stage_retrieval_final import MultiStageRetrievalSystem
            print("✅ 多阶段检索系统导入成功")
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            print("请确保安装了必要的依赖:")
            print("pip install faiss-cpu sentence-transformers torch")
            return
        
        print("\n📊 系统准备就绪，可以开始测试！")
        print("使用方法:")
        print("1. 运行: python test_with_existing_config.py")
        print("2. 或者直接运行: python demo_multi_stage_retrieval.py")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test() 