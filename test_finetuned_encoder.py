#!/usr/bin/env python3
"""
测试微调模型加载
验证FinbertEncoder是否能正确加载微调模型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_finetuned_encoder():
    """测试微调模型加载"""
    print("=" * 60)
    print("测试微调模型加载")
    print("=" * 60)
    
    # 1. 测试中文微调模型
    print("\n1. 测试中文微调模型:")
    print("-" * 40)
    try:
        from xlm.components.encoder.finbert import FinbertEncoder
        from config.parameters import Config
        
        config = Config()
        chinese_model_path = config.encoder.chinese_model_path
        
        print(f"中文模型路径: {chinese_model_path}")
        print(f"绝对路径: {Path(chinese_model_path).absolute()}")
        
        encoder = FinbertEncoder(
            model_name=chinese_model_path,
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        print("✓ 中文微调模型加载成功")
        
        # 测试编码
        test_texts = ["这是一个测试文本", "金融数据分析"]
        embeddings = encoder.encode(test_texts)
        print(f"✓ 编码测试成功，维度: {embeddings.shape}")
        print(f"✓ 嵌入维度: {encoder.get_embedding_dimension()}")
        
    except Exception as e:
        print(f"✗ 中文微调模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. 测试英文微调模型
    print("\n2. 测试英文微调模型:")
    print("-" * 40)
    try:
        english_model_path = config.encoder.english_model_path
        
        print(f"英文模型路径: {english_model_path}")
        print(f"绝对路径: {Path(english_model_path).absolute()}")
        
        encoder = FinbertEncoder(
            model_name=english_model_path,
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        print("✓ 英文微调模型加载成功")
        
        # 测试编码
        test_texts = ["This is a test text", "Financial data analysis"]
        embeddings = encoder.encode(test_texts)
        print(f"✓ 编码测试成功，维度: {embeddings.shape}")
        print(f"✓ 嵌入维度: {encoder.get_embedding_dimension()}")
        
    except Exception as e:
        print(f"✗ 英文微调模型加载失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. 测试多阶段检索系统
    print("\n3. 测试多阶段检索系统:")
    print("-" * 40)
    try:
        from alphafin_data_process.multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        system = MultiStageRetrievalSystem(
            data_path=Path("data/alphafin/alphafin_merged_generated_qa.json"),
            dataset_type="chinese",
            use_existing_config=True
        )
        print("✓ 多阶段检索系统初始化成功")
        
        # 测试搜索
        result = system.search("什么是股票投资？")
        print(f"✓ 搜索测试成功，返回 {len(result.get('results', []))} 个结果")
        
    except Exception as e:
        print(f"✗ 多阶段检索系统测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_finetuned_encoder() 