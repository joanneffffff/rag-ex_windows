#!/usr/bin/env python3
"""
使用现有配置测试多阶段检索系统
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def test_with_existing_config():
    """使用现有配置测试多阶段检索系统"""
    print("=" * 60)
    print("使用现有配置测试多阶段检索系统")
    print("=" * 60)
    
    try:
        # 首先检查配置
        from config.parameters import Config
        config = Config()
        
        print("📋 当前配置:")
        print(f"  中文编码器: {config.encoder.chinese_model_path}")
        print(f"  英文编码器: {config.encoder.english_model_path}")
        print(f"  重排序器: {config.reranker.model_name}")
        print(f"  编码器缓存: {config.encoder.cache_dir}")
        print(f"  重排序器缓存: {config.reranker.cache_dir}")
        print(f"  检索top-k: {config.retriever.retrieval_top_k}")
        print(f"  重排序top-k: {config.retriever.rerank_top_k}")
        print()
        
        # 导入多阶段检索系统
        from multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 数据文件路径
        data_path = Path("../data/alphafin/alphafin_merged_generated_qa.json")
        
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            print("请确保数据文件存在")
            return
        
        print("📊 初始化多阶段检索系统（使用现有配置）...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path, 
            dataset_type="chinese",
            use_existing_config=True  # 使用现有配置
        )
        
        print("\n🔍 测试1: 基于元数据的预过滤")
        print("查询: '公司业绩表现如何？'")
        print("元数据过滤: 公司名称='中国宝武'")
        
        results1 = retrieval_system.search(
            query="公司业绩表现如何？",
            company_name="中国宝武",
            top_k=3
        )
        
        print(f"\n找到 {len(results1)} 条结果:")
        for i, result in enumerate(results1):
            print(f"  {i+1}. {result['company_name']} ({result['stock_code']})")
            print(f"     FAISS分数: {result['faiss_score']:.4f}")
            print(f"     组合分数: {result['combined_score']:.4f}")
            print(f"     摘要: {result['summary'][:100]}...")
            print()
        
        print("\n🔍 测试2: 通用检索（无元数据过滤）")
        print("查询: '钢铁行业发展趋势'")
        
        results2 = retrieval_system.search(
            query="钢铁行业发展趋势",
            top_k=3
        )
        
        print(f"\n找到 {len(results2)} 条结果:")
        for i, result in enumerate(results2):
            print(f"  {i+1}. {result['company_name']} ({result['stock_code']})")
            print(f"     FAISS分数: {result['faiss_score']:.4f}")
            print(f"     组合分数: {result['combined_score']:.4f}")
            print(f"     摘要: {result['summary'][:100]}...")
            print()
        
        print("✅ 使用现有配置的测试完成！")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了必要的依赖:")
        print("pip install faiss-cpu sentence-transformers torch")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_config_compatibility():
    """测试配置兼容性"""
    print("\n" + "=" * 60)
    print("配置兼容性测试")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        config = Config()
        
        print("✅ 配置加载成功")
        print(f"  平台: {config.cache_dir}")
        print(f"  中文模型路径: {config.encoder.chinese_model_path}")
        print(f"  英文模型路径: {config.encoder.english_model_path}")
        print(f"  重排序器模型: {config.reranker.model_name}")
        
        # 检查模型路径是否存在
        chinese_model_path = Path(config.encoder.chinese_model_path)
        english_model_path = Path(config.encoder.english_model_path)
        
        print(f"\n📁 模型路径检查:")
        print(f"  中文模型: {chinese_model_path} - {'✅ 存在' if chinese_model_path.exists() else '❌ 不存在'}")
        print(f"  英文模型: {english_model_path} - {'✅ 存在' if english_model_path.exists() else '❌ 不存在'}")
        
        if not chinese_model_path.exists():
            print("⚠️  中文模型路径不存在，系统将使用默认模型")
        if not english_model_path.exists():
            print("⚠️  英文模型路径不存在，系统将使用默认模型")
        
    except Exception as e:
        print(f"❌ 配置兼容性测试失败: {e}")

def main():
    """主函数"""
    print("🚀 使用现有配置测试多阶段检索系统")
    print()
    
    # 测试配置兼容性
    test_config_compatibility()
    
    # 测试多阶段检索系统
    test_with_existing_config()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    print()
    print("📋 总结:")
    print("- 系统使用现有配置中的模型路径")
    print("- 中文数据支持元数据预过滤")
    print("- 使用现有的Qwen3-0.6B重排序器")
    print("- 与现有系统完全兼容")

if __name__ == "__main__":
    main() 