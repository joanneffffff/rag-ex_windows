#!/usr/bin/env python3
"""
测试多阶段检索系统
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def test_multi_stage_retrieval():
    """测试多阶段检索系统"""
    try:
        # 导入多阶段检索系统
        from multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 数据文件路径 - 使用正确的路径
        data_path = Path("../data/alphafin/alphafin_merged_generated_qa.json")
        
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            print("请确保数据文件存在")
            return
        
        print("✅ 开始测试多阶段检索系统...")
        print(f"数据文件: {data_path}")
        
        # 初始化检索系统（中文数据）
        print("\n📊 初始化检索系统...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path, 
            dataset_type="chinese"
        )
        
        # 测试检索功能
        print("\n🔍 测试检索功能...")
        
        # 测试1：基于公司名称的检索
        print("\n测试1: 基于公司名称的检索")
        results1 = retrieval_system.search(
            query="公司业绩表现如何？",
            company_name="中国宝武",
            top_k=3
        )
        
        print(f"找到 {len(results1)} 条结果:")
        for i, result in enumerate(results1):
            print(f"  {i+1}. {result['company_name']} ({result['stock_code']}) - 分数: {result['combined_score']:.4f}")
        
        # 测试2：通用检索
        print("\n测试2: 通用检索")
        results2 = retrieval_system.search(
            query="钢铁行业发展趋势",
            top_k=3
        )
        
        print(f"找到 {len(results2)} 条结果:")
        for i, result in enumerate(results2):
            print(f"  {i+1}. {result['company_name']} ({result['stock_code']}) - 分数: {result['combined_score']:.4f}")
        
        # 测试3：英文查询（应该也能工作）
        print("\n测试3: 英文查询")
        results3 = retrieval_system.search(
            query="steel industry development",
            top_k=3
        )
        
        print(f"找到 {len(results3)} 条结果:")
        for i, result in enumerate(results3):
            print(f"  {i+1}. {result['company_name']} ({result['stock_code']}) - 分数: {result['combined_score']:.4f}")
        
        print("\n✅ 多阶段检索系统测试完成！")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保安装了必要的依赖:")
        print("pip install faiss-cpu sentence-transformers torch")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_english_dataset():
    """测试英文数据集（如果有的话）"""
    try:
        # 检查是否有英文数据文件 - 使用原始tatqa数据
        english_data_path = Path("../data/tatqa_dataset_raw/tatqa_dataset_train.json")
        
        if not english_data_path.exists():
            print(f"❌ 英文数据文件不存在: {english_data_path}")
            return
        
        print("\n🌍 测试英文数据集...")
        print("注意：原始tatqa数据格式可能不同，需要预处理")
        
        # 这里可以添加tatqa数据的预处理逻辑
        print("tatqa数据需要先转换为标准格式才能使用多阶段检索系统")
        
    except Exception as e:
        print(f"❌ 英文数据集测试失败: {e}")

def main():
    """主函数"""
    print("🚀 多阶段检索系统测试")
    print("=" * 50)
    
    # 测试中文数据集
    test_multi_stage_retrieval()
    
    # 测试英文数据集
    test_english_dataset()
    
    print("\n" + "=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    main() 