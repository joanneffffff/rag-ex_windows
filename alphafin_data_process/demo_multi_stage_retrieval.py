#!/usr/bin/env python3
"""
多阶段检索系统演示
展示如何与现有的RAG系统集成
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

def demo_chinese_data():
    """演示中文数据（AlphaFin）的多阶段检索"""
    print("=" * 60)
    print("中文数据（AlphaFin）多阶段检索演示")
    print("=" * 60)
    
    try:
        # 导入多阶段检索系统
        from multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 数据文件路径
        data_path = Path("../data/alphafin/alphafin_merged_generated_qa.json")
        
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            return
        
        print("📊 初始化中文数据检索系统...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path, 
            dataset_type="chinese"
        )
        
        print("\n🔍 演示1: 基于元数据的预过滤 + FAISS + Qwen重排序")
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
            print(f"     分数: {result['combined_score']:.4f}")
            print(f"     摘要: {result['summary'][:100]}...")
            print()
        
        print("\n🔍 演示2: 通用检索（无元数据过滤）")
        print("查询: '钢铁行业发展趋势'")
        
        results2 = retrieval_system.search(
            query="钢铁行业发展趋势",
            top_k=3
        )
        
        print(f"\n找到 {len(results2)} 条结果:")
        for i, result in enumerate(results2):
            print(f"  {i+1}. {result['company_name']} ({result['stock_code']})")
            print(f"     分数: {result['combined_score']:.4f}")
            print(f"     摘要: {result['summary'][:100]}...")
            print()
        
        print("✅ 中文数据多阶段检索演示完成！")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()

def demo_english_data():
    """演示英文数据（TatQA）的检索（无元数据支持）"""
    print("\n" + "=" * 60)
    print("英文数据（TatQA）检索演示")
    print("=" * 60)
    
    print("📝 说明：TatQA数据没有元数据字段，因此不支持元数据预过滤")
    print("📝 检索流程：FAISS + Qwen重排序")
    
    try:
        # 检查tatqa数据文件
        tatqa_path = Path("../data/tatqa_dataset_raw/tatqa_dataset_train.json")
        
        if not tatqa_path.exists():
            print(f"❌ TatQA数据文件不存在: {tatqa_path}")
            print("注意：原始TatQA数据需要预处理才能使用多阶段检索系统")
            return
        
        print(f"✅ 找到TatQA数据文件: {tatqa_path}")
        print("注意：原始TatQA数据格式与AlphaFin不同，需要预处理")
        print("建议：使用现有的OptimizedDataLoader处理TatQA数据")
        
        # 这里可以添加tatqa数据预处理的示例
        print("\n📋 TatQA数据预处理建议:")
        print("1. 使用xlm.utils.optimized_data_loader.OptimizedDataLoader")
        print("2. 将table和paragraphs转换为标准格式")
        print("3. 生成context字段用于检索")
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")

def demo_integration_with_existing_system():
    """演示如何与现有系统集成"""
    print("\n" + "=" * 60)
    print("与现有系统集成演示")
    print("=" * 60)
    
    print("🔗 多阶段检索系统可以与现有的RAG系统集成:")
    print()
    print("1. 现有系统架构:")
    print("   - run_optimized_ui.py (主入口)")
    print("   - xlm/ui/optimized_rag_ui.py (UI界面)")
    print("   - xlm/components/retriever/reranker.py (QwenReranker)")
    print("   - xlm/utils/optimized_data_loader.py (数据加载)")
    print()
    print("2. 多阶段检索系统:")
    print("   - 预过滤：基于元数据（仅中文数据）")
    print("   - FAISS检索：基于嵌入向量")
    print("   - Qwen重排序：使用Qwen3-0.6B")
    print()
    print("3. 集成方式:")
    print("   - 替换现有的检索逻辑")
    print("   - 保持UI界面不变")
    print("   - 支持中英文双语检索")
    print()
    print("4. 优势:")
    print("   - 中文数据：元数据预过滤提高精度")
    print("   - 英文数据：纯向量检索保持灵活性")
    print("   - 统一的重排序：Qwen3-0.6B")
    print("   - 与现有系统兼容")

def main():
    """主函数"""
    print("🚀 多阶段检索系统演示")
    print("展示如何与现有的RAG系统集成")
    print()
    
    # 演示中文数据
    demo_chinese_data()
    
    # 演示英文数据
    demo_english_data()
    
    # 演示集成
    demo_integration_with_existing_system()
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print()
    print("📋 总结:")
    print("- 中文数据（AlphaFin）：支持完整的多阶段检索")
    print("- 英文数据（TatQA）：支持FAISS + Qwen重排序")
    print("- 与现有系统完全兼容")
    print("- 可以替换现有的检索逻辑")

if __name__ == "__main__":
    main() 