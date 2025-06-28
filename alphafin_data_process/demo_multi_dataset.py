#!/usr/bin/env python3
"""
多数据集检索系统演示脚本
展示如何处理中文（alphafin）和英文（tatqa）数据集
"""

import json
from pathlib import Path
from multi_stage_retrieval_final import MultiStageRetrievalSystem

def demo_chinese_dataset():
    """演示中文数据集（alphafin）的检索"""
    print("="*60)
    print("中文数据集（AlphaFin）检索演示")
    print("="*60)
    
    # 数据文件路径
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    index_dir = Path("data/alphafin/retrieval_index")
    
    # 初始化检索系统（中文数据）
    print("正在初始化中文数据集检索系统...")
    retrieval_system = MultiStageRetrievalSystem(data_path, dataset_type="chinese")
    
    # 保存索引
    retrieval_system.save_index(index_dir)
    
    # 演示检索
    print("\n--- 示例1: 基于公司名称的检索 ---")
    results1 = retrieval_system.search(
        query="公司业绩表现如何？",
        company_name="中国宝武",
        top_k=3
    )
    
    for i, result in enumerate(results1):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")
    
    print("\n--- 示例2: 通用检索（无元数据过滤）---")
    results2 = retrieval_system.search(
        query="钢铁行业发展趋势",
        top_k=3
    )
    
    for i, result in enumerate(results2):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")

def demo_english_dataset():
    """演示英文数据集（tatqa）的检索"""
    print("\n" + "="*60)
    print("英文数据集（TAT-QA）检索演示")
    print("="*60)
    
    # 检查tatqa数据是否存在
    tatqa_data_path = Path("data/tatqa_dataset_raw/tatqa_dataset_raw.json")
    if not tatqa_data_path.exists():
        print(f"TAT-QA数据文件不存在: {tatqa_data_path}")
        print("请确保TAT-QA数据已正确放置在data/tatqa_dataset_raw/目录下")
        return
    
    # 数据文件路径
    index_dir = Path("data/tatqa_dataset_raw/retrieval_index")
    
    # 初始化检索系统（英文数据）
    print("正在初始化英文数据集检索系统...")
    retrieval_system = MultiStageRetrievalSystem(tatqa_data_path, dataset_type="english")
    
    # 保存索引
    retrieval_system.save_index(index_dir)
    
    # 演示检索
    print("\n--- 示例1: 财务数据查询 ---")
    results1 = retrieval_system.search(
        query="What is the revenue growth?",
        top_k=3
    )
    
    for i, result in enumerate(results1):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")
    
    print("\n--- 示例2: 利润相关查询 ---")
    results2 = retrieval_system.search(
        query="profit margin analysis",
        top_k=3
    )
    
    for i, result in enumerate(results2):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")

def compare_encoders():
    """比较不同编码器的效果"""
    print("\n" + "="*60)
    print("编码器比较")
    print("="*60)
    
    # 中文数据
    chinese_data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    if chinese_data_path.exists():
        print("\n--- 中文数据编码器比较 ---")
        
        # 多语言编码器
        print("1. 多语言编码器 (distiluse-base-multilingual-cased-v2)")
        try:
            retrieval_system_multilingual = MultiStageRetrievalSystem(
                chinese_data_path, 
                dataset_type="chinese",
                model_name="distiluse-base-multilingual-cased-v2"
            )
            results_multilingual = retrieval_system_multilingual.search(
                query="公司业绩表现",
                top_k=2
            )
            print(f"   检索结果数量: {len(results_multilingual)}")
        except Exception as e:
            print(f"   错误: {e}")
        
        # 英文编码器（用于对比）
        print("2. 英文编码器 (all-MiniLM-L6-v2)")
        try:
            retrieval_system_english = MultiStageRetrievalSystem(
                chinese_data_path, 
                dataset_type="chinese",
                model_name="all-MiniLM-L6-v2"
            )
            results_english = retrieval_system_english.search(
                query="公司业绩表现",
                top_k=2
            )
            print(f"   检索结果数量: {len(results_english)}")
        except Exception as e:
            print(f"   错误: {e}")

def main():
    """主函数"""
    print("多数据集检索系统演示")
    print("支持中文（AlphaFin）和英文（TAT-QA）数据集")
    
    # 演示中文数据集
    demo_chinese_dataset()
    
    # 演示英文数据集
    demo_english_dataset()
    
    # 比较编码器
    compare_encoders()
    
    print("\n" + "="*60)
    print("演示完成")
    print("="*60)
    print("\n系统特点总结:")
    print("1. 中文数据集 (AlphaFin):")
    print("   - 使用多语言编码器: distiluse-base-multilingual-cased-v2")
    print("   - 支持元数据预过滤（公司名称、股票代码、报告日期）")
    print("   - 中文分词处理")
    print("2. 英文数据集 (TAT-QA):")
    print("   - 使用英文编码器: all-MiniLM-L6-v2")
    print("   - 不支持元数据预过滤")
    print("   - 英文分词处理")
    print("3. 通用功能:")
    print("   - FAISS向量检索")
    print("   - BM25重排序")
    print("   - 混合评分机制")

if __name__ == '__main__':
    main() 