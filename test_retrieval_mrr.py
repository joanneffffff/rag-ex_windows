#!/usr/bin/env python3
"""
测试检索质量 - MRR评估
使用evaluate_mrr/alphafin_eval.jsonl和tatqa_eval.jsonl作为测试集
支持两种模式：
1. 评估数据context加入知识库 - 测试真实检索能力
2. 评估数据context不加入知识库 - 避免数据泄露
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

# 导入必要的类型
from xlm.dto.dto import DocumentWithMetadata

def load_eval_data(eval_file: str) -> List[Dict[str, Any]]:
    """加载评估数据"""
    data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_mrr(ranks: List[int]) -> float:
    """计算MRR (Mean Reciprocal Rank)"""
    if not ranks:
        return 0.0
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
    return float(np.mean(reciprocal_ranks))

def calculate_hit_rate(ranks: List[int], k: int = 1) -> float:
    """计算Hit@k"""
    if not ranks:
        return 0.0
    hits = [1 if rank <= k and rank > 0 else 0 for rank in ranks]
    return float(np.mean(hits))

def test_retrieval_with_eval_context(include_eval_data: bool = True):
    """测试检索质量 - 可选择是否包含评估数据到知识库"""
    mode = "包含评估数据" if include_eval_data else "不包含评估数据"
    print("=" * 60)
    print(f"测试检索质量 - MRR评估 ({mode})")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("1. 加载编码器...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        print("   ✅ 编码器加载成功")
        
        print(f"\n2. 加载训练数据（知识库）...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,  # 加载所有数据
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=include_eval_data  # 控制是否包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 训练数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n3. 加载评估数据...")
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval.jsonl")
        
        print(f"   ✅ 评估数据加载成功:")
        print(f"      AlphaFin评估样本: {len(alphafin_eval)}")
        print(f"      TatQA评估样本: {len(tatqa_eval)}")
        
        print("\n4. 创建检索器...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,
            batch_size=8,
            cache_dir=config.encoder.cache_dir
        )
        print("   ✅ 检索器创建成功")
        
        print("\n5. 测试中文检索质量 (AlphaFin)...")
        chinese_ranks = []
        chinese_queries = []
        
        for i, sample in enumerate(tqdm(alphafin_eval[:100], desc="测试中文检索")):  # 测试前100个样本
            query = sample.get('question', '')
            context = sample.get('context', '')
            
            if not query or not context:
                continue
                
            # 检索相关文档
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=20, 
                return_scores=True, 
                language='zh'
            )
            
            # 处理返回值：可能是元组(documents, scores)或只是documents
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            # 检查正确答案是否在检索结果中
            found_rank = 0
            for rank, doc in enumerate(retrieved_docs, 1):
                if context in doc.content or doc.content in context:
                    found_rank = rank
                    break
            
            chinese_ranks.append(found_rank)
            chinese_queries.append(query)
            
            if i < 5:  # 显示前5个样本的详细信息
                print(f"   查询 {i+1}: {query[:50]}...")
                print(f"   找到位置: {found_rank}")
                if found_rank > 0:
                    print(f"   相关文档: {retrieved_docs[found_rank-1].content[:100]}...")
                print()
        
        print("\n6. 测试英文检索质量 (TatQA)...")
        english_ranks = []
        english_queries = []
        
        for i, sample in enumerate(tqdm(tatqa_eval[:100], desc="测试英文检索")):  # 测试前100个样本
            query = sample.get('question', '')
            context = sample.get('context', '')
            
            if not query or not context:
                continue
                
            # 检索相关文档
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=20, 
                return_scores=True, 
                language='en'
            )
            
            # 处理返回值：可能是元组(documents, scores)或只是documents
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            # 检查正确答案是否在检索结果中
            found_rank = 0
            for rank, doc in enumerate(retrieved_docs, 1):
                if context in doc.content or doc.content in context:
                    found_rank = rank
                    break
            
            english_ranks.append(found_rank)
            english_queries.append(query)
            
            if i < 5:  # 显示前5个样本的详细信息
                print(f"   Query {i+1}: {query[:50]}...")
                print(f"   Found at rank: {found_rank}")
                if found_rank > 0:
                    print(f"   Relevant doc: {retrieved_docs[found_rank-1].content[:100]}...")
                print()
        
        print("\n" + "=" * 60)
        print(f"检索质量评估结果 ({mode})")
        print("=" * 60)
        
        # 计算中文检索指标
        chinese_mrr = calculate_mrr(chinese_ranks)
        chinese_hit1 = calculate_hit_rate(chinese_ranks, k=1)
        chinese_hit5 = calculate_hit_rate(chinese_ranks, k=5)
        chinese_hit10 = calculate_hit_rate(chinese_ranks, k=10)
        
        print(f"中文检索 (AlphaFin):")
        print(f"  样本数: {len(chinese_ranks)}")
        print(f"  MRR: {chinese_mrr:.4f}")
        print(f"  Hit@1: {chinese_hit1:.4f}")
        print(f"  Hit@5: {chinese_hit5:.4f}")
        print(f"  Hit@10: {chinese_hit10:.4f}")
        
        # 计算英文检索指标
        english_mrr = calculate_mrr(english_ranks)
        english_hit1 = calculate_hit_rate(english_ranks, k=1)
        english_hit5 = calculate_hit_rate(english_ranks, k=5)
        english_hit10 = calculate_hit_rate(english_ranks, k=10)
        
        print(f"\n英文检索 (TatQA):")
        print(f"  样本数: {len(english_ranks)}")
        print(f"  MRR: {english_mrr:.4f}")
        print(f"  Hit@1: {english_hit1:.4f}")
        print(f"  Hit@5: {english_hit5:.4f}")
        print(f"  Hit@10: {english_hit10:.4f}")
        
        # 总体指标
        all_ranks = chinese_ranks + english_ranks
        overall_mrr = calculate_mrr(all_ranks)
        overall_hit1 = calculate_hit_rate(all_ranks, k=1)
        overall_hit5 = calculate_hit_rate(all_ranks, k=5)
        overall_hit10 = calculate_hit_rate(all_ranks, k=10)
        
        print(f"\n总体检索:")
        print(f"  样本数: {len(all_ranks)}")
        print(f"  MRR: {overall_mrr:.4f}")
        print(f"  Hit@1: {overall_hit1:.4f}")
        print(f"  Hit@5: {overall_hit5:.4f}")
        print(f"  Hit@10: {overall_hit10:.4f}")
        
        # 保存结果
        suffix = "with_eval" if include_eval_data else "without_eval"
        results = {
            "mode": mode,
            "chinese": {
                "mrr": chinese_mrr,
                "hit1": chinese_hit1,
                "hit5": chinese_hit5,
                "hit10": chinese_hit10,
                "sample_count": len(chinese_ranks)
            },
            "english": {
                "mrr": english_mrr,
                "hit1": english_hit1,
                "hit5": english_hit5,
                "hit10": english_hit10,
                "sample_count": len(english_ranks)
            },
            "overall": {
                "mrr": overall_mrr,
                "hit1": overall_hit1,
                "hit5": overall_hit5,
                "hit10": overall_hit10,
                "sample_count": len(all_ranks)
            }
        }
        
        filename = f"retrieval_mrr_results_{suffix}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {filename}")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_retrieval_modes():
    """对比两种检索模式的效果"""
    print("=" * 80)
    print("对比检索模式：包含评估数据 vs 不包含评估数据")
    print("=" * 80)
    
    # 测试包含评估数据的模式
    print("\n🔍 测试模式1：包含评估数据到知识库")
    results_with_eval = test_retrieval_with_eval_context(include_eval_data=True)
    
    print("\n" + "=" * 80)
    
    # 测试不包含评估数据的模式
    print("\n🔍 测试模式2：不包含评估数据到知识库")
    results_without_eval = test_retrieval_with_eval_context(include_eval_data=False)
    
    # 对比结果
    if results_with_eval and results_without_eval:
        print("\n" + "=" * 80)
        print("模式对比结果")
        print("=" * 80)
        
        print(f"{'指标':<15} {'包含评估数据':<15} {'不包含评估数据':<15} {'差异':<10}")
        print("-" * 60)
        
        # 中文对比
        ch_zh_mrr_diff = results_with_eval["chinese"]["mrr"] - results_without_eval["chinese"]["mrr"]
        ch_zh_hit1_diff = results_with_eval["chinese"]["hit1"] - results_without_eval["chinese"]["hit1"]
        
        print(f"{'中文MRR':<15} {results_with_eval['chinese']['mrr']:<15.4f} {results_without_eval['chinese']['mrr']:<15.4f} {ch_zh_mrr_diff:+.4f}")
        print(f"{'中文Hit@1':<15} {results_with_eval['chinese']['hit1']:<15.4f} {results_without_eval['chinese']['hit1']:<15.4f} {ch_zh_hit1_diff:+.4f}")
        
        # 英文对比
        ch_en_mrr_diff = results_with_eval["english"]["mrr"] - results_without_eval["english"]["mrr"]
        ch_en_hit1_diff = results_with_eval["english"]["hit1"] - results_without_eval["english"]["hit1"]
        
        print(f"{'英文MRR':<15} {results_with_eval['english']['mrr']:<15.4f} {results_without_eval['english']['mrr']:<15.4f} {ch_en_mrr_diff:+.4f}")
        print(f"{'英文Hit@1':<15} {results_with_eval['english']['hit1']:<15.4f} {results_without_eval['english']['hit1']:<15.4f} {ch_en_hit1_diff:+.4f}")
        
        # 总体对比
        ch_overall_mrr_diff = results_with_eval["overall"]["mrr"] - results_without_eval["overall"]["mrr"]
        ch_overall_hit1_diff = results_with_eval["overall"]["hit1"] - results_without_eval["overall"]["hit1"]
        
        print(f"{'总体MRR':<15} {results_with_eval['overall']['mrr']:<15.4f} {results_without_eval['overall']['mrr']:<15.4f} {ch_overall_mrr_diff:+.4f}")
        print(f"{'总体Hit@1':<15} {results_with_eval['overall']['hit1']:<15.4f} {results_without_eval['overall']['hit1']:<15.4f} {ch_overall_hit1_diff:+.4f}")
        
        # 保存对比结果
        comparison_results = {
            "with_eval_data": results_with_eval,
            "without_eval_data": results_without_eval,
            "differences": {
                "chinese_mrr_diff": ch_zh_mrr_diff,
                "chinese_hit1_diff": ch_zh_hit1_diff,
                "english_mrr_diff": ch_en_mrr_diff,
                "english_hit1_diff": ch_en_hit1_diff,
                "overall_mrr_diff": ch_overall_mrr_diff,
                "overall_hit1_diff": ch_overall_hit1_diff
            }
        }
        
        with open("retrieval_comparison_results.json", "w", encoding="utf-8") as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n对比结果已保存到: retrieval_comparison_results.json")
        
        # 分析建议
        print(f"\n📊 分析建议:")
        if ch_overall_mrr_diff > 0.1:
            print(f"   ✅ 包含评估数据显著提升了检索质量 (MRR提升 {ch_overall_mrr_diff:.4f})")
            print(f"   💡 建议：评估数据的context应该加入知识库以测试真实检索能力")
        elif ch_overall_mrr_diff < -0.1:
            print(f"   ⚠️  包含评估数据降低了检索质量 (MRR下降 {abs(ch_overall_mrr_diff):.4f})")
            print(f"   💡 建议：可能存在数据泄露问题，需要进一步分析")
        else:
            print(f"   🔍 两种模式效果相近，差异不大")
            print(f"   💡 建议：可以根据具体需求选择模式")

def test_retrieval_quality():
    """原始测试函数 - 保持向后兼容"""
    return test_retrieval_with_eval_context(include_eval_data=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试检索质量 - MRR评估")
    parser.add_argument("--mode", choices=["with_eval", "without_eval", "compare"], 
                       default="compare", help="测试模式")
    
    args = parser.parse_args()
    
    if args.mode == "with_eval":
        success = test_retrieval_with_eval_context(include_eval_data=True)
        if success:
            print("\n🎉 包含评估数据的检索质量测试完成！")
        else:
            print("\n❌ 包含评估数据的检索质量测试失败！")
    elif args.mode == "without_eval":
        success = test_retrieval_with_eval_context(include_eval_data=False)
        if success:
            print("\n🎉 不包含评估数据的检索质量测试完成！")
        else:
            print("\n❌ 不包含评估数据的检索质量测试失败！")
    else:  # compare
        compare_retrieval_modes()
        print("\n🎉 检索模式对比测试完成！") 