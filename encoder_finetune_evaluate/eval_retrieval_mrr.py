#!/usr/bin/env python3
"""
测试检索质量 - MRR评估 (CPU版本)
使用evaluate_mrr/alphafin_eval.jsonl和tatqa_eval_enhanced.jsonl作为测试集
支持两种模式：
1. 评估数据context加入知识库 - 测试真实检索能力
2. 评估数据context不加入知识库 - 避免数据泄露

改进的匹配策略：
1. relevant_doc_ids匹配（最严格，适用于英文数据）
2. ID匹配（适用于中文数据）
3. 内容哈希匹配
4. 相似度匹配
5. 模糊文本匹配
"""

import sys
import os
import json
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

# 导入必要的类型
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

# 导入增强版评估函数
sys.path.append(str(Path(__file__).parent.parent))
from enhanced_evaluation_functions import find_correct_document_rank_enhanced

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

def calculate_content_hash(text: str) -> str:
    """计算文本内容的哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def find_correct_document_rank(
    context: str, 
    retrieved_docs: List[DocumentWithMetadata], 
    sample: Dict[str, Any],
    encoder=None
) -> int:
    """
    使用多种策略查找正确答案的排名（兼容旧版本）
    
    Args:
        context: 正确答案的context
        retrieved_docs: 检索到的文档列表
        sample: 评估样本
        encoder: 编码器（用于相似度计算）
    
    Returns:
        找到的排名，0表示未找到
    """
    # 使用增强版函数
    return find_correct_document_rank_enhanced(context, retrieved_docs, sample, encoder)

def test_retrieval_with_eval_context(include_eval_data: bool = True):
    """测试检索质量 - 可选择是否包含评估数据到知识库"""
    mode = "包含评估数据" if include_eval_data else "不包含评估数据"
    print("=" * 60)
    print(f"测试检索质量 - MRR评估 ({mode}) - CPU版本")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("1. 加载编码器（CPU模式）...")
        encoder_ch = FinbertEncoder(
            model_name="./models/finetuned_alphafin_zh_optimized",
            cache_dir=config.encoder.cache_dir,
            device="cpu"  # 强制使用CPU
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
            device="cpu"  # 强制使用CPU
        )
        print("   ✅ 编码器加载成功（CPU模式）")
        
        print("\n2. 加载训练数据（知识库）...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,  # 加载所有数据
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=include_eval_data  # 直接控制是否包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 训练数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n3. 加载评估数据...")
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        # 使用增强版TatQA数据
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl")
        
        print(f"   ✅ 评估数据加载成功:")
        print(f"      AlphaFin评估样本: {len(alphafin_eval)}")
        print(f"      TatQA增强版评估样本: {len(tatqa_eval)}")
        
        print("\n4. 创建检索器（CPU模式）...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,  # 强制不使用GPU
            batch_size=4,   # 减小batch_size以适应CPU
            cache_dir=config.encoder.cache_dir
        )
        print("   ✅ 检索器创建成功（CPU模式）")
        
        print("\n5. 测试中文检索质量 (AlphaFin)...")
        chinese_ranks = []
        chinese_queries = []
        
        for i, sample in enumerate(tqdm(alphafin_eval[:50], desc="测试中文检索")):  # 减少测试样本以适应CPU
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
            found_rank = find_correct_document_rank(
                context, retrieved_docs, sample, encoder_ch
            )
            
            chinese_ranks.append(found_rank)
            chinese_queries.append(query)
            
            if i < 3:  # 显示前3个样本的详细信息
                print(f"   查询 {i+1}: {query[:50]}...")
                print(f"   找到位置: {found_rank}")
                if found_rank > 0:
                    print(f"   相关文档: {retrieved_docs[found_rank-1].content[:100]}...")
                print()
        
        print("\n6. 测试英文检索质量 (TatQA增强版)...")
        english_ranks = []
        english_queries = []
        
        for i, sample in enumerate(tqdm(tatqa_eval[:50], desc="测试英文检索")):  # 减少测试样本以适应CPU
            query = sample.get('query', '')
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
            
            # 检查正确答案是否在检索结果中（使用增强版函数）
            found_rank = find_correct_document_rank_enhanced(
                context, retrieved_docs, sample, encoder_en
            )
            
            english_ranks.append(found_rank)
            english_queries.append(query)
            
            if i < 3:  # 显示前3个样本的详细信息
                print(f"   Query {i+1}: {query[:50]}...")
                print(f"   Found at rank: {found_rank}")
                print(f"   Relevant doc IDs: {sample.get('relevant_doc_ids', [])}")
                if found_rank > 0:
                    print(f"   Relevant doc: {retrieved_docs[found_rank-1].content[:100]}...")
                print()
        
        print("\n" + "=" * 60)
        print(f"检索质量评估结果 ({mode}) - CPU版本")
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
        
        print(f"\n英文检索 (TatQA增强版):")
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
        
        return {
            'chinese': {
                'mrr': chinese_mrr,
                'hit_at_1': chinese_hit1,
                'hit_at_5': chinese_hit5,
                'hit_at_10': chinese_hit10,
                'samples': len(chinese_ranks)
            },
            'english': {
                'mrr': english_mrr,
                'hit_at_1': english_hit1,
                'hit_at_5': english_hit5,
                'hit_at_10': english_hit10,
                'samples': len(english_ranks)
            },
            'overall': {
                'mrr': overall_mrr,
                'hit_at_1': overall_hit1,
                'hit_at_5': overall_hit5,
                'hit_at_10': overall_hit10,
                'samples': len(all_ranks)
            }
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_retrieval_modes():
    """比较不同检索模式的效果"""
    print("=" * 60)
    print("比较不同检索模式的效果")
    print("=" * 60)
    
    # 测试包含评估数据的模式
    print("\n1. 测试包含评估数据的模式...")
    results_with_eval = test_retrieval_with_eval_context(include_eval_data=True)
    
    # 测试不包含评估数据的模式
    print("\n2. 测试不包含评估数据的模式...")
    results_without_eval = test_retrieval_with_eval_context(include_eval_data=False)
    
    # 比较结果
    print("\n" + "=" * 60)
    print("模式比较结果")
    print("=" * 60)
    
    if results_with_eval and results_without_eval:
        print("包含评估数据 vs 不包含评估数据:")
        print(f"中文MRR: {results_with_eval['chinese']['mrr']:.4f} vs {results_without_eval['chinese']['mrr']:.4f}")
        print(f"英文MRR: {results_with_eval['english']['mrr']:.4f} vs {results_without_eval['english']['mrr']:.4f}")
        print(f"总体MRR: {results_with_eval['overall']['mrr']:.4f} vs {results_without_eval['overall']['mrr']:.4f}")
    else:
        print("❌ 比较失败")

def test_retrieval_quality():
    """测试检索质量"""
    print("=" * 60)
    print("测试检索质量")
    print("=" * 60)
    
    # 默认测试不包含评估数据的模式
    results = test_retrieval_with_eval_context(include_eval_data=False)
    
    if results:
        print("\n🎉 测试完成！")
        print(f"总体MRR: {results['overall']['mrr']:.4f}")
        print(f"总体Hit@1: {results['overall']['hit_at_1']:.4f}")
    else:
        print("❌ 测试失败")

def evaluate_retrieval_quality(include_eval_data=True, max_eval_samples=None):
    """
    完整评估检索质量
    
    Args:
        include_eval_data: 是否包含评估数据到知识库
        max_eval_samples: 最大评估样本数
    """
    print("=" * 60)
    print(f"完整评估检索质量 (CPU版本)")
    print(f"包含评估数据: {include_eval_data}")
    print(f"最大样本数: {max_eval_samples}")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("1. 加载编码器（CPU模式）...")
        encoder_ch = FinbertEncoder(
            model_name="./models/finetuned_alphafin_zh_optimized",
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=include_eval_data
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n3. 加载评估数据...")
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl")  # 使用增强版
        
        if max_eval_samples:
            alphafin_eval = alphafin_eval[:max_eval_samples]
            tatqa_eval = tatqa_eval[:max_eval_samples]
        
        print(f"   ✅ 评估数据加载成功:")
        print(f"      AlphaFin评估样本: {len(alphafin_eval)}")
        print(f"      TatQA增强版评估样本: {len(tatqa_eval)}")
        
        print("\n4. 创建检索器（CPU模式）...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,
            batch_size=4,
            cache_dir=config.encoder.cache_dir
        )
        print("   ✅ 检索器创建成功")
        
        # 评估中文数据
        print(f"\n--- 评估中文数据 (AlphaFin) ---")
        chinese_results = evaluate_dataset(
            eval_data=alphafin_eval,
            retriever=retriever,
            encoder=encoder_ch,
            language='zh',
            dataset_name="AlphaFin"
        )
        
        # 评估英文数据
        print(f"\n--- 评估英文数据 (TatQA增强版) ---")
        english_results = evaluate_dataset(
            eval_data=tatqa_eval,
            retriever=retriever,
            encoder=encoder_en,
            language='en',
            dataset_name="TatQA"
        )
        
        # 汇总结果
        print(f"\n=== 评估结果汇总 ===")
        print(f"中文数据 (AlphaFin):")
        print(f"  MRR: {chinese_results['mrr']:.4f}")
        print(f"  Hit@1: {chinese_results['hit_at_1']:.4f}")
        print(f"  Hit@3: {chinese_results['hit_at_3']:.4f}")
        print(f"  Hit@5: {chinese_results['hit_at_5']:.4f}")
        print(f"  Hit@10: {chinese_results['hit_at_10']:.4f}")
        print(f"  总样本数: {chinese_results['total_samples']}")
        print(f"  找到正确答案的样本数: {chinese_results['found_samples']}")
        
        print(f"\n英文数据 (TatQA增强版):")
        print(f"  MRR: {english_results['mrr']:.4f}")
        print(f"  Hit@1: {english_results['hit_at_1']:.4f}")
        print(f"  Hit@3: {english_results['hit_at_3']:.4f}")
        print(f"  Hit@5: {english_results['hit_at_5']:.4f}")
        print(f"  Hit@10: {english_results['hit_at_10']:.4f}")
        print(f"  总样本数: {english_results['total_samples']}")
        print(f"  找到正确答案的样本数: {english_results['found_samples']}")
        
        # 计算总体指标
        total_samples = chinese_results['total_samples'] + english_results['total_samples']
        total_found = chinese_results['found_samples'] + english_results['found_samples']
        overall_mrr = (chinese_results['mrr'] + english_results['mrr']) / 2
        overall_hit_at_1 = (chinese_results['hit_at_1'] + english_results['hit_at_1']) / 2
        
        print(f"\n总体指标:")
        print(f"  总体MRR: {overall_mrr:.4f}")
        print(f"  总体Hit@1: {overall_hit_at_1:.4f}")
        print(f"  总样本数: {total_samples}")
        print(f"  总找到数: {total_found}")
        print(f"  总体召回率: {total_found/total_samples:.4f}")
        
        return {
            'chinese': chinese_results,
            'english': english_results,
            'overall': {
                'mrr': overall_mrr,
                'hit_at_1': overall_hit_at_1,
                'total_samples': total_samples,
                'found_samples': total_found
            }
        }
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_dataset(eval_data, retriever, encoder, language, dataset_name):
    """
    评估单个数据集的检索质量
    
    Args:
        eval_data: 评估数据列表
        retriever: 检索器
        encoder: 编码器
        language: 语言 ('zh' 或 'en')
        dataset_name: 数据集名称
    
    Returns:
        评估结果字典
    """
    print(f"开始评估 {dataset_name} 数据集 ({len(eval_data)} 个样本)...")
    
    mrr_scores = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    hit_at_10 = 0
    found_samples = 0
    
    for i, sample in enumerate(eval_data):
        if i % 50 == 0:  # 减少进度显示频率以适应CPU
            print(f"  处理进度: {i}/{len(eval_data)}")
        
        query = sample.get('query', sample.get('question', ''))
        context = sample.get('context', '')
        
        if not query or not context:
            continue
        
        try:
            # 检索
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=20, 
                return_scores=True, 
                language=language
            )
            
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            # 找到正确答案的排名（使用增强版函数）
            found_rank = find_correct_document_rank_enhanced(
                context=context,
                retrieved_docs=retrieved_docs,
                sample=sample,
                encoder=encoder
            )
            
            if found_rank > 0:
                found_samples += 1
                mrr_score = 1.0 / found_rank
                mrr_scores.append(mrr_score)
                
                if found_rank == 1:
                    hit_at_1 += 1
                if found_rank <= 3:
                    hit_at_3 += 1
                if found_rank <= 5:
                    hit_at_5 += 1
                if found_rank <= 10:
                    hit_at_10 += 1
            else:
                mrr_scores.append(0.0)
                
        except Exception as e:
            print(f"   样本 {i} 处理失败: {e}")
            mrr_scores.append(0.0)
    
    # 计算指标
    total_samples = len(eval_data)
    mrr = sum(mrr_scores) / total_samples if total_samples > 0 else 0.0
    hit_at_1_rate = hit_at_1 / total_samples if total_samples > 0 else 0.0
    hit_at_3_rate = hit_at_3 / total_samples if total_samples > 0 else 0.0
    hit_at_5_rate = hit_at_5 / total_samples if total_samples > 0 else 0.0
    hit_at_10_rate = hit_at_10 / total_samples if total_samples > 0 else 0.0
    
    print(f"  {dataset_name} 评估完成:")
    print(f"    MRR: {mrr:.4f}")
    print(f"    Hit@1: {hit_at_1_rate:.4f} ({hit_at_1}/{total_samples})")
    print(f"    Hit@3: {hit_at_3_rate:.4f} ({hit_at_3}/{total_samples})")
    print(f"    Hit@5: {hit_at_5_rate:.4f} ({hit_at_5}/{total_samples})")
    print(f"    Hit@10: {hit_at_10_rate:.4f} ({hit_at_10}/{total_samples})")
    print(f"    找到正确答案: {found_samples}/{total_samples}")
    
    return {
        'mrr': mrr,
        'hit_at_1': hit_at_1_rate,
        'hit_at_3': hit_at_3_rate,
        'hit_at_5': hit_at_5_rate,
        'hit_at_10': hit_at_10_rate,
        'total_samples': total_samples,
        'found_samples': found_samples
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评估检索质量（CPU版本）")
    parser.add_argument("--include_eval_data", action="store_true", 
                       help="是否将评估数据包含在知识库中")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大评估样本数，None表示评估所有样本")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，只评估少量样本")
    parser.add_argument("--compare_modes", action="store_true",
                       help="比较不同检索模式")
    
    args = parser.parse_args()
    
    if args.compare_modes:
        print("=== 比较检索模式 ===")
        compare_retrieval_modes()
    elif args.test_mode:
        print("=== 测试模式（CPU版本）===")
        test_retrieval_with_eval_context(include_eval_data=args.include_eval_data)
    else:
        print("=== 完整评估模式（CPU版本）===")
        # 默认评估所有数据
        max_samples = args.max_samples if args.max_samples else None
        results = evaluate_retrieval_quality(
            include_eval_data=args.include_eval_data,
            max_eval_samples=max_samples
        )
        
        if results:
            print("\n🎉 评估完成！")
            print(f"总体MRR: {results['overall']['mrr']:.4f}")
            print(f"总体Hit@1: {results['overall']['hit_at_1']:.4f}")
        else:
            print("❌ 评估失败") 