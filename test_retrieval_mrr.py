#!/usr/bin/env python3
"""
测试检索质量 - MRR评估
使用evaluate_mrr/alphafin_eval.jsonl和tatqa_eval.jsonl作为测试集
支持两种模式：
1. 评估数据context加入知识库 - 测试真实检索能力
2. 评估数据context不加入知识库 - 避免数据泄露

改进的匹配策略：
1. ID匹配（最鲁棒）
2. 内容哈希匹配
3. 相似度匹配
4. 模糊文本匹配
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
    使用多种策略查找正确答案的排名
    
    Args:
        context: 正确答案的context
        retrieved_docs: 检索到的文档列表
        sample: 评估样本
        encoder: 编码器（用于相似度计算）
    
    Returns:
        找到的排名，0表示未找到
    """
    if not context or not retrieved_docs:
        return 0
    
    # 策略1: ID匹配（最鲁棒）- 仅适用于中文数据
    correct_doc_id = sample.get('doc_id') or sample.get('id') or sample.get('document_id')
    if correct_doc_id:
        for rank, doc in enumerate(retrieved_docs, 1):
            # 尝试从文档内容中提取doc_id（如果是JSON格式）
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_id = doc_data.get('doc_id') or doc_data.get('id')
                    if doc_id == correct_doc_id:
                        return rank
            except:
                pass
            
            # 尝试从元数据中获取doc_id
            doc_id = getattr(doc, 'id', None) or getattr(doc.metadata, 'id', None) or getattr(doc.metadata, 'doc_id', None)
            if doc_id == correct_doc_id:
                return rank
    
    # 策略2: 内容哈希匹配
    context_hash = calculate_content_hash(context.strip())
    for rank, doc in enumerate(retrieved_docs, 1):
        # 处理JSON格式的文档内容
        doc_content = doc.content
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                # 提取context字段
                doc_context = doc_data.get('context', '')
                if doc_context:
                    doc_content = doc_context
        except:
            pass
        
        doc_hash = calculate_content_hash(doc_content.strip())
        if doc_hash == context_hash:
            return rank
    
    # 策略3: 精确文本匹配（改进版）
    context_clean = context.strip().lower()
    for rank, doc in enumerate(retrieved_docs, 1):
        # 处理JSON格式的文档内容
        doc_content = doc.content
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                # 提取context字段
                doc_context = doc_data.get('context', '')
                if doc_context:
                    doc_content = doc_context
        except:
            pass
        
        doc_content_clean = doc_content.strip().lower()
        
        # 检查context是否包含在文档中，或文档是否包含在context中
        if (context_clean in doc_content_clean or 
            doc_content_clean in context_clean or
            context_clean == doc_content_clean):
            return rank
    
    # 策略4: 模糊文本匹配（使用关键词）
    context_words = set(context_clean.split())
    if len(context_words) > 3:  # 至少需要3个词
        for rank, doc in enumerate(retrieved_docs, 1):
            # 处理JSON格式的文档内容
            doc_content = doc.content
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    # 提取context字段
                    doc_context = doc_data.get('context', '')
                    if doc_context:
                        doc_content = doc_context
            except:
                pass
            
            doc_content_clean = doc_content.strip().lower()
            doc_words = set(doc_content_clean.split())
            
            # 计算词汇重叠度
            overlap = len(context_words.intersection(doc_words))
            overlap_ratio = overlap / len(context_words)
            
            # 如果重叠度超过70%，认为匹配
            if overlap_ratio > 0.7:
                return rank
    
    # 策略5: 相似度匹配（如果有编码器）
    if encoder and len(context) > 10:  # 确保context足够长
        try:
            context_embedding = encoder.encode([context])
            
            # 准备文档内容用于编码
            doc_contents = []
            for doc in retrieved_docs:
                doc_content = doc.content
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        # 提取context字段
                        doc_context = doc_data.get('context', '')
                        if doc_context:
                            doc_content = doc_context
                except:
                    pass
                doc_contents.append(doc_content)
            
            doc_embeddings = encoder.encode(doc_contents)
            
            # 计算余弦相似度
            similarities = []
            for doc_emb in doc_embeddings:
                cos_sim = np.dot(context_embedding[0], doc_emb) / (
                    np.linalg.norm(context_embedding[0]) * np.linalg.norm(doc_emb)
                )
                similarities.append(cos_sim)
            
            # 找到最高相似度的文档
            max_sim_idx = int(np.argmax(similarities))
            max_similarity = similarities[max_sim_idx]
            
            # 如果相似度超过阈值，认为匹配
            if max_similarity > 0.8:
                return max_sim_idx + 1
                
        except Exception as e:
            print(f"相似度计算失败: {e}")
    
    return 0

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
            found_rank = find_correct_document_rank(
                context, retrieved_docs, sample, encoder_ch
            )
            
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
            found_rank = find_correct_document_rank(
                context, retrieved_docs, sample, encoder_en
            )
            
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

def evaluate_retrieval_quality(include_eval_data=True, max_eval_samples=None):
    """
    评估检索质量
    
    Args:
        include_eval_data: 是否将评估数据包含在知识库中
        max_eval_samples: 最大评估样本数，None表示评估所有样本
    """
    print("=== 检索质量评估 ===")
    print(f"包含评估数据到知识库: {include_eval_data}")
    print(f"最大评估样本数: {max_eval_samples if max_eval_samples else '全部'}")
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("\n1. 加载编码器...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        print("   ✅ 编码器加载成功")
        
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
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval.jsonl")
        
        # 如果指定了最大样本数，则进行采样
        if max_eval_samples:
            alphafin_eval = alphafin_eval[:max_eval_samples]
            tatqa_eval = tatqa_eval[:max_eval_samples]
        
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
        
        print("\n5. 开始评估...")
        
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
        print(f"\n--- 评估英文数据 (TatQA) ---")
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
        
        print(f"\n英文数据 (TatQA):")
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
        if i % 100 == 0:
            print(f"  处理进度: {i}/{len(eval_data)}")
        
        query = sample['query']
        context = sample['context']
        
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
            
            # 找到正确答案的排名
            found_rank = find_correct_document_rank(
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
    
    parser = argparse.ArgumentParser(description="评估检索质量")
    parser.add_argument("--include_eval_data", action="store_true", 
                       help="是否将评估数据包含在知识库中")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="最大评估样本数，None表示评估所有样本")
    parser.add_argument("--test_mode", action="store_true",
                       help="测试模式，只评估少量样本")
    
    args = parser.parse_args()
    
    if args.test_mode:
        print("=== 测试模式 ===")
        test_retrieval_with_eval_context(include_eval_data=args.include_eval_data)
    else:
        print("=== 完整评估模式 ===")
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