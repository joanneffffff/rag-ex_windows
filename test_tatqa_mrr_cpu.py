#!/usr/bin/env python3
"""
简化版TatQA MRR测试脚本（CPU版本）
专门用于测试TatQA增强版数据的检索质量
使用现有FAISS索引
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def load_eval_data(eval_file: str):
    """加载评估数据"""
    data = []
    with open(eval_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def calculate_mrr(ranks):
    """计算MRR"""
    if not ranks:
        return 0.0
    reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in ranks]
    return float(np.mean(reciprocal_ranks))

def calculate_hit_rate(ranks, k=1):
    """计算Hit@k"""
    if not ranks:
        return 0.0
    hits = [1 if rank <= k and rank > 0 else 0 for rank in ranks]
    return float(np.mean(hits))

def test_tatqa_mrr():
    """测试TatQA的MRR"""
    print("=" * 60)
    print("TatQA MRR测试（CPU版本）- 使用现有索引")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        from enhanced_evaluation_functions import find_correct_document_rank_enhanced
        
        config = Config()
        
        print("1. 加载编码器（CPU模式）...")
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
            device="cpu"  # 强制使用CPU
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False  # 不包含评估数据，避免数据泄露
        )
        
        english_chunks = data_loader.english_docs
        chinese_chunks = data_loader.chinese_docs
        print(f"   ✅ 英文chunks: {len(english_chunks)}")
        print(f"   ✅ 中文chunks: {len(chinese_chunks)}")
        
        print("\n3. 加载TatQA增强版评估数据...")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl")
        print(f"   ✅ TatQA评估样本: {len(tatqa_eval)}")
        
        print("\n4. 创建检索器（使用现有索引）...")
        retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_en,  # 使用同一个编码器
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,  # 强制不使用GPU
            batch_size=4,   # 减小batch_size
            cache_dir=config.encoder.cache_dir
        )
        print("   ✅ 检索器创建成功（使用现有索引）")
        
        print("\n5. 开始评估TatQA...")
        ranks = []
        found_count = 0
        total_samples = min(50, len(tatqa_eval))  # 减少样本数量以加快测试
        
        for i, sample in enumerate(tqdm(tatqa_eval[:total_samples], desc="评估TatQA")):
            query = sample.get('query', '')
            context = sample.get('context', '')
            relevant_doc_ids = sample.get('relevant_doc_ids', [])
            
            if not query or not context:
                continue
            
            try:
                # 检索
                retrieved_result = retriever.retrieve(
                    text=query, 
                    top_k=20, 
                    return_scores=True, 
                    language='en'
                )
                
                if isinstance(retrieved_result, tuple):
                    retrieved_docs, scores = retrieved_result
                else:
                    retrieved_docs = retrieved_result
                
                # 使用增强版函数查找排名
                found_rank = find_correct_document_rank_enhanced(
                    context=context,
                    retrieved_docs=retrieved_docs,
                    sample=sample,
                    encoder=encoder_en
                )
                
                ranks.append(found_rank)
                if found_rank > 0:
                    found_count += 1
                
                # 显示前几个样本的详细信息
                if i < 3:
                    print(f"\n样本 {i+1}:")
                    print(f"  问题: {query[:80]}...")
                    print(f"  相关文档ID: {relevant_doc_ids}")
                    print(f"  找到排名: {found_rank}")
                    if found_rank > 0:
                        print(f"  相关文档: {retrieved_docs[found_rank-1].content[:100]}...")
                
            except Exception as e:
                print(f"   样本 {i} 处理失败: {e}")
                ranks.append(0)
        
        # 计算指标
        mrr = calculate_mrr(ranks)
        hit_at_1 = calculate_hit_rate(ranks, k=1)
        hit_at_3 = calculate_hit_rate(ranks, k=3)
        hit_at_5 = calculate_hit_rate(ranks, k=5)
        hit_at_10 = calculate_hit_rate(ranks, k=10)
        
        print(f"\n" + "=" * 60)
        print("TatQA MRR评估结果（CPU版本）")
        print("=" * 60)
        print(f"总样本数: {len(ranks)}")
        print(f"找到正确答案: {found_count}")
        print(f"召回率: {found_count/len(ranks):.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"Hit@1: {hit_at_1:.4f}")
        print(f"Hit@3: {hit_at_3:.4f}")
        print(f"Hit@5: {hit_at_5:.4f}")
        print(f"Hit@10: {hit_at_10:.4f}")
        
        # 保存结果
        results = {
            "dataset": "TatQA增强版",
            "total_samples": len(ranks),
            "found_samples": found_count,
            "recall": found_count/len(ranks),
            "mrr": mrr,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "hit_at_10": hit_at_10,
            "mode": "CPU",
            "enhanced_evaluation": True,
            "used_existing_index": True
        }
        
        with open("tatqa_mrr_results_cpu.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: tatqa_mrr_results_cpu.json")
        print("\n🎉 TatQA MRR测试完成！")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_tatqa_mrr() 