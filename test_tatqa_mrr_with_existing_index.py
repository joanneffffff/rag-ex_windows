#!/usr/bin/env python3
"""
TatQA MRR测试脚本（使用现有FAISS索引 + 添加eval context）
将eval context添加到现有的FAISS索引文件中
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

def test_tatqa_mrr_with_existing_index():
    """使用现有FAISS索引 + 添加eval context的TatQA MRR测试"""
    print("=" * 60)
    print("TatQA MRR测试（使用现有FAISS索引 + 添加eval context）")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.dto.dto import DocumentWithMetadata
        from enhanced_evaluation_functions import find_correct_document_rank_enhanced
        import faiss
        
        config = Config()
        
        print("1. 加载编码器（CPU模式）...")
        encoder = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
            device="cpu"
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载现有FAISS索引...")
        existing_index_path = "models/embedding_cache/finetuned_finbert_tatqa_3896_a7ea3a736341a1bf.faiss"
        existing_embeddings_path = "models/embedding_cache/finetuned_finbert_tatqa_3896_a7ea3a736341a1bf.npy"
        
        # 加载现有索引
        index = faiss.read_index(existing_index_path)
        existing_embeddings = np.load(existing_embeddings_path)
        
        print(f"   ✅ 现有索引: {index.ntotal} 个向量，维度: {index.d}")
        print(f"   ✅ 现有嵌入: {existing_embeddings.shape}")
        
        print("\n3. 加载TatQA增强版评估数据...")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl")
        print(f"   ✅ TatQA评估样本: {len(tatqa_eval)}")
        
        print("\n4. 准备eval context文档...")
        # 从评估数据中提取所有唯一的context
        eval_knowledge_base = {}
        for i, sample in enumerate(tatqa_eval):
            context = sample.get('context', '').strip()
            if context and context not in eval_knowledge_base:
                doc_id = f"eval_doc_{len(eval_knowledge_base)}"
                eval_knowledge_base[context] = {
                    'id': doc_id,
                    'content': context,
                    'relevant_doc_ids': sample.get('relevant_doc_ids', [])
                }
        
        print(f"   ✅ eval context文档数: {len(eval_knowledge_base)}")
        
        print("\n5. 编码eval context文档...")
        eval_contexts = list(eval_knowledge_base.keys())
        print(f"   编码 {len(eval_contexts)} 个eval context文档...")
        
        # 分批编码以适应CPU
        batch_size = 8
        eval_embeddings = []
        
        for i in tqdm(range(0, len(eval_contexts), batch_size), desc="编码eval context"):
            batch_contexts = eval_contexts[i:i+batch_size]
            batch_embeddings = encoder.encode(batch_contexts)
            eval_embeddings.extend(batch_embeddings)
        
        eval_embeddings_array = np.array(eval_embeddings, dtype=np.float32)
        print(f"   ✅ eval context编码完成，维度: {eval_embeddings_array.shape}")
        
        print("\n6. 将eval context添加到FAISS索引...")
        # 添加eval context到索引
        index.add(eval_embeddings_array)
        
        # 合并嵌入向量
        combined_embeddings = np.vstack([existing_embeddings, eval_embeddings_array])
        
        print(f"   ✅ 索引更新完成，总向量数: {index.ntotal}")
        print(f"   ✅ 合并嵌入维度: {combined_embeddings.shape}")
        
        # 保存更新后的索引和嵌入
        updated_index_path = "models/embedding_cache/finetuned_finbert_tatqa_with_eval.faiss"
        updated_embeddings_path = "models/embedding_cache/finetuned_finbert_tatqa_with_eval.npy"
        
        faiss.write_index(index, updated_index_path)
        np.save(updated_embeddings_path, combined_embeddings)
        
        print(f"   ✅ 更新后的索引已保存到: {updated_index_path}")
        print(f"   ✅ 更新后的嵌入已保存到: {updated_embeddings_path}")
        
        print("\n7. 开始评估TatQA...")
        ranks = []
        found_count = 0
        total_samples = min(100, len(tatqa_eval))  # 测试前100个样本
        
        # 构建完整的知识库（现有 + eval context）
        all_contexts = []
        all_doc_info = []
        
        # 添加现有文档（这里需要从原始数据中获取，暂时用占位符）
        for i in range(existing_embeddings.shape[0]):
            all_contexts.append(f"existing_doc_{i}")
            all_doc_info.append({
                'id': f"existing_doc_{i}",
                'content': f"existing_doc_{i}",
                'relevant_doc_ids': []
            })
        
        # 添加eval context
        for context_text, doc_info in eval_knowledge_base.items():
            all_contexts.append(context_text)
            all_doc_info.append(doc_info)
        
        for i, sample in enumerate(tqdm(tatqa_eval[:total_samples], desc="评估TatQA")):
            query = sample.get('query', '')
            context = sample.get('context', '')
            relevant_doc_ids = sample.get('relevant_doc_ids', [])
            
            if not query or not context:
                continue
            
            try:
                # 编码查询
                query_embedding = encoder.encode([query])[0].reshape(1, -1)
                
                # 检索
                scores, indices = index.search(query_embedding, k=20)
                
                # 构建检索结果
                retrieved_docs = []
                for idx in indices[0]:
                    if idx < len(all_contexts):
                        context_text = all_contexts[idx]
                        doc_info = all_doc_info[idx]
                        doc = DocumentWithMetadata(
                            content=json.dumps({
                                'context': context_text,
                                'doc_id': doc_info['id'],
                                'relevant_doc_ids': doc_info['relevant_doc_ids']
                            }),
                            metadata={'doc_id': doc_info['id']}
                        )
                        retrieved_docs.append(doc)
                
                # 使用增强版函数查找排名
                found_rank = find_correct_document_rank_enhanced(
                    context=context,
                    retrieved_docs=retrieved_docs,
                    sample=sample,
                    encoder=encoder
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
        print("TatQA MRR评估结果（使用现有索引 + eval context）")
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
            "used_existing_index": True,
            "added_eval_context": True,
            "total_vectors": index.ntotal,
            "existing_vectors": existing_embeddings.shape[0],
            "eval_vectors": eval_embeddings_array.shape[0]
        }
        
        with open("tatqa_mrr_results_with_existing_index.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: tatqa_mrr_results_with_existing_index.json")
        print(f"更新后的FAISS索引: {updated_index_path}")
        print("\n🎉 TatQA MRR测试完成！")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_tatqa_mrr_with_existing_index() 