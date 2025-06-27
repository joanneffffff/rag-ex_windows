#!/usr/bin/env python3
"""
严格版TatQA MRR测试脚本（CPU版本）
不将评估数据的context添加到知识库，避免数据泄露
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

def test_tatqa_mrr_strict():
    """严格版TatQA MRR测试（不包含评估数据到知识库）"""
    print("=" * 60)
    print("严格版TatQA MRR测试（CPU版本）")
    print("不包含评估数据到知识库，避免数据泄露")
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
        
        print("\n2. 加载训练数据作为知识库...")
        # 使用训练数据作为知识库，而不是评估数据
        train_data = load_eval_data("evaluate_mrr/tatqa_train_qc_enhanced.jsonl")
        print(f"   ✅ 训练数据样本: {len(train_data)}")
        
        print("\n3. 加载TatQA增强版评估数据...")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval_enhanced.jsonl")
        print(f"   ✅ TatQA评估样本: {len(tatqa_eval)}")
        
        print("\n4. 准备知识库文档（仅使用训练数据）...")
        # 只从训练数据中提取context作为知识库
        knowledge_base = {}
        eval_contexts = set()  # 记录评估数据中的context，避免重复
        
        # 先记录评估数据中的context
        for sample in tatqa_eval:
            context = sample.get('context', '').strip()
            if context:
                eval_contexts.add(context)
        
        # 从训练数据中构建知识库，排除评估数据中的context
        for i, sample in enumerate(train_data):
            context = sample.get('context', '').strip()
            if context and context not in eval_contexts and context not in knowledge_base:
                doc_id = f"train_doc_{len(knowledge_base)}"
                knowledge_base[context] = {
                    'id': doc_id,
                    'content': context,
                    'relevant_doc_ids': sample.get('relevant_doc_ids', [])
                }
        
        print(f"   ✅ 知识库文档数: {len(knowledge_base)}")
        print(f"   ✅ 排除的评估context数: {len(eval_contexts)}")
        
        if len(knowledge_base) == 0:
            print("   ⚠️  警告：知识库为空！所有训练数据的context都在评估数据中")
            print("   这可能表明训练和评估数据有重叠")
            return None
        
        print("\n5. 编码知识库文档...")
        contexts = list(knowledge_base.keys())
        print(f"   编码 {len(contexts)} 个文档...")
        
        # 分批编码以适应CPU
        batch_size = 8
        all_embeddings = []
        
        for i in tqdm(range(0, len(contexts), batch_size), desc="编码文档"):
            batch_contexts = contexts[i:i+batch_size]
            batch_embeddings = encoder.encode(batch_contexts)
            all_embeddings.extend(batch_embeddings)
        
        embeddings_array = np.array(all_embeddings, dtype=np.float32)
        print(f"   ✅ 编码完成，维度: {embeddings_array.shape}")
        
        print("\n6. 创建FAISS索引...")
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 使用内积索引
        index.add(embeddings_array)
        print(f"   ✅ FAISS索引创建完成，包含 {index.ntotal} 个向量")
        
        print("\n7. 开始评估TatQA...")
        ranks = []
        found_count = 0
        total_samples = min(50, len(tatqa_eval))  # 减少样本数量
        
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
                    if idx < len(contexts):
                        context_text = contexts[idx]
                        doc_info = knowledge_base[context_text]
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
                    else:
                        print(f"  未找到正确答案（正确答案不在知识库中）")
                
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
        print("严格版TatQA MRR评估结果（CPU版本）")
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
            "strict_mode": True,
            "knowledge_base_size": len(knowledge_base),
            "excluded_eval_contexts": len(eval_contexts)
        }
        
        with open("tatqa_mrr_results_strict.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: tatqa_mrr_results_strict.json")
        print("\n🎉 严格版TatQA MRR测试完成！")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_tatqa_mrr_strict() 