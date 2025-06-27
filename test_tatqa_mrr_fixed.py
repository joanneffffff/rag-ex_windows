#!/usr/bin/env python3
"""
修复版TatQA MRR测试脚本
正确加载原始训练数据，构建完整的知识库，然后添加eval context
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

def test_tatqa_mrr_fixed():
    """修复版TatQA MRR测试"""
    print("=" * 60)
    print("修复版TatQA MRR测试")
    print("=" * 60)
    
    try:
        from config.parameters import Config
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.dto.dto import DocumentWithMetadata
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
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
        
        print("\n2. 加载原始训练数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,  # 加载所有数据
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False  # 不包含评估数据，我们后面手动添加
        )
        
        english_chunks = data_loader.english_docs
        print(f"   ✅ 英文训练数据: {len(english_chunks)} 个chunks")
        
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
        
        print("\n5. 构建完整知识库...")
        # 构建完整的知识库（原始训练数据 + eval context）
        all_documents = []
        all_doc_info = []
        
        # 添加原始训练数据
        for i, doc in enumerate(english_chunks):
            all_documents.append(doc.content)
            # 使用简单的文档ID
            doc_id = f"train_doc_{i}"
            
            all_doc_info.append({
                'id': doc_id,
                'content': doc.content,
                'relevant_doc_ids': []
            })
        
        # 添加eval context
        for context_text, doc_info in eval_knowledge_base.items():
            all_documents.append(context_text)
            all_doc_info.append(doc_info)
        
        print(f"   ✅ 完整知识库: {len(all_documents)} 个文档")
        print(f"      - 训练数据: {len(english_chunks)} 个")
        print(f"      - eval context: {len(eval_knowledge_base)} 个")
        
        print("\n6. 编码所有文档...")
        # 分批编码以适应CPU
        batch_size = 8
        all_embeddings = []
        
        for i in tqdm(range(0, len(all_documents), batch_size), desc="编码文档"):
            batch_docs = all_documents[i:i+batch_size]
            batch_embeddings = encoder.encode(batch_docs)
            all_embeddings.extend(batch_embeddings)
        
        all_embeddings_array = np.array(all_embeddings, dtype=np.float32)
        print(f"   ✅ 编码完成，维度: {all_embeddings_array.shape}")
        
        print("\n7. 创建FAISS索引...")
        # 创建FAISS索引
        dimension = all_embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # 使用内积索引
        index.add(all_embeddings_array)
        
        print(f"   ✅ FAISS索引创建完成，总向量数: {index.ntotal}")
        
        # 保存索引和嵌入
        index_path = "models/embedding_cache/finetuned_finbert_tatqa_complete.faiss"
        embeddings_path = "models/embedding_cache/finetuned_finbert_tatqa_complete.npy"
        
        faiss.write_index(index, index_path)
        np.save(embeddings_path, all_embeddings_array)
        
        print(f"   ✅ 索引已保存到: {index_path}")
        print(f"   ✅ 嵌入已保存到: {embeddings_path}")
        
        print("\n8. 开始评估TatQA...")
        ranks = []
        found_count = 0
        total_samples = len(tatqa_eval)  # 评估全部样本
        
        print(f"   将评估全部 {total_samples} 个TatQA样本...")
        
        for i, sample in enumerate(tqdm(tatqa_eval, desc="评估TatQA")):
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
                    if idx < len(all_doc_info):
                        doc_info = all_doc_info[idx]
                        # 创建DocumentWithMetadata对象，使用source字段存储doc_id
                        from xlm.dto.dto import DocumentMetadata
                        metadata = DocumentMetadata(source=doc_info['id'])
                        doc = DocumentWithMetadata(
                            content=doc_info['content'],
                            metadata=metadata
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
                        retrieved_doc = retrieved_docs[found_rank-1]
                        print(f"  相关文档: {retrieved_doc.content[:100]}...")
                        print(f"  文档ID: {retrieved_doc.metadata.source}")
                
                # 每100个样本显示一次进度
                if (i + 1) % 100 == 0:
                    current_mrr = calculate_mrr(ranks)
                    current_recall = found_count / len(ranks)
                    print(f"\n   进度: {i+1}/{total_samples}, 当前MRR: {current_mrr:.4f}, 召回率: {current_recall:.4f}")
                
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
        print("修复版TatQA MRR评估结果")
        print("=" * 60)
        print(f"总样本数: {len(ranks)}")
        print(f"找到正确答案: {found_count}")
        print(f"召回率: {found_count/len(ranks):.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"Hit@1: {hit_at_1:.4f}")
        print(f"Hit@3: {hit_at_3:.4f}")
        print(f"Hit@5: {hit_at_5:.4f}")
        print(f"Hit@10: {hit_at_10:.4f}")
        print(f"知识库大小: {len(all_documents)} 个文档")
        
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
            "knowledge_base_size": len(all_documents),
            "training_docs": len(english_chunks),
            "eval_docs": len(eval_knowledge_base),
            "index_path": index_path
        }
        
        with open("tatqa_mrr_results_fixed.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: tatqa_mrr_results_fixed.json")
        print(f"完整索引: {index_path}")
        print("\n🎉 修复版TatQA MRR测试完成！")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_tatqa_mrr_fixed() 