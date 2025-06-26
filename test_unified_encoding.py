#!/usr/bin/env python3
"""测试统一编码的效果"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_unified_encoding():
    """测试统一编码的效果"""
    print("=== 测试统一编码效果 ===")
    
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
        
        print("\n2. 加载包含评估数据的知识库...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,  # 只加载100个样本
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True  # 包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 知识库加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n3. 创建检索器（统一编码）...")
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
        
        print("\n4. 测试中文检索...")
        # 加载评估数据用于测试
        def load_eval_data(eval_file: str):
            data = []
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        test_sample = alphafin_eval[0]
        
        query = test_sample['query']
        context = test_sample['context']
        doc_id = test_sample['doc_id']
        
        print(f"   查询: {query}")
        print(f"   正确答案ID: {doc_id}")
        
        # 检索
        retrieved_result = retriever.retrieve(
            text=query, 
            top_k=20, 
            return_scores=True, 
            language='zh'
        )
        
        if isinstance(retrieved_result, tuple):
            retrieved_docs, scores = retrieved_result
        else:
            retrieved_docs = retrieved_result
            scores = []
        
        print(f"   检索到 {len(retrieved_docs)} 个文档")
        
        # 使用改进的匹配逻辑
        from test_retrieval_mrr import find_correct_document_rank
        
        found_rank = find_correct_document_rank(
            context=context,
            retrieved_docs=retrieved_docs,
            sample=test_sample,
            encoder=encoder_ch
        )
        
        print(f"   找到正确答案的排名: {found_rank}")
        
        if found_rank > 0:
            print(f"   ✅ 成功找到正确答案！")
            matched_doc = retrieved_docs[found_rank-1]
            print(f"   匹配文档内容: {matched_doc.content[:200]}...")
            
            # 显示分数
            if scores and found_rank <= len(scores):
                print(f"   匹配文档分数: {scores[found_rank-1]:.4f}")
        else:
            print(f"   ❌ 未找到正确答案")
            print(f"   前3个检索结果:")
            for i, doc in enumerate(retrieved_docs[:3]):
                score_info = f" (分数: {scores[i]:.4f})" if scores and i < len(scores) else ""
                print(f"     {i+1}. {doc.content[:100]}...{score_info}")
        
        print("\n5. 测试英文检索...")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval.jsonl")
        test_sample_en = tatqa_eval[0]
        
        query_en = test_sample_en['query']
        context_en = test_sample_en['context']
        
        print(f"   查询: {query_en}")
        
        # 检索
        retrieved_result_en = retriever.retrieve(
            text=query_en, 
            top_k=20, 
            return_scores=True, 
            language='en'
        )
        
        if isinstance(retrieved_result_en, tuple):
            retrieved_docs_en, scores_en = retrieved_result_en
        else:
            retrieved_docs_en = retrieved_result_en
            scores_en = []
        
        print(f"   检索到 {len(retrieved_docs_en)} 个文档")
        
        found_rank_en = find_correct_document_rank(
            context=context_en,
            retrieved_docs=retrieved_docs_en,
            sample=test_sample_en,
            encoder=encoder_en
        )
        
        print(f"   找到正确答案的排名: {found_rank_en}")
        
        if found_rank_en > 0:
            print(f"   ✅ 成功找到正确答案！")
            if scores_en and found_rank_en <= len(scores_en):
                print(f"   匹配文档分数: {scores_en[found_rank_en-1]:.4f}")
        else:
            print(f"   ❌ 未找到正确答案")
        
        print("\n🎉 统一编码测试完成！")
        
        # 总结结果
        print(f"\n📊 测试结果总结:")
        print(f"   中文检索: {'✅ 成功' if found_rank > 0 else '❌ 失败'} (排名: {found_rank})")
        print(f"   英文检索: {'✅ 成功' if found_rank_en > 0 else '❌ 失败'} (排名: {found_rank_en})")
        
        # 优势分析
        print(f"\n🚀 统一编码的优势:")
        print(f"   ✅ 训练数据和评估数据一起编码，避免重复计算")
        print(f"   ✅ 所有数据在同一个向量空间中，检索更准确")
        print(f"   ✅ 简化了数据加载流程，减少代码复杂度")
        print(f"   ✅ 避免了数据不一致的问题")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_unified_encoding() 