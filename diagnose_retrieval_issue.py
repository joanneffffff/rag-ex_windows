#!/usr/bin/env python3
"""诊断检索性能下降问题"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def diagnose_retrieval_issue():
    """诊断检索性能下降问题"""
    print("=== 诊断检索性能下降问题 ===")
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("1. 检查数据加载配置...")
        
        # 测试不同的数据加载配置
        configs_to_test = [
            ("默认配置（包含评估数据）", True),
            ("排除评估数据", False)
        ]
        
        for config_name, include_eval in configs_to_test:
            print(f"\n--- {config_name} ---")
            
            data_loader = OptimizedDataLoader(
                data_dir="data",
                max_samples=100,  # 只加载100个样本用于测试
                chinese_document_level=True,
                english_chunk_level=True,
                include_eval_data=include_eval
            )
            
            chinese_docs = data_loader.chinese_docs
            english_docs = data_loader.english_docs
            
            print(f"  中文文档数: {len(chinese_docs)}")
            print(f"  英文文档数: {len(english_docs)}")
            
            # 检查评估数据
            eval_chinese_count = sum(1 for doc in chinese_docs if 'eval' in doc.metadata.source)
            eval_english_count = sum(1 for doc in english_docs if 'eval' in doc.metadata.source)
            print(f"  评估文档数: 中文{eval_chinese_count}, 英文{eval_english_count}")
        
        print("\n2. 检查评估数据格式...")
        
        # 加载评估数据
        def load_eval_data(eval_file: str):
            data = []
            with open(eval_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        
        alphafin_eval = load_eval_data("evaluate_mrr/alphafin_eval.jsonl")
        tatqa_eval = load_eval_data("evaluate_mrr/tatqa_eval.jsonl")
        
        print(f"  AlphaFin评估样本数: {len(alphafin_eval)}")
        print(f"  TatQA评估样本数: {len(tatqa_eval)}")
        
        # 检查第一个样本
        if alphafin_eval:
            first_sample = alphafin_eval[0]
            print(f"  第一个AlphaFin样本:")
            print(f"    Query: {first_sample['query']}")
            print(f"    Doc ID: {first_sample['doc_id']}")
            print(f"    Context长度: {len(first_sample['context'])}")
        
        print("\n3. 测试检索器性能...")
        
        # 加载编码器
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        
        # 使用包含评估数据的配置
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True  # 包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"  知识库大小: 中文{len(chinese_chunks)}, 英文{len(english_chunks)}")
        
        # 创建检索器
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
        
        print("\n4. 测试中文检索...")
        
        if alphafin_eval:
            test_sample = alphafin_eval[0]
            query = test_sample['query']
            context = test_sample['context']
            doc_id = test_sample['doc_id']
            
            print(f"  查询: {query}")
            print(f"  目标Doc ID: {doc_id}")
            
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
            
            print(f"  检索到 {len(retrieved_docs)} 个文档")
            
            # 检查是否包含目标文档
            from test_retrieval_mrr import find_correct_document_rank
            
            found_rank = find_correct_document_rank(
                context=context,
                retrieved_docs=retrieved_docs,
                sample=test_sample,
                encoder=encoder_ch
            )
            
            print(f"  找到正确答案的排名: {found_rank}")
            
            if found_rank > 0:
                print(f"  ✅ 成功找到正确答案！")
                matched_doc = retrieved_docs[found_rank-1]
                print(f"  匹配文档内容: {matched_doc.content[:200]}...")
                
                if scores and found_rank <= len(scores):
                    print(f"  匹配文档分数: {scores[found_rank-1]:.4f}")
            else:
                print(f"  ❌ 未找到正确答案")
                
                # 检查知识库中是否有这个doc_id
                print(f"  检查知识库中是否有doc_id={doc_id}的文档...")
                target_found = False
                for i, doc in enumerate(chinese_chunks):
                    try:
                        if doc.content.startswith('{'):
                            doc_data = json.loads(doc.content)
                            doc_id_in_doc = doc_data.get('doc_id')
                            if doc_id_in_doc == doc_id:
                                print(f"    ✅ 在知识库位置{i}找到doc_id={doc_id}的文档")
                                target_found = True
                                break
                    except:
                        pass
                
                if not target_found:
                    print(f"    ❌ 知识库中没有doc_id={doc_id}的文档")
                
                # 显示前3个检索结果
                print(f"  前3个检索结果:")
                for i, doc in enumerate(retrieved_docs[:3]):
                    score_info = f" (分数: {scores[i]:.4f})" if scores and i < len(scores) else ""
                    print(f"    {i+1}. {doc.content[:100]}...{score_info}")
        
        print("\n5. 测试英文检索...")
        
        if tatqa_eval:
            test_sample_en = tatqa_eval[0]
            query_en = test_sample_en['query']
            context_en = test_sample_en['context']
            
            print(f"  查询: {query_en}")
            
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
            
            print(f"  检索到 {len(retrieved_docs_en)} 个文档")
            
            found_rank_en = find_correct_document_rank(
                context=context_en,
                retrieved_docs=retrieved_docs_en,
                sample=test_sample_en,
                encoder=encoder_en
            )
            
            print(f"  找到正确答案的排名: {found_rank_en}")
            
            if found_rank_en > 0:
                print(f"  ✅ 成功找到正确答案！")
            else:
                print(f"  ❌ 未找到正确答案")
        
        print("\n6. 可能的问题分析...")
        
        print("  可能的原因:")
        print("  1. 数据加载顺序问题：评估数据可能没有正确添加到知识库")
        print("  2. 文档格式问题：评估数据的格式可能与匹配逻辑不兼容")
        print("  3. 编码器问题：编码器可能没有正确加载或使用")
        print("  4. 检索器配置问题：检索参数可能不合适")
        print("  5. 匹配逻辑问题：匹配函数可能有问题")
        
        print("\n  建议的解决方案:")
        print("  1. 确保评估数据正确加载到知识库")
        print("  2. 检查文档格式是否一致")
        print("  3. 验证编码器是否正常工作")
        print("  4. 调整检索参数（top_k, batch_size等）")
        print("  5. 调试匹配逻辑")
        
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_retrieval_issue() 