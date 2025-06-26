#!/usr/bin/env python3
"""调试匹配问题"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def debug_chinese_matching():
    """调试中文匹配问题"""
    print("=== 调试中文匹配问题 ===")
    
    # 加载评估数据
    with open("evaluate_mrr/alphafin_eval.jsonl", "r", encoding="utf-8") as f:
        first_sample = json.loads(f.readline().strip())
    
    print(f"评估样本:")
    print(f"  Query: {first_sample['query']}")
    print(f"  Doc ID: {first_sample['doc_id']}")
    print(f"  Context: {first_sample['context'][:100]}...")
    
    # 检查知识库中是否有这个doc_id
    from xlm.utils.optimized_data_loader import OptimizedDataLoader
    
    data_loader = OptimizedDataLoader(
        data_dir="data",
        max_samples=100,
        chinese_document_level=True,
        english_chunk_level=True,
        include_eval_data=False
    )
    
    chinese_docs = data_loader.chinese_docs
    print(f"\n知识库中文文档数量: {len(chinese_docs)}")
    
    # 查找doc_id=7785的文档
    target_doc_id = 7785
    found_docs = []
    
    for i, doc in enumerate(chinese_docs):
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                doc_id = doc_data.get('doc_id')
                if doc_id == target_doc_id:
                    found_docs.append((i, doc))
        except:
            pass
    
    print(f"\n找到doc_id={target_doc_id}的文档数量: {len(found_docs)}")
    
    if found_docs:
        for idx, doc in found_docs:
            print(f"\n文档 {idx}:")
            print(f"  内容: {doc.content[:200]}...")
    else:
        print(f"❌ 知识库中没有doc_id={target_doc_id}的文档")
        
        # 检查知识库中的doc_id分布
        print(f"\n检查知识库中的doc_id分布:")
        doc_ids = []
        for doc in chinese_docs[:20]:  # 只检查前20个
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_id = doc_data.get('doc_id')
                    if doc_id:
                        doc_ids.append(doc_id)
            except:
                pass
        
        print(f"  前20个文档的doc_id: {doc_ids}")
    
    # 测试内容匹配
    print(f"\n=== 测试内容匹配 ===")
    from test_retrieval_mrr import find_correct_document_rank
    
    # 创建包含评估数据的知识库
    from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
    
    # 添加评估数据到知识库
    eval_doc = DocumentWithMetadata(
        content=json.dumps({
            'query': first_sample['query'],
            'context': first_sample['context'],
            'doc_id': first_sample['doc_id'],
            'answer': first_sample['answer']
        }),
        metadata=DocumentMetadata(
            source='eval_alphafin',
            created_at='',
            author='',
            language='chinese'
        )
    )
    
    # 将评估文档添加到知识库末尾
    test_docs = chinese_docs + [eval_doc]
    print(f"测试知识库大小: {len(test_docs)}")
    
    # 测试匹配
    rank = find_correct_document_rank(
        context=first_sample['context'],
        retrieved_docs=test_docs,
        sample=first_sample
    )
    
    print(f"匹配结果: 排名 {rank}")
    
    if rank > 0:
        print(f"✅ 成功找到匹配文档！")
        matched_doc = test_docs[rank-1]
        print(f"匹配文档内容: {matched_doc.content[:200]}...")
    else:
        print(f"❌ 仍然未找到匹配文档")

def test_retrieval_with_eval_data():
    """测试包含评估数据的检索"""
    print(f"\n=== 测试包含评估数据的检索 ===")
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
        
        config = Config()
        
        # 加载编码器
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        
        # 加载少量数据
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=50,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False
        )
        
        chinese_chunks = data_loader.chinese_docs
        
        # 添加评估数据
        with open("evaluate_mrr/alphafin_eval.jsonl", "r", encoding="utf-8") as f:
            first_sample = json.loads(f.readline().strip())
        
        eval_doc = DocumentWithMetadata(
            content=json.dumps({
                'query': first_sample['query'],
                'context': first_sample['context'],
                'doc_id': first_sample['doc_id'],
                'answer': first_sample['answer']
            }),
            metadata=DocumentMetadata(
                source='eval_alphafin',
                created_at='',
                author='',
                language='chinese'
            )
        )
        
        chinese_chunks.append(eval_doc)
        
        # 创建检索器
        retriever = BilingualRetriever(
            encoder_en=encoder_ch,  # 使用中文编码器作为英文编码器（临时）
            encoder_ch=encoder_ch,
            corpus_documents_en=[],
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,
            batch_size=8,
            cache_dir=config.encoder.cache_dir
        )
        
        # 测试检索
        query = first_sample['query']
        print(f"查询: {query}")
        
        retrieved_result = retriever.retrieve(
            text=query, 
            top_k=10, 
            return_scores=True, 
            language='zh'
        )
        
        if isinstance(retrieved_result, tuple):
            retrieved_docs, scores = retrieved_result
        else:
            retrieved_docs = retrieved_result
            scores = []
        
        print(f"检索到 {len(retrieved_docs)} 个文档")
        
        # 检查是否包含评估文档
        from test_retrieval_mrr import find_correct_document_rank
        
        found_rank = find_correct_document_rank(
            context=first_sample['context'],
            retrieved_docs=retrieved_docs,
            sample=first_sample,
            encoder=encoder_ch
        )
        
        print(f"找到正确答案的排名: {found_rank}")
        
        if found_rank > 0:
            print(f"✅ 成功找到正确答案！")
            matched_doc = retrieved_docs[found_rank-1]
            print(f"匹配文档: {matched_doc.content[:200]}...")
        else:
            print(f"❌ 未找到正确答案")
            print(f"前3个检索结果:")
            for i, doc in enumerate(retrieved_docs[:3]):
                print(f"  {i+1}. {doc.content[:100]}...")
                
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_chinese_matching()
    test_retrieval_with_eval_data() 