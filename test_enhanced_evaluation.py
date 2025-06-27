#!/usr/bin/env python3
"""
测试增强版评估函数
使用新的relevant_doc_ids进行严格匹配
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from enhanced_evaluation_functions import find_correct_document_rank_enhanced
from xlm.components.retriever.document import DocumentWithMetadata
from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert import FinbertEncoder
from config.parameters import Config

def test_enhanced_matching():
    """测试增强版匹配函数"""
    print("=" * 60)
    print("测试增强版评估函数")
    print("=" * 60)
    
    # 加载配置和编码器
    config = Config()
    encoder = FinbertEncoder(
        model_name="models/finetuned_finbert_tatqa",
        cache_dir=config.encoder.cache_dir,
    )
    
    # 加载评估数据
    eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]
    
    print(f"加载了 {len(eval_data)} 个评估样本")
    
    # 测试前几个样本
    test_samples = eval_data[:5]
    
    for i, sample in enumerate(test_samples):
        print(f"\n--- 测试样本 {i+1} ---")
        print(f"问题: {sample['query']}")
        print(f"正确答案: {sample['answer']}")
        print(f"相关文档ID: {sample.get('relevant_doc_ids', [])}")
        
        # 模拟检索结果（这里用正确答案作为检索结果）
        mock_docs = [
            DocumentWithMetadata(
                content=json.dumps({
                    "context": sample["context"],
                    "doc_id": sample.get("relevant_doc_ids", [""])[0] if sample.get("relevant_doc_ids") else ""
                }),
                metadata={"doc_id": sample.get("relevant_doc_ids", [""])[0] if sample.get("relevant_doc_ids") else ""}
            )
        ]
        
        # 使用增强版函数查找排名
        rank = find_correct_document_rank_enhanced(
            context=sample["context"],
            retrieved_docs=mock_docs,
            sample=sample,
            encoder=encoder
        )
        
        print(f"找到排名: {rank}")
        print(f"匹配成功: {'是' if rank == 1 else '否'}")

def test_with_real_retriever():
    """使用真实检索器测试"""
    print("\n" + "=" * 60)
    print("使用真实检索器测试")
    print("=" * 60)
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        # 加载编码器
        encoder = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        
        # 加载数据
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True
        )
        
        # 创建检索器
        retriever = BilingualRetriever(
            chinese_docs=data_loader.chinese_docs,
            english_docs=data_loader.english_docs,
            encoder=encoder,
            top_k=20
        )
        
        # 加载评估数据
        eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = [json.loads(line) for line in f]
        
        print(f"加载了 {len(eval_data)} 个评估样本")
        
        # 测试前3个样本
        test_samples = eval_data[:3]
        
        for i, sample in enumerate(test_samples):
            print(f"\n--- 真实检索测试 {i+1} ---")
            print(f"问题: {sample['query']}")
            print(f"相关文档ID: {sample.get('relevant_doc_ids', [])}")
            
            try:
                # 真实检索
                retrieved_docs = retriever.retrieve(sample['query'], top_k=20)
                print(f"检索到 {len(retrieved_docs)} 个文档")
                
                # 使用增强版函数查找排名
                rank = find_correct_document_rank_enhanced(
                    context=sample["context"],
                    retrieved_docs=retrieved_docs,
                    sample=sample,
                    encoder=encoder
                )
                
                print(f"找到排名: {rank}")
                print(f"匹配成功: {'是' if rank > 0 else '否'}")
                
                # 显示前3个检索结果的部分内容
                for j, doc in enumerate(retrieved_docs[:3]):
                    doc_content = doc.content
                    try:
                        if doc.content.startswith('{'):
                            doc_data = json.loads(doc.content)
                            doc_content = doc_data.get('context', doc.content)[:100] + "..."
                    except:
                        doc_content = doc.content[:100] + "..."
                    
                    print(f"  排名{j+1}: {doc_content}")
                
            except Exception as e:
                print(f"检索失败: {e}")
    
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_enhanced_matching()
    test_with_real_retriever() 