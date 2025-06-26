#!/usr/bin/env python3
"""测试改进后的匹配逻辑"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_matching_strategies():
    """测试各种匹配策略"""
    print("=== 测试改进后的匹配逻辑 ===")
    
    # 模拟评估数据
    eval_sample = {
        'query': '安井食品（603345）何时发布年度报告?',
        'context': '一份发布日期为 2020-04-14 00:00:00 的研究报告，其标题是："安井食品（603345）2019年度报告点评:业绩靓丽灿烂盛开,稳健经营优势尽显"。报告摘要内容：安井食品（603345...',
        'answer': '这个股票的下月最终收益结果是:\'涨\',上涨概率:极大',
        'doc_id': 7785
    }
    
    # 模拟知识库文档（JSON格式）
    from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
    
    # 正确的文档（应该匹配）
    correct_doc = DocumentWithMetadata(
        content=json.dumps({
            'query': '安井食品（603345）何时发布年度报告?',
            'context': '一份发布日期为 2020-04-14 00:00:00 的研究报告，其标题是："安井食品（603345）2019年度报告点评:业绩靓丽灿烂盛开,稳健经营优势尽显"。报告摘要内容：安井食品（603345...',
            'doc_id': 7785
        }),
        metadata=DocumentMetadata(
            source='alphafin',
            created_at='',
            author='',
            language='chinese'
        )
    )
    
    # 错误的文档（不应该匹配）
    wrong_doc = DocumentWithMetadata(
        content=json.dumps({
            'query': '其他问题',
            'context': '其他内容',
            'doc_id': 9999
        }),
        metadata=DocumentMetadata(
            source='alphafin',
            created_at='',
            author='',
            language='chinese'
        )
    )
    
    # 测试匹配函数
    from test_retrieval_mrr import find_correct_document_rank
    
    print("\n1. 测试ID匹配策略")
    retrieved_docs = [wrong_doc, correct_doc]  # 正确的文档在第二位
    rank = find_correct_document_rank(
        context=eval_sample['context'],
        retrieved_docs=retrieved_docs,
        sample=eval_sample
    )
    print(f"   找到的排名: {rank} (期望: 2)")
    print(f"   测试结果: {'✅ 通过' if rank == 2 else '❌ 失败'}")
    
    print("\n2. 测试内容匹配策略")
    # 创建一个没有doc_id的样本
    eval_sample_no_id = {
        'query': '测试问题',
        'context': '这是一个测试内容，用于验证文本匹配功能。',
        'answer': '测试答案'
    }
    
    # 创建匹配的文档
    matching_doc = DocumentWithMetadata(
        content=json.dumps({
            'query': '测试问题',
            'context': '这是一个测试内容，用于验证文本匹配功能。',
        }),
        metadata=DocumentMetadata(
            source='test',
            created_at='',
            author='',
            language='chinese'
        )
    )
    
    retrieved_docs = [wrong_doc, matching_doc]  # 匹配的文档在第二位
    rank = find_correct_document_rank(
        context=eval_sample_no_id['context'],
        retrieved_docs=retrieved_docs,
        sample=eval_sample_no_id
    )
    print(f"   找到的排名: {rank} (期望: 2)")
    print(f"   测试结果: {'✅ 通过' if rank == 2 else '❌ 失败'}")
    
    print("\n3. 测试未找到的情况")
    non_matching_context = "这是一个完全不匹配的内容"
    rank = find_correct_document_rank(
        context=non_matching_context,
        retrieved_docs=retrieved_docs,
        sample=eval_sample_no_id
    )
    print(f"   找到的排名: {rank} (期望: 0)")
    print(f"   测试结果: {'✅ 通过' if rank == 0 else '❌ 失败'}")

def test_real_data_sample():
    """测试真实数据样本"""
    print("\n=== 测试真实数据样本 ===")
    
    try:
        # 加载一个真实的评估样本
        with open("evaluate_mrr/alphafin_eval.jsonl", "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                sample = json.loads(first_line)
                print(f"真实样本:")
                print(f"  Query: {sample['query']}")
                print(f"  Context: {sample['context'][:100]}...")
                print(f"  Doc ID: {sample['doc_id']}")
                
                # 加载少量知识库数据
                from xlm.utils.optimized_data_loader import OptimizedDataLoader
                
                data_loader = OptimizedDataLoader(
                    data_dir="data",
                    max_samples=50,  # 加载50个样本
                    chinese_document_level=True,
                    english_chunk_level=True,
                    include_eval_data=False
                )
                
                chinese_docs = data_loader.chinese_docs
                print(f"\n知识库中文文档数量: {len(chinese_docs)}")
                
                if chinese_docs:
                    # 测试检索和匹配
                    from test_retrieval_mrr import find_correct_document_rank
                    
                    # 模拟检索结果（取前10个文档）
                    retrieved_docs = chinese_docs[:10]
                    rank = find_correct_document_rank(
                        context=sample['context'],
                        retrieved_docs=retrieved_docs,
                        sample=sample
                    )
                    print(f"匹配结果: 排名 {rank}")
                    
                    if rank > 0:
                        print(f"✅ 找到匹配文档！")
                        matched_doc = retrieved_docs[rank-1]
                        print(f"匹配文档内容前100字符: {matched_doc.content[:100]}...")
                    else:
                        print(f"❌ 未找到匹配文档")
                        
    except Exception as e:
        print(f"测试真实数据失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_matching_strategies()
    test_real_data_sample() 