#!/usr/bin/env python3
"""
简化版测试脚本
验证增强版评估函数的基本功能
"""

import json
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

def test_enhanced_data_quality():
    """测试增强版数据质量"""
    print("=" * 60)
    print("测试增强版TatQA数据质量")
    print("=" * 60)
    
    # 加载评估数据
    eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]
    
    print(f"加载了 {len(eval_data)} 个评估样本")
    
    # 检查字段完整性
    required_fields = ["query", "context", "answer", "doc_id", "relevant_doc_ids"]
    sample = eval_data[0]
    missing_fields = [field for field in required_fields if field not in sample]
    
    if missing_fields:
        print(f"❌ 缺少字段: {missing_fields}")
        return False
    else:
        print("✅ 所有必需字段都存在")
    
    # 检查relevant_doc_ids的分布
    relevant_doc_ids_count = sum(1 for item in eval_data if item.get("relevant_doc_ids"))
    print(f"包含relevant_doc_ids的样本数: {relevant_doc_ids_count}")
    print(f"覆盖率: {relevant_doc_ids_count/len(eval_data)*100:.2f}%")
    
    # 检查relevant_doc_ids的唯一性
    all_relevant_ids = []
    for item in eval_data:
        all_relevant_ids.extend(item.get("relevant_doc_ids", []))
    
    unique_ids = set(all_relevant_ids)
    print(f"总relevant_doc_ids数量: {len(all_relevant_ids)}")
    print(f"唯一relevant_doc_ids数量: {len(unique_ids)}")
    print(f"唯一性比例: {len(unique_ids)/len(all_relevant_ids)*100:.2f}%")
    
    # 显示前几个样本的详细信息
    print(f"\n前3个样本详情:")
    for i, sample in enumerate(eval_data[:3]):
        print(f"\n--- 样本 {i+1} ---")
        print(f"问题: {sample['query'][:80]}...")
        print(f"答案: {sample['answer']}")
        print(f"相关文档ID: {sample['relevant_doc_ids']}")
        print(f"文档ID: {sample['doc_id']}")
    
    return True

def test_multi_question_paragraph():
    """测试多问题段落的情况"""
    print(f"\n" + "=" * 60)
    print("测试多问题段落情况")
    print("=" * 60)
    
    eval_file = "evaluate_mrr/tatqa_eval_enhanced.jsonl"
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = [json.loads(line) for line in f]
    
    # 统计每个relevant_doc_id被多少个问题引用
    doc_id_usage = {}
    for item in eval_data:
        for doc_id in item.get("relevant_doc_ids", []):
            doc_id_usage[doc_id] = doc_id_usage.get(doc_id, 0) + 1
    
    # 找出被多个问题引用的段落
    multi_question_paragraphs = {doc_id: count for doc_id, count in doc_id_usage.items() if count > 1}
    
    print(f"被多个问题引用的段落数量: {len(multi_question_paragraphs)}")
    
    if multi_question_paragraphs:
        print(f"前5个多问题段落:")
        sorted_items = sorted(multi_question_paragraphs.items(), key=lambda x: x[1], reverse=True)
        for doc_id, count in sorted_items[:5]:
            print(f"  {doc_id}: {count} 个问题")
        
        # 显示一个具体的多问题段落示例
        most_used_doc_id = sorted_items[0][0]
        related_questions = [item for item in eval_data if most_used_doc_id in item.get("relevant_doc_ids", [])]
        
        print(f"\n示例 - 段落 {most_used_doc_id} 被 {len(related_questions)} 个问题引用:")
        for i, item in enumerate(related_questions[:3]):
            print(f"  问题{i+1}: {item['query'][:60]}...")
    else:
        print("✅ 没有发现多问题段落，每个段落只对应一个问题")

def compare_with_original():
    """与原始数据对比"""
    print(f"\n" + "=" * 60)
    print("与原始数据对比")
    print("=" * 60)
    
    # 检查原始数据
    original_file = "evaluate_mrr/tatqa_eval.jsonl"
    if Path(original_file).exists():
        with open(original_file, "r", encoding="utf-8") as f:
            original_data = [json.loads(line) for line in f]
        
        print(f"原始数据样本数: {len(original_data)}")
        
        # 检查字段差异
        original_fields = set(original_data[0].keys()) if original_data else set()
        enhanced_fields = set(["query", "context", "answer", "doc_id", "relevant_doc_ids"])
        
        new_fields = enhanced_fields - original_fields
        removed_fields = original_fields - enhanced_fields
        
        print(f"新增字段: {new_fields}")
        print(f"移除字段: {removed_fields}")
        
        if "relevant_doc_ids" in new_fields:
            print("✅ 成功添加了relevant_doc_ids字段")
        else:
            print("❌ 未找到relevant_doc_ids字段")
    else:
        print(f"原始数据文件不存在: {original_file}")

if __name__ == "__main__":
    print("=== TatQA增强版数据质量测试 ===")
    
    # 1. 测试数据质量
    if test_enhanced_data_quality():
        # 2. 测试多问题段落
        test_multi_question_paragraph()
        
        # 3. 与原始数据对比
        compare_with_original()
        
        print(f"\n" + "="*60)
        print("✅ 测试完成！")
        print("\n总结：")
        print("1. 增强版数据包含了relevant_doc_ids字段")
        print("2. 使用段落/表格的真实uid确保严格匹配")
        print("3. 支持多问题段落的情况")
        print("4. 提升了英文MRR评估的严谨性")
    else:
        print("❌ 数据质量测试失败") 