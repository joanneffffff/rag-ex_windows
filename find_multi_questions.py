#!/usr/bin/env python3
"""
快速查找包含多个问题的段落示例
"""

import json
from collections import defaultdict

def find_multi_question_examples():
    """查找包含多个问题的段落示例"""
    
    # 加载数据
    data = []
    with open("evaluate_mrr/tatqa_eval_upgraded.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"加载了 {len(data)} 个样本")
    
    # 按relevant_doc_ids分组
    doc_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids:
            doc_groups[relevant_doc_ids[0]].append(item)
    
    # 找出包含多个问题的文档
    multi_questions = {doc_id: items for doc_id, items in doc_groups.items() if len(items) > 1}
    
    print(f"找到 {len(multi_questions)} 个包含多个问题的段落/表格")
    
    # 显示前3个示例
    for i, (doc_id, items) in enumerate(list(multi_questions.items())[:3]):
        print(f"\n{'='*60}")
        print(f"示例 {i+1}: {doc_id}")
        print(f"包含 {len(items)} 个问题")
        print(f"{'='*60}")
        
        # 显示上下文（所有问题共享的上下文）
        print(f"共享上下文: {items[0]['context'][:200]}...")
        print()
        
        # 显示所有问题
        for j, item in enumerate(items):
            print(f"问题 {j+1}:")
            print(f"  查询: {item['query']}")
            print(f"  答案: {item['answer']}")
            print(f"  相关文档ID: {item['relevant_doc_ids']}")
            print()

if __name__ == "__main__":
    find_multi_question_examples() 