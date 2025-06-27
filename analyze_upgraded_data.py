#!/usr/bin/env python3
"""
分析升级后的TatQA数据
找出来自同一段落/表格的问题示例
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_upgraded_data():
    """分析升级后的数据，找出相关示例"""
    print("=== 分析升级后的TatQA数据 ===")
    
    # 加载升级后的数据
    upgraded_eval_path = "evaluate_mrr/tatqa_eval_upgraded.jsonl"
    
    if not Path(upgraded_eval_path).exists():
        print(f"❌ 升级后的评估数据不存在: {upgraded_eval_path}")
        return
    
    # 加载数据
    data = []
    with open(upgraded_eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ 加载了 {len(data)} 个评估样本")
    
    # 按doc_id分组
    doc_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids:
            # 提取基础doc_id（去掉chunk_id部分）
            base_doc_id = relevant_doc_ids[0].rsplit('_', 1)[0] if '_' in relevant_doc_ids[0] else relevant_doc_ids[0]
            doc_groups[base_doc_id].append(item)
    
    print(f"📊 按文档分组统计:")
    print(f"  总文档数: {len(doc_groups)}")
    
    # 找出包含多个问题的文档
    multi_question_docs = {doc_id: items for doc_id, items in doc_groups.items() if len(items) > 1}
    print(f"  包含多个问题的文档数: {len(multi_question_docs)}")
    
    # 显示前几个多问题文档的示例
    print(f"\n=== 来自同一文档的多个问题示例 ===")
    
    for i, (doc_id, items) in enumerate(list(multi_question_docs.items())[:5]):
        print(f"\n📄 文档 {i+1}: {doc_id}")
        print(f"   包含 {len(items)} 个问题")
        
        # 按chunk_id分组
        chunk_groups = defaultdict(list)
        for item in items:
            relevant_doc_ids = item.get('relevant_doc_ids', [])
            if relevant_doc_ids:
                chunk_id = relevant_doc_ids[0].rsplit('_', 1)[1] if '_' in relevant_doc_ids[0] else 'unknown'
                chunk_groups[chunk_id].append(item)
        
        # 显示每个chunk的问题
        for chunk_id, chunk_items in chunk_groups.items():
            print(f"\n   📍 Chunk: {chunk_id}")
            print(f"   包含 {len(chunk_items)} 个问题:")
            
            for j, item in enumerate(chunk_items[:3]):  # 只显示前3个问题
                print(f"     {j+1}. 问题: {item['query'][:80]}...")
                print(f"        答案: {item['answer'][:50]}...")
                print(f"        相关文档ID: {item['relevant_doc_ids']}")
                print()
            
            if len(chunk_items) > 3:
                print(f"     ... 还有 {len(chunk_items) - 3} 个问题")
    
    # 找出来自同一段落的问题示例
    print(f"\n=== 来自同一段落的问题示例 ===")
    
    para_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids and 'para_' in relevant_doc_ids[0]:
            para_groups[relevant_doc_ids[0]].append(item)
    
    # 显示包含多个问题的段落
    multi_para_questions = {para_id: items for para_id, items in para_groups.items() if len(items) > 1}
    
    for i, (para_id, items) in enumerate(list(multi_para_questions.items())[:3]):
        print(f"\n📝 段落 {i+1}: {para_id}")
        print(f"   包含 {len(items)} 个问题:")
        
        for j, item in enumerate(items):
            print(f"     {j+1}. 问题: {item['query'][:80]}...")
            print(f"        答案: {item['answer'][:50]}...")
            print(f"        上下文: {item['context'][:100]}...")
            print()
    
    # 找出来自同一表格的问题示例
    print(f"\n=== 来自同一表格的问题示例 ===")
    
    table_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids and 'table_' in relevant_doc_ids[0]:
            table_groups[relevant_doc_ids[0]].append(item)
    
    # 显示包含多个问题的表格
    multi_table_questions = {table_id: items for table_id, items in table_groups.items() if len(items) > 1}
    
    for i, (table_id, items) in enumerate(list(multi_table_questions.items())[:3]):
        print(f"\n📊 表格 {i+1}: {table_id}")
        print(f"   包含 {len(items)} 个问题:")
        
        for j, item in enumerate(items):
            print(f"     {j+1}. 问题: {item['query'][:80]}...")
            print(f"        答案: {item['answer'][:50]}...")
            print(f"        上下文: {item['context'][:100]}...")
            print()
    
    # 统计信息
    print(f"\n=== 数据统计 ===")
    print(f"总样本数: {len(data)}")
    print(f"包含relevant_doc_ids的样本数: {sum(1 for item in data if item.get('relevant_doc_ids'))}")
    print(f"段落问题数: {sum(1 for item in data if item.get('relevant_doc_ids') and 'para_' in item['relevant_doc_ids'][0])}")
    print(f"表格问题数: {sum(1 for item in data if item.get('relevant_doc_ids') and 'table_' in item['relevant_doc_ids'][0])}")
    print(f"包含多个问题的段落数: {len(multi_para_questions)}")
    print(f"包含多个问题的表格数: {len(multi_table_questions)}")

def show_specific_examples():
    """显示特定的示例数据"""
    print(f"\n=== 特定示例数据 ===")
    
    # 加载数据
    upgraded_eval_path = "evaluate_mrr/tatqa_eval_upgraded.jsonl"
    data = []
    with open(upgraded_eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # 找出一个包含多个问题的段落示例
    para_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids and 'para_' in relevant_doc_ids[0]:
            para_groups[relevant_doc_ids[0]].append(item)
    
    # 找出包含最多问题的段落
    max_para = max(para_groups.items(), key=lambda x: len(x[1])) if para_groups else None
    
    if max_para:
        para_id, items = max_para
        print(f"\n🎯 包含最多问题的段落: {para_id}")
        print(f"   包含 {len(items)} 个问题")
        print(f"   上下文: {items[0]['context'][:200]}...")
        print()
        
        for i, item in enumerate(items):
            print(f"   问题 {i+1}:")
            print(f"     查询: {item['query']}")
            print(f"     答案: {item['answer']}")
            print(f"     相关文档ID: {item['relevant_doc_ids']}")
            print()
    
    # 找出一个包含多个问题的表格示例
    table_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids and 'table_' in relevant_doc_ids[0]:
            table_groups[relevant_doc_ids[0]].append(item)
    
    # 找出包含最多问题的表格
    max_table = max(table_groups.items(), key=lambda x: len(x[1])) if table_groups else None
    
    if max_table:
        table_id, items = max_table
        print(f"\n🎯 包含最多问题的表格: {table_id}")
        print(f"   包含 {len(items)} 个问题")
        print(f"   上下文: {items[0]['context'][:200]}...")
        print()
        
        for i, item in enumerate(items):
            print(f"   问题 {i+1}:")
            print(f"     查询: {item['query']}")
            print(f"     答案: {item['answer']}")
            print(f"     相关文档ID: {item['relevant_doc_ids']}")
            print()

def main():
    """主函数"""
    print("=== TatQA升级数据分析工具 ===")
    print("分析来自同一段落/表格的问题示例")
    print()
    
    # 1. 分析升级后的数据
    analyze_upgraded_data()
    
    # 2. 显示特定示例
    show_specific_examples()
    
    print(f"\n" + "="*50)
    print("✅ 分析完成！")
    print("\n总结：")
    print("1. 升级后的数据成功添加了relevant_doc_ids字段")
    print("2. 可以清楚地识别来自同一段落或表格的多个问题")
    print("3. 这确保了评估时能够进行严格的doc_id匹配")
    print("4. 避免了因模糊匹配导致的高估问题")

if __name__ == "__main__":
    main() 