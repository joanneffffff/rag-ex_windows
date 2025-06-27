#!/usr/bin/env python3
"""
展示原始TatQA数据中一个段落/表格包含多个问题的情况
"""

import json

def show_multi_questions_example():
    """展示一个段落/表格包含多个问题的示例"""
    
    # 加载原始TatQA数据
    with open("data/tatqa_dataset_raw/tatqa_dataset_dev.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 个原始文档")
    
    # 找到包含多个问题的文档
    multi_question_docs = []
    for doc in data:
        if len(doc.get('questions', [])) > 1:
            multi_question_docs.append(doc)
    
    print(f"找到 {len(multi_question_docs)} 个包含多个问题的文档")
    
    # 显示前2个示例
    for i, doc in enumerate(multi_question_docs[:2]):
        print(f"\n{'='*80}")
        print(f"示例 {i+1}: 包含 {len(doc['questions'])} 个问题的文档")
        print(f"{'='*80}")
        
        # 显示段落
        if doc.get('paragraphs'):
            print(f"\n📝 段落内容:")
            for j, para in enumerate(doc['paragraphs']):
                print(f"  段落 {j+1} (order: {para.get('order', 'N/A')}):")
                print(f"    {para['text']}")
                print()
        
        # 显示表格（如果有）
        if doc.get('table'):
            print(f"📊 表格内容:")
            table_data = doc['table'].get('table', [])
            for row in table_data[:5]:  # 只显示前5行
                print(f"    {row}")
            if len(table_data) > 5:
                print(f"    ... 还有 {len(table_data) - 5} 行")
            print()
        
        # 显示所有问题
        print(f"❓ 问题列表:")
        for j, question in enumerate(doc['questions']):
            print(f"  问题 {j+1} (order: {question.get('order', 'N/A')}):")
            print(f"    查询: {question['question']}")
            print(f"    答案: {question['answer']}")
            print(f"    答案类型: {question.get('answer_type', 'N/A')}")
            print(f"    答案来源: {question.get('answer_from', 'N/A')}")
            print(f"    相关段落: {question.get('rel_paragraphs', [])}")
            print(f"    需要比较: {question.get('req_comparison', False)}")
            print(f"    单位: {question.get('scale', 'N/A')}")
            print()
        
        # 分析问题分布
        print(f"📊 问题分布分析:")
        para_questions = {}
        for question in doc['questions']:
            rel_paras = question.get('rel_paragraphs', [])
            for para in rel_paras:
                if para not in para_questions:
                    para_questions[para] = []
                para_questions[para].append(question)
        
        for para_id, questions in para_questions.items():
            print(f"  段落 {para_id}: {len(questions)} 个问题")
            for q in questions:
                print(f"    - {q['question'][:60]}...")
        
        print()

def show_paragraph_multi_questions():
    """专门展示一个段落包含多个问题的情况"""
    
    # 加载原始TatQA数据
    with open("data/tatqa_dataset_raw/tatqa_dataset_dev.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*80}")
    print("专门展示一个段落包含多个问题的情况")
    print(f"{'='*80}")
    
    # 统计每个段落被多少个问题引用
    para_question_count = {}
    
    for doc in data:
        for question in doc.get('questions', []):
            rel_paras = question.get('rel_paragraphs', [])
            for para in rel_paras:
                if para not in para_question_count:
                    para_question_count[para] = []
                para_question_count[para].append(question)
    
    # 找出被多个问题引用的段落
    multi_question_paras = {para: questions for para, questions in para_question_count.items() if len(questions) > 1}
    
    print(f"找到 {len(multi_question_paras)} 个被多个问题引用的段落")
    
    # 显示前3个示例
    for i, (para_id, questions) in enumerate(list(multi_question_paras.items())[:3]):
        print(f"\n📝 段落 {para_id} 被 {len(questions)} 个问题引用:")
        
        for j, question in enumerate(questions):
            print(f"  问题 {j+1}: {question['question']}")
            print(f"    答案: {question['answer']}")
            print(f"    答案来源: {question.get('answer_from', 'N/A')}")
            print()

if __name__ == "__main__":
    show_multi_questions_example()
    show_paragraph_multi_questions() 