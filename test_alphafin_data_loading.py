#!/usr/bin/env python3
"""
测试AlphaFin数据加载功能
"""

import json
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_alphafin_questions(data_path: str, max_questions: int = 10):
    """从AlphaFin数据集加载问题"""
    questions = []
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_questions:
                    break
                try:
                    data = json.loads(line.strip())
                    if 'question' in data:
                        questions.append(data['question'])
                    elif 'query' in data:
                        questions.append(data['query'])
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"❌ 加载AlphaFin数据失败: {e}")
        # 使用默认问题作为备选
        questions = [
            "什么是股票投资？",
            "请解释债券的基本概念",
            "基金投资与股票投资有什么区别？",
            "什么是市盈率？",
            "请解释什么是ETF基金"
        ]
    
    print(f"✅ 加载了 {len(questions)} 个问题")
    return questions

def main():
    """主函数"""
    print("🧪 测试AlphaFin数据加载")
    print("=" * 30)
    
    # 测试数据文件路径
    data_paths = [
        "evaluate_mrr/alphafin_train_qc.jsonl",
        "evaluate_mrr/alphafin_eval.jsonl",
        "data/alphafin/alphafin_rag_ready.json"
    ]
    
    for data_path in data_paths:
        print(f"\n📁 测试数据文件: {data_path}")
        
        if os.path.exists(data_path):
            print(f"✅ 文件存在")
            
            # 尝试加载问题
            questions = load_alphafin_questions(data_path, max_questions=5)
            
            print(f"📝 加载的问题:")
            for i, question in enumerate(questions):
                print(f"   {i+1}. {question}")
        else:
            print(f"❌ 文件不存在")
    
    print(f"\n🎉 数据加载测试完成")

if __name__ == "__main__":
    main() 