#!/usr/bin/env python3
"""分析评估数据格式，了解可用的匹配字段"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def analyze_eval_data():
    """分析评估数据格式"""
    print("=== 分析评估数据格式 ===")
    
    # 分析中文评估数据
    print("\n--- 中文评估数据 (AlphaFin) ---")
    try:
        with open("evaluate_mrr/alphafin_eval.jsonl", "r", encoding="utf-8") as f:
            samples = []
            for i, line in enumerate(f):
                if i >= 5:  # 只分析前5个样本
                    break
                if line.strip():
                    sample = json.loads(line)
                    samples.append(sample)
            
            if samples:
                print(f"字段名: {list(samples[0].keys())}")
                print(f"样本数量: {len(samples)}")
                
                for i, sample in enumerate(samples):
                    print(f"\n样本 {i+1}:")
                    for key, value in sample.items():
                        if isinstance(value, str):
                            if len(value) > 100:
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")
                            
    except Exception as e:
        print(f"读取中文评估数据失败: {e}")
    
    # 分析英文评估数据
    print("\n--- 英文评估数据 (TatQA) ---")
    try:
        with open("evaluate_mrr/tatqa_eval.jsonl", "r", encoding="utf-8") as f:
            samples = []
            for i, line in enumerate(f):
                if i >= 5:  # 只分析前5个样本
                    break
                if line.strip():
                    sample = json.loads(line)
                    samples.append(sample)
            
            if samples:
                print(f"字段名: {list(samples[0].keys())}")
                print(f"样本数量: {len(samples)}")
                
                for i, sample in enumerate(samples):
                    print(f"\n样本 {i+1}:")
                    for key, value in sample.items():
                        if isinstance(value, str):
                            if len(value) > 100:
                                print(f"  {key}: {value[:100]}...")
                            else:
                                print(f"  {key}: {value}")
                        else:
                            print(f"  {key}: {value}")
                            
    except Exception as e:
        print(f"读取英文评估数据失败: {e}")

def check_knowledge_base_format():
    """检查知识库文档格式"""
    print("\n=== 检查知识库文档格式 ===")
    
    try:
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        print("加载少量知识库数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=10,  # 只加载10个样本
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False
        )
        
        chinese_docs = data_loader.chinese_docs
        english_docs = data_loader.english_docs
        
        print(f"中文文档数量: {len(chinese_docs)}")
        print(f"英文文档数量: {len(english_docs)}")
        
        if chinese_docs:
            print("\n--- 中文文档示例 ---")
            doc = chinese_docs[0]
            print(f"内容长度: {len(doc.content)}")
            print(f"内容前100字符: {doc.content[:100]}...")
            print(f"元数据: {doc.metadata}")
        
        if english_docs:
            print("\n--- 英文文档示例 ---")
            doc = english_docs[0]
            print(f"内容长度: {len(doc.content)}")
            print(f"内容前100字符: {doc.content[:100]}...")
            print(f"元数据: {doc.metadata}")
            
    except Exception as e:
        print(f"检查知识库格式失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_eval_data()
    check_knowledge_base_format() 