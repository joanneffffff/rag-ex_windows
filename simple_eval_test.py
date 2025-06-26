#!/usr/bin/env python3
"""简化的评估测试脚本"""

import json
import sys
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).parent))

# 导入必要的类型
from xlm.dto.dto import DocumentWithMetadata

def check_eval_data_format():
    """检查评估数据格式"""
    print("=== 检查评估数据格式 ===")
    
    # 检查中文评估数据
    try:
        with open("evaluate_mrr/alphafin_eval.jsonl", "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                print("中文评估数据字段:", list(data.keys()))
                print("示例question:", data.get('question', '')[:50] + "...")
                print("示例context:", data.get('context', '')[:50] + "...")
    except Exception as e:
        print(f"读取中文评估数据失败: {e}")
    
    # 检查英文评估数据
    try:
        with open("evaluate_mrr/tatqa_eval.jsonl", "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                data = json.loads(first_line)
                print("\n英文评估数据字段:", list(data.keys()))
                print("示例question:", data.get('question', '')[:50] + "...")
                print("示例context:", data.get('context', '')[:50] + "...")
    except Exception as e:
        print(f"读取英文评估数据失败: {e}")

def test_basic_retrieval():
    """测试基本检索功能"""
    print("\n=== 测试基本检索功能 ===")
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("1. 加载编码器...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载少量数据...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=100,  # 只加载100个样本进行测试
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=False
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 数据加载成功:")
        print(f"      中文chunks: {len(chinese_chunks)}")
        print(f"      英文chunks: {len(english_chunks)}")
        
        print("\n3. 创建检索器...")
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
        print("   ✅ 检索器创建成功")
        
        print("\n4. 测试简单检索...")
        # 测试中文检索
        test_query_zh = "什么是股票？"
        result_zh = retriever.retrieve(text=test_query_zh, top_k=3, language='zh')
        print(f"   中文查询: {test_query_zh}")
        print(f"   检索结果数量: {len(result_zh)}")
        if result_zh and len(result_zh) > 0:
            print(f"   第一个结果: {result_zh[0].content[:100]}...")
        
        # 测试英文检索
        test_query_en = "What is stock?"
        result_en = retriever.retrieve(text=test_query_en, top_k=3, language='en')
        print(f"   英文查询: {test_query_en}")
        print(f"   检索结果数量: {len(result_en)}")
        if result_en and len(result_en) > 0:
            print(f"   第一个结果: {result_en[0].content[:100]}...")
        
        print("\n✅ 基本检索功能测试成功！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    check_eval_data_format()
    test_basic_retrieval() 