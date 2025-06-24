#!/usr/bin/env python3
"""
Linux简化测试脚本 - 测试基础RAG功能
"""

import os
import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """测试基础导入"""
    print("=== 基础导入测试 ===")
    
    try:
        from config.parameters import Config
        print("✅ 配置导入成功")
        
        from xlm.dto.dto import DocumentWithMetadata
        print("✅ DTO导入成功")
        
        from xlm.components.encoder.encoder import Encoder
        print("✅ 编码器导入成功")
        
        from xlm.components.retriever.sbert_retriever import SBERTRetriever
        print("✅ 检索器导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        traceback.print_exc()
        return False

def test_simple_rag():
    """测试简单RAG功能"""
    print("\n=== 简单RAG测试 ===")
    
    try:
        from xlm.components.encoder.encoder import Encoder
        from xlm.components.retriever.sbert_retriever import SBERTRetriever
        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
        
        # 创建测试文档
        test_docs = [
            DocumentWithMetadata(
                content="净利润是公司在一定期间内的总收入减去总成本后的余额。",
                metadata=DocumentMetadata(source="test", created_at="2024", author="test")
            ),
            DocumentWithMetadata(
                content="Net income is the total revenue minus total costs of a company over a period.",
                metadata=DocumentMetadata(source="test", created_at="2024", author="test")
            ),
            DocumentWithMetadata(
                content="营业收入是指企业在正常经营活动中产生的收入。",
                metadata=DocumentMetadata(source="test", created_at="2024", author="test")
            ),
            DocumentWithMetadata(
                content="Revenue refers to income generated from normal business activities.",
                metadata=DocumentMetadata(source="test", created_at="2024", author="test")
            )
        ]
        
        # 初始化编码器
        print("初始化编码器...")
        encoder = Encoder(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # 初始化检索器
        print("初始化检索器...")
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=test_docs,
            use_faiss=False  # 避免FAISS问题
        )
        
        # 测试检索
        print("测试检索功能...")
        queries = [
            "什么是净利润？",
            "What is net income?",
            "营业收入是什么？",
            "What is revenue?"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            try:
                results = retriever.retrieve(text=query, top_k=2)
                print(f"  找到 {len(results)} 个文档:")
                for i, doc in enumerate(results):
                    print(f"    {i+1}. {doc.content[:50]}...")
            except Exception as e:
                print(f"  检索失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单RAG测试失败: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n=== 数据加载测试 ===")
    
    # 检查中文数据
    chinese_data_paths = [
        "data/alphafin/alphafin_rag_ready_generated_cleaned.json",
        "evaluate_mrr/alphafin_train_qc.jsonl"
    ]
    
    print("中文数据文件:")
    for path in chinese_data_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)  # MB
            print(f"  ✅ {path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {path} (不存在)")
    
    # 检查英文数据
    english_data_paths = [
        "data/tatqa_dataset_raw/tatqa_dataset_train.json",
        "evaluate_mrr/tatqa_train_qc.jsonl"
    ]
    
    print("\n英文数据文件:")
    for path in english_data_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)  # MB
            print(f"  ✅ {path} ({size:.1f} MB)")
        else:
            print(f"  ❌ {path} (不存在)")
    
    return chinese_data_paths, english_data_paths

def main():
    """主测试函数"""
    print("🚀 Linux简化RAG系统测试")
    print("=" * 60)
    
    # 1. 测试基础导入
    imports_ok = test_basic_imports()
    
    # 2. 测试数据加载
    chinese_paths, english_paths = test_data_loading()
    
    # 3. 测试简单RAG
    rag_ok = test_simple_rag()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    
    if imports_ok and rag_ok:
        print("✅ 基础功能可以正常运行")
        print("\n💡 下一步:")
        print("  1. 运行: python run_ui.py")
        print("  2. 或者运行: python test_simple_rag.py")
    else:
        print("❌ 系统存在问题，请检查错误信息")
    
    print(f"\n测试结果汇总:")
    print(f"  基础导入: {'✅' if imports_ok else '❌'}")
    print(f"  简单RAG: {'✅' if rag_ok else '❌'}")

if __name__ == "__main__":
    import torch
    main() 