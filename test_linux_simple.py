#!/usr/bin/env python3
"""
简化的Linux GPU环境测试脚本
"""

import os
import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_gpu():
    """检查GPU环境"""
    print("=== GPU环境检查 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ CUDA可用")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name()}")
            return "cuda"
        else:
            print("⚠️  CUDA不可用，使用CPU")
            return "cpu"
    except ImportError:
        print("⚠️  PyTorch未安装")
        return "cpu"
    except Exception as e:
        print(f"⚠️  GPU检查失败: {e}")
        return "cpu"

def check_models():
    """检查模型文件"""
    print("\n=== 模型文件检查 ===")
    
    try:
        from config.parameters import Config
        config = Config()
        
        models = [
            ("中文编码器", config.encoder.chinese_model_path),
            ("英文编码器", config.encoder.english_model_path),
            ("重排序器", config.reranker.model_name),
            ("生成器", config.generator.model_name)
        ]
        
        for name, path in models:
            if os.path.exists(path):
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name}: {path} (不存在)")
        
        return config
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

def check_data():
    """检查数据文件"""
    print("\n=== 数据文件检查 ===")
    
    data_files = [
        ("中文数据1", "data/alphafin/alphafin_rag_ready_generated_cleaned.json"),
        ("中文数据2", "evaluate_mrr/alphafin_train_qc.jsonl"),
        ("英文数据1", "data/tatqa_dataset_raw/tatqa_dataset_train.json"),
        ("英文数据2", "evaluate_mrr/tatqa_train_qc.jsonl")
    ]
    
    available_data = {}
    
    for name, path in data_files:
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024**2)  # MB
            print(f"✅ {name}: {path} ({size:.1f} MB)")
            if "中文" in name:
                available_data["chinese"] = path
            elif "英文" in name:
                available_data["english"] = path
        else:
            print(f"❌ {name}: {path} (不存在)")
    
    return available_data

def test_basic_retrieval():
    """测试基本检索功能"""
    print("\n=== 基本检索测试 ===")
    
    try:
        from xlm.components.encoder.encoder import Encoder
        from xlm.components.retriever.sbert_retriever import SBERTRetriever
        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
        
        # 创建测试文档
        test_docs = [
            DocumentWithMetadata(
                content="这是一个关于净利润的测试文档。净利润是公司收入减去所有费用后的剩余金额。",
                metadata=DocumentMetadata(
                    doc_id="test_1",
                    source="test",
                    language="chinese"
                )
            ),
            DocumentWithMetadata(
                content="This is a test document about net income. Net income is the remaining amount after subtracting all expenses from revenue.",
                metadata=DocumentMetadata(
                    doc_id="test_2",
                    source="test",
                    language="english"
                )
            )
        ]
        
        # 加载编码器
        print("加载编码器...")
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"  # 先用CPU测试
        )
        
        # 创建检索器
        print("创建检索器...")
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=test_docs,
            use_faiss=False  # 先用简单模式
        )
        
        # 测试检索
        test_queries = ["什么是净利润？", "What is net income?"]
        
        for query in test_queries:
            try:
                docs, scores = retriever.retrieve(query, top_k=2, return_scores=True)
                print(f"✅ '{query}' -> 检索到 {len(docs)} 个文档")
                if docs:
                    print(f"   最高分数: {scores[0]:.4f}")
                    print(f"   文档内容: {docs[0].content[:100]}...")
            except Exception as e:
                print(f"❌ '{query}' -> 检索失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本检索测试失败: {e}")
        traceback.print_exc()
        return False

def test_enhanced_system(config, data_files):
    """测试增强系统"""
    print("\n=== 增强系统测试 ===")
    
    try:
        from xlm.registry.retriever import load_enhanced_retriever
        
        print("加载增强检索器...")
        
        # 准备数据路径
        chinese_path = data_files.get("chinese", "")
        english_path = data_files.get("english", "")
        
        if not chinese_path and not english_path:
            print("⚠️  没有可用的数据文件，跳过增强系统测试")
            return False
        
        # 加载增强检索器
        retriever = load_enhanced_retriever(
            config=config,
            chinese_data_path=chinese_path if chinese_path else None,
            english_data_path=english_path if english_path else None
        )
        
        print("✅ 增强检索器加载成功")
        
        # 测试查询
        test_queries = [
            "什么是净利润？",
            "What is net income?",
            "公司的营业收入是多少？",
            "What is the company's revenue?"
        ]
        
        for query in test_queries:
            try:
                docs, scores = retriever.retrieve(query, top_k=3, return_scores=True)
                print(f"✅ '{query}' -> 检索到 {len(docs)} 个文档")
                if docs:
                    print(f"   最高分数: {scores[0]:.4f}")
            except Exception as e:
                print(f"❌ '{query}' -> 检索失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 增强系统测试失败: {e}")
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 Linux GPU环境RAG系统测试")
    print("=" * 60)
    
    # 1. 检查GPU
    device = check_gpu()
    
    # 2. 检查模型
    config = check_models()
    if not config:
        print("❌ 配置加载失败")
        return
    
    # 3. 检查数据
    data_files = check_data()
    
    # 4. 基本检索测试
    basic_ok = test_basic_retrieval()
    
    # 5. 增强系统测试
    enhanced_ok = test_enhanced_system(config, data_files)
    
    # 总结
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    print(f"设备: {device}")
    print(f"基本检索: {'✅' if basic_ok else '❌'}")
    print(f"增强系统: {'✅' if enhanced_ok else '❌'}")
    
    if basic_ok and enhanced_ok:
        print("\n✅ 系统可以正常运行！")
        print("\n💡 下一步:")
        print("  1. 运行: python run_enhanced_ui_linux.py")
        print("  2. 或者运行: python test_dual_space_retriever.py")
    else:
        print("\n❌ 系统存在问题，请检查错误信息")

if __name__ == "__main__":
    main() 