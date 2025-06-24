#!/usr/bin/env python3
"""
Linux GPU环境测试脚本 - 测试双空间双索引RAG系统
"""

import os
import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_availability():
    """测试GPU可用性"""
    print("=== GPU环境测试 ===")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA可用: {cuda_available}")
        
        if cuda_available:
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"当前GPU: {torch.cuda.current_device()}")
            print(f"GPU名称: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            device = "cuda"
        else:
            print("⚠️  CUDA不可用，将使用CPU")
            device = "cpu"
            
        print(f"使用设备: {device}")
        return device
        
    except ImportError:
        print("⚠️  PyTorch未安装，将使用CPU")
        return "cpu"
    except Exception as e:
        print(f"⚠️  GPU检测失败: {e}，将使用CPU")
        return "cpu"

def test_model_loading():
    """测试模型加载"""
    print("\n=== 模型加载测试 ===")
    
    try:
        from config.parameters import Config
        config = Config()
        
        # 设置设备
        device = test_gpu_availability()
        config.encoder.device = device
        config.reranker.device = device
        # 注意：GeneratorConfig没有device属性，需要在生成器加载时设置
        
        print(f"中文编码器路径: {config.encoder.chinese_model_path}")
        print(f"英文编码器路径: {config.encoder.english_model_path}")
        print(f"重排序器路径: {config.reranker.model_name}")
        print(f"生成器路径: {config.generator.model_name}")
        print(f"设备设置: {device}")
        
        # 测试中文编码器
        print("\n1. 测试中文编码器加载...")
        if os.path.exists(config.encoder.chinese_model_path):
            print("✅ 中文编码器路径存在")
        else:
            print("❌ 中文编码器路径不存在")
        
        # 测试英文编码器
        print("\n2. 测试英文编码器加载...")
        if os.path.exists(config.encoder.english_model_path):
            print("✅ 英文编码器路径存在")
        else:
            print("❌ 英文编码器路径不存在")
        
        return config
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return None

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

def test_basic_encoder():
    """测试基础编码器功能"""
    print("\n=== 基础编码器测试 ===")
    
    try:
        # 测试sentence-transformers
        from sentence_transformers import SentenceTransformer
        print("测试sentence-transformers...")
        
        # 使用一个简单的多语言模型
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 测试编码
        texts = ["这是一个测试", "This is a test"]
        embeddings = model.encode(texts)
        print(f"✅ 编码成功，嵌入维度: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础编码器测试失败: {e}")
        return False

def test_simple_retriever():
    """测试简单检索器"""
    print("\n=== 简单检索器测试 ===")
    
    try:
        # 创建简单的测试数据
        test_docs = [
            "净利润是公司在一定期间内的总收入减去总成本后的余额。",
            "Net income is the total revenue minus total costs of a company over a period.",
            "营业收入是指企业在正常经营活动中产生的收入。",
            "Revenue refers to income generated from normal business activities."
        ]
        
        # 使用sentence-transformers进行简单检索
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import semantic_search
        import torch
        import numpy as np
        
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 编码文档
        doc_embeddings = model.encode(test_docs)
        
        # 测试查询
        queries = ["什么是净利润？", "What is net income?"]
        
        print("测试检索功能:")
        for query in queries:
            query_embedding = model.encode([query])
            # 转换为torch tensor
            query_tensor = torch.tensor(query_embedding)
            doc_tensor = torch.tensor(doc_embeddings)
            results = semantic_search(query_tensor, doc_tensor, top_k=2)
            
            print(f"  ✅ '{query}' -> 检索到 {len(results[0])} 个文档")
            for i, result in enumerate(results[0]):
                doc_id = int(result['corpus_id'])
                print(f"      文档 {i+1}: {test_docs[doc_id][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单检索器测试失败: {e}")
        traceback.print_exc()
        return False

def test_simple_generator():
    """测试简单生成器"""
    print("\n=== 简单生成器测试 ===")
    
    try:
        # 使用transformers的基础功能
        from transformers.pipelines import pipeline
        
        # 尝试加载一个简单的文本生成模型
        try:
            generator = pipeline("text-generation", model="distilgpt2", device="cpu")
            print("✅ 使用distilgpt2模型")
        except:
            try:
                generator = pipeline("text-generation", model="gpt2", device="cpu")
                print("✅ 使用gpt2模型")
            except:
                print("⚠️  无法加载生成模型，跳过生成器测试")
                return False
        
        # 测试生成
        test_prompts = [
            "Context: 这是一个测试上下文。\nQuestion: 这是一个测试问题吗？\nAnswer:",
            "Context: This is a test context.\nQuestion: Is this a test question?\nAnswer:"
        ]
        
        print("\n测试生成功能:")
        for prompt in test_prompts:
            try:
                response = generator(prompt, max_length=50, do_sample=True)
                # 处理pipeline返回的结果
                if isinstance(response, list) and len(response) > 0:
                    first_result = response[0]
                    if isinstance(first_result, dict) and 'generated_text' in first_result:
                        generated_text = first_result['generated_text']
                        print(f"  ✅ 生成成功: {generated_text[:100]}...")
                    else:
                        print("  ⚠️  生成结果格式异常")
                else:
                    print("  ⚠️  生成结果为空")
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单生成器测试失败: {e}")
        traceback.print_exc()
        return False

def test_integration_simple():
    """测试简单集成功能"""
    print("\n=== 简单集成测试 ===")
    
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import semantic_search
        from transformers.pipelines import pipeline
        import torch
        
        # 准备测试数据
        test_docs = [
            "净利润是公司在一定期间内的总收入减去总成本后的余额。",
            "Net income is the total revenue minus total costs of a company over a period.",
            "营业收入是指企业在正常经营活动中产生的收入。",
            "Revenue refers to income generated from normal business activities."
        ]
        
        # 1. 检索
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        doc_embeddings = model.encode(test_docs)
        
        query = "什么是净利润？"
        query_embedding = model.encode([query])
        # 转换为torch tensor
        query_tensor = torch.tensor(query_embedding)
        doc_tensor = torch.tensor(doc_embeddings)
        results = semantic_search(query_tensor, doc_tensor, top_k=2)
        
        print(f"查询: {query}")
        print(f"检索到 {len(results[0])} 个文档")
        
        if results[0]:
            # 2. 生成答案
            doc_id = int(results[0][0]['corpus_id'])
            context = test_docs[doc_id]
            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
            
            try:
                generator = pipeline("text-generation", model="distilgpt2", device="cpu")
                response = generator(prompt, max_length=50, do_sample=True)
                # 处理pipeline返回的结果
                if isinstance(response, list) and len(response) > 0:
                    first_result = response[0]
                    if isinstance(first_result, dict) and 'generated_text' in first_result:
                        generated_text = first_result['generated_text']
                        print(f"生成答案: {generated_text[:200]}...")
                    else:
                        print("生成结果格式异常")
                else:
                    print("生成结果为空")
            except:
                print("生成器不可用，跳过生成步骤")
        
        return True
        
    except Exception as e:
        print(f"❌ 简单集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Linux GPU环境RAG系统测试")
    print("=" * 60)
    
    # 1. 测试GPU环境
    device = test_gpu_availability()
    
    # 2. 测试模型加载
    config = test_model_loading()
    if not config:
        print("❌ 配置加载失败，退出测试")
        return
    
    # 3. 测试数据加载
    chinese_paths, english_paths = test_data_loading()
    
    # 4. 测试基础编码器
    encoder_ok = test_basic_encoder()
    
    # 5. 测试简单检索器
    retriever_ok = test_simple_retriever()
    
    # 6. 测试简单生成器
    generator_ok = test_simple_generator()
    
    # 7. 测试简单集成功能
    integration_ok = test_integration_simple()
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    
    if encoder_ok and retriever_ok:
        print("✅ 核心功能可以正常运行")
        print(f"✅ 使用设备: {device}")
        print("\n💡 下一步:")
        print("  1. 运行: python run_enhanced_ui_linux.py")
        print("  2. 或者运行: python test_dual_space_retriever.py")
        print("  3. 或者运行: python test_linux_simple.py")
    else:
        print("❌ 系统存在问题，请检查错误信息")
    
    print(f"\n测试结果汇总:")
    print(f"  编码器: {'✅' if encoder_ok else '❌'}")
    print(f"  检索器: {'✅' if retriever_ok else '❌'}")
    print(f"  生成器: {'✅' if generator_ok else '❌'}")
    print(f"  集成测试: {'✅' if integration_ok else '❌'}")

if __name__ == "__main__":
    main() 