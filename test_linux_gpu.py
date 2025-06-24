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
        config.generator.device = device
        
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

def test_enhanced_retriever(config, chinese_data_path="", english_data_path=""):
    """测试增强检索器"""
    print("\n=== 增强检索器测试 ===")
    
    try:
        from xlm.registry.retriever import load_enhanced_retriever
        
        print("加载增强检索器...")
        
        # 处理数据路径
        chinese_path = chinese_data_path if chinese_data_path else None
        english_path = english_data_path if english_data_path else None
        
        retriever = load_enhanced_retriever(
            config=config,
            chinese_data_path=chinese_path,
            english_data_path=english_path
        )
        
        print("✅ 增强检索器加载成功")
        
        # 测试检索功能
        test_queries = [
            "什么是净利润？",
            "What is net income?",
            "公司的营业收入是多少？",
            "What is the company's revenue?"
        ]
        
        print("\n测试检索功能:")
        for query in test_queries:
            try:
                docs, scores = retriever.retrieve(query, top_k=3, return_scores=True)
                print(f"  ✅ '{query}' -> 检索到 {len(docs)} 个文档")
                if docs:
                    print(f"      最高分数: {scores[0]:.4f}")
            except Exception as e:
                print(f"  ❌ '{query}' -> 检索失败: {e}")
        
        return retriever
        
    except Exception as e:
        print(f"❌ 增强检索器测试失败: {e}")
        traceback.print_exc()
        return None

def test_generator(config):
    """测试生成器"""
    print("\n=== 生成器测试 ===")
    
    try:
        from xlm.registry.generator import load_generator
        
        print("加载生成器...")
        generator = load_generator(
            generator_model_name=config.generator.model_name,
            use_local_llm=True
        )
        
        print("✅ 生成器加载成功")
        
        # 测试生成功能
        test_prompts = [
            "Context: 这是一个测试上下文。\nQuestion: 这是一个测试问题吗？\nAnswer:",
            "Context: This is a test context.\nQuestion: Is this a test question?\nAnswer:"
        ]
        
        print("\n测试生成功能:")
        for prompt in test_prompts:
            try:
                response = generator.generate([prompt])
                print(f"  ✅ 生成成功: {response[:100]}...")
            except Exception as e:
                print(f"  ❌ 生成失败: {e}")
        
        return generator
        
    except Exception as e:
        print(f"❌ 生成器测试失败: {e}")
        traceback.print_exc()
        return None

def test_integration(retriever, generator):
    """测试集成功能"""
    print("\n=== 集成测试 ===")
    
    if not retriever or not generator:
        print("❌ 检索器或生成器未加载，跳过集成测试")
        return
    
    test_queries = [
        "什么是净利润？",
        "What is net income?"
    ]
    
    print("测试完整RAG流程:")
    for query in test_queries:
        try:
            print(f"\n查询: {query}")
            
            # 1. 检索
            docs, scores = retriever.retrieve(query, top_k=2, return_scores=True)
            print(f"  检索到 {len(docs)} 个文档")
            
            if docs:
                # 2. 生成答案
                context = "\n".join([doc.content for doc in docs[:2]])
                prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                
                response = generator.generate([prompt])
                print(f"  生成答案: {response[:200]}...")
            else:
                print("  未检索到相关文档")
                
        except Exception as e:
            print(f"  ❌ 集成测试失败: {e}")

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
    
    # 选择可用的数据文件
    chinese_data = ""
    english_data = ""
    
    for path in chinese_paths:
        if os.path.exists(path):
            chinese_data = path
            break
    
    for path in english_paths:
        if os.path.exists(path):
            english_data = path
            break
    
    print(f"\n选择的数据文件:")
    print(f"  中文: {chinese_data}")
    print(f"  英文: {english_data}")
    
    # 4. 测试增强检索器
    retriever = test_enhanced_retriever(config, chinese_data, english_data)
    
    # 5. 测试生成器
    generator = test_generator(config)
    
    # 6. 测试集成功能
    test_integration(retriever, generator)
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    
    if retriever and generator:
        print("✅ 系统可以正常运行")
        print(f"✅ 使用设备: {device}")
        print("\n💡 下一步:")
        print("  1. 运行: python run_enhanced_ui_linux.py")
        print("  2. 或者运行: python test_dual_space_retriever.py")
    else:
        print("❌ 系统存在问题，请检查错误信息")

if __name__ == "__main__":
    main() 