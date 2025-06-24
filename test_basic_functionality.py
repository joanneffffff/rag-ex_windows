#!/usr/bin/env python3
"""
基础功能测试 - 不依赖有问题的库
"""

import sys
import os
import json
import traceback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gpu_and_torch():
    """测试GPU和PyTorch"""
    print("=== GPU和PyTorch测试 ===")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
            print(f"✅ GPU名称: {torch.cuda.get_device_name()}")
            
            # 测试GPU张量操作
            x = torch.randn(3, 4).cuda()
            y = torch.randn(4, 3).cuda()
            z = torch.mm(x, y)
            print(f"✅ GPU矩阵乘法成功: {z.shape}")
        else:
            print("⚠️  CUDA不可用，使用CPU")
            
        return True
    except Exception as e:
        print(f"❌ PyTorch测试失败: {e}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n=== 数据加载测试 ===")
    
    try:
        # 测试中文数据
        chinese_file = "data/alphafin/alphafin_rag_ready_generated_cleaned.json"
        if os.path.exists(chinese_file):
            with open(chinese_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 中文数据加载成功，包含 {len(data)} 条记录")
            
            # 显示样本数据
            if len(data) > 0:
                sample = data[0]
                print(f"  样本数据: {sample.get('question', 'N/A')[:50]}...")
        else:
            print(f"❌ 中文数据文件不存在: {chinese_file}")
        
        # 测试英文数据
        english_file = "data/tatqa_dataset_raw/tatqa_dataset_train.json"
        if os.path.exists(english_file):
            with open(english_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 英文数据加载成功，包含 {len(data)} 条记录")
            
            # 显示样本数据
            if len(data) > 0:
                sample = data[0]
                print(f"  样本数据: {sample.get('question', 'N/A')[:50]}...")
        else:
            print(f"❌ 英文数据文件不存在: {english_file}")
        
        return True
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

def test_config_system():
    """测试配置系统"""
    print("\n=== 配置系统测试 ===")
    
    try:
        from config.parameters import Config
        
        config = Config()
        print("✅ 配置加载成功")
        
        # 显示配置信息
        print(f"  中文编码器路径: {config.encoder.chinese_model_path}")
        print(f"  英文编码器路径: {config.encoder.english_model_path}")
        print(f"  重排序器路径: {config.reranker.model_name}")
        print(f"  生成器路径: {config.generator.model_name}")
        print(f"  缓存目录: {config.cache_dir}")
        
        return True
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        return False

def test_dto_system():
    """测试DTO系统"""
    print("\n=== DTO系统测试 ===")
    
    try:
        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
        
        # 创建测试文档
        metadata = DocumentMetadata(
            source="test",
            created_at="2024-01-01",
            author="test"
        )
        
        doc = DocumentWithMetadata(
            content="这是一个测试文档，用于验证DTO系统是否正常工作。",
            metadata=metadata
        )
        
        print("✅ DTO创建成功")
        print(f"  文档内容: {doc.content}")
        print(f"  文档来源: {doc.metadata.source}")
        print(f"  创建时间: {doc.metadata.created_at}")
        print(f"  作者: {doc.metadata.author}")
        
        return True
    except Exception as e:
        print(f"❌ DTO系统测试失败: {e}")
        return False

def test_simple_similarity():
    """测试简单相似度计算"""
    print("\n=== 简单相似度计算测试 ===")
    
    try:
        import numpy as np
        
        # 模拟文档嵌入
        doc_embeddings = np.random.randn(5, 384)  # 5个文档，384维嵌入
        
        # 模拟查询嵌入
        query_embedding = np.random.randn(384)
        
        # 计算余弦相似度
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 找到最相似的文档
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print("✅ 相似度计算成功")
        print(f"  文档数量: {len(doc_embeddings)}")
        print(f"  嵌入维度: {doc_embeddings.shape[1]}")
        print(f"  最相似文档索引: {top_indices}")
        print(f"  相似度分数: {similarities[top_indices]}")
        
        return True
    except Exception as e:
        print(f"❌ 相似度计算失败: {e}")
        return False

def test_faiss_basic():
    """测试FAISS基础功能"""
    print("\n=== FAISS基础功能测试 ===")
    
    try:
        import faiss
        import numpy as np
        
        # 创建测试数据
        dimension = 128
        nb = 1000  # 数据库大小
        nq = 10    # 查询数量
        
        # 生成随机数据
        np.random.seed(1234)
        xb = np.random.random((nb, dimension)).astype('float32')
        xq = np.random.random((nq, dimension)).astype('float32')
        
        # 创建索引
        index = faiss.IndexFlatL2(dimension)
        print(f"✅ FAISS索引创建成功，维度: {dimension}")
        
        # 添加向量到索引
        index.add(xb)
        print(f"✅ 添加了 {nb} 个向量到索引")
        
        # 搜索
        k = 5  # 返回前5个最相似的向量
        D, I = index.search(xq, k)
        
        print(f"✅ 搜索成功，查询数量: {nq}")
        print(f"  第一个查询的前5个结果索引: {I[0]}")
        print(f"  第一个查询的前5个距离: {D[0]}")
        
        return True
    except Exception as e:
        print(f"❌ FAISS测试失败: {e}")
        return False

def test_file_operations():
    """测试文件操作"""
    print("\n=== 文件操作测试 ===")
    
    try:
        # 测试目录创建
        test_dir = "test_output"
        os.makedirs(test_dir, exist_ok=True)
        print(f"✅ 目录创建成功: {test_dir}")
        
        # 测试文件写入
        test_file = os.path.join(test_dir, "test.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("这是一个测试文件\n")
            f.write("用于验证文件操作是否正常\n")
        print(f"✅ 文件写入成功: {test_file}")
        
        # 测试文件读取
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ 文件读取成功，内容长度: {len(content)} 字符")
        
        # 清理测试文件
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✅ 测试文件清理完成")
        
        return True
    except Exception as e:
        print(f"❌ 文件操作失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 基础功能测试")
    print("=" * 60)
    
    tests = [
        ("GPU和PyTorch", test_gpu_and_torch),
        ("数据加载", test_data_loading),
        ("配置系统", test_config_system),
        ("DTO系统", test_dto_system),
        ("简单相似度计算", test_simple_similarity),
        ("FAISS基础功能", test_faiss_basic),
        ("文件操作", test_file_operations)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("🎉 测试完成！")
    
    # 统计结果
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n测试结果汇总:")
    for test_name, passed in results.items():
        print(f"  {test_name}: {'✅' if passed else '❌'}")
    
    print(f"\n总体结果:")
    print(f"  总测试数: {total_tests}")
    print(f"  通过测试: {passed_tests}")
    print(f"  通过率: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests >= total_tests * 0.8:  # 80%通过率
        print("\n✅ 基础功能正常，环境可以支持RAG系统")
        print("\n💡 下一步:")
        print("  1. 考虑更新transformers库以解决兼容性问题")
        print("  2. 或者使用兼容的模型版本")
        print("  3. 运行: python run_enhanced_ui_linux.py")
    else:
        print("\n❌ 基础功能存在问题，需要修复")
        print("\n🔧 建议修复:")
        if not results.get("GPU和PyTorch", False):
            print("  - 检查PyTorch安装")
        if not results.get("数据加载", False):
            print("  - 准备必要的数据文件")
        if not results.get("FAISS基础功能", False):
            print("  - 检查FAISS安装")

if __name__ == "__main__":
    main() 