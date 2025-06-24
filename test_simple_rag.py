#!/usr/bin/env python3
"""
简单RAG系统测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """测试基础导入"""
    print("=== 测试基础导入 ===")
    
    try:
        from config.parameters import Config
        print("✅ 配置导入成功")
        
        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
        print("✅ DTO导入成功")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_simple_encoder():
    """测试简单编码器"""
    print("\n=== 测试简单编码器 ===")
    
    try:
        from xlm.components.encoder.encoder import Encoder
        
        # 使用一个简单的模型
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"  # 先用CPU测试
        )
        
        # 测试编码
        texts = ["这是一个测试", "This is a test"]
        embeddings = encoder.encode(texts)
        print(f"✅ 编码成功，嵌入维度: {embeddings.shape}")
        
        return True
    except Exception as e:
        print(f"❌ 编码器测试失败: {e}")
        return False

def test_simple_retriever():
    """测试简单检索器"""
    print("\n=== 测试简单检索器 ===")
    
    try:
        from xlm.components.encoder.encoder import Encoder
        from xlm.components.retriever.sbert_retriever import SBERTRetriever
        from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
        
        # 创建测试文档
        test_docs = [
            DocumentWithMetadata(
                content="净利润是公司在一定期间内的总收入减去总成本后的余额。",
                metadata=DocumentMetadata(source="test", created_at="2024-01-01", author="test")
            ),
            DocumentWithMetadata(
                content="Net income is the total revenue minus total costs of a company over a period.",
                metadata=DocumentMetadata(source="test", created_at="2024-01-01", author="test")
            )
        ]
        
        # 创建编码器
        encoder = Encoder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # 创建检索器
        retriever = SBERTRetriever(
            encoder=encoder,
            corpus_documents=test_docs,
            use_faiss=False  # 先用简单模式
        )
        
        # 测试检索
        query = "什么是净利润？"
        result = retriever.retrieve(query, top_k=1, return_scores=True)
        
        if isinstance(result, tuple):
            docs, scores = result
        else:
            docs = result
            scores = []
        
        print(f"✅ 检索成功，查询: {query}")
        if docs:
            if isinstance(docs, list):
                print(f"  找到 {len(docs)} 个文档")
                print(f"  第一个文档: {docs[0].content[:50]}...")
            else:
                print(f"  找到文档: {docs.content[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ 检索器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 简单RAG系统测试")
    print("=" * 50)
    
    # 1. 测试基础导入
    imports_ok = test_basic_imports()
    
    # 2. 测试编码器
    encoder_ok = test_simple_encoder()
    
    # 3. 测试检索器
    retriever_ok = test_simple_retriever()
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")
    
    print(f"\n测试结果:")
    print(f"  基础导入: {'✅' if imports_ok else '❌'}")
    print(f"  编码器: {'✅' if encoder_ok else '❌'}")
    print(f"  检索器: {'✅' if retriever_ok else '❌'}")
    
    if imports_ok and encoder_ok and retriever_ok:
        print("\n✅ RAG系统基本功能正常！")
        print("\n💡 下一步:")
        print("  1. 运行: python run_enhanced_ui_linux.py")
        print("  2. 或者运行: python test_dual_space_retriever.py")
    else:
        print("\n❌ 系统存在问题，需要修复")

if __name__ == "__main__":
    main() 