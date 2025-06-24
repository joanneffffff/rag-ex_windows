#!/usr/bin/env python3
"""
测试双空间双索引检索器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.registry.retriever import load_enhanced_retriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

def create_test_documents():
    """创建测试文档"""
    chinese_docs = [
        DocumentWithMetadata(
            content="安井食品是一家专注于速冻食品生产的企业，主要产品包括火锅料制品、面米制品和菜肴制品。",
            metadata=DocumentMetadata(
                doc_id="test_zh_1",
                source="test",
                language="chinese",
                question="安井食品主要生产什么产品？",
                answer="火锅料制品、面米制品和菜肴制品"
            )
        ),
        DocumentWithMetadata(
            content="2020年安井食品实现营业收入52.66亿元，同比增长23.66%。",
            metadata=DocumentMetadata(
                doc_id="test_zh_2",
                source="test",
                language="chinese",
                question="安井食品2020年营业收入是多少？",
                answer="52.66亿元"
            )
        )
    ]
    
    english_docs = [
        DocumentWithMetadata(
            content="Apple Inc. is an American multinational technology company that specializes in consumer electronics, computer software, and online services.",
            metadata=DocumentMetadata(
                doc_id="test_en_1",
                source="test",
                language="english",
                question="What does Apple Inc. specialize in?",
                answer="consumer electronics, computer software, and online services"
            )
        ),
        DocumentWithMetadata(
            content="In 2023, Apple reported total revenue of $394.33 billion, with iPhone sales accounting for 52% of total revenue.",
            metadata=DocumentMetadata(
                doc_id="test_en_2",
                source="test",
                language="english",
                question="What was Apple's total revenue in 2023?",
                answer="$394.33 billion"
            )
        )
    ]
    
    return chinese_docs, english_docs

def test_dual_space_retriever():
    """测试双空间检索器"""
    print("=== 测试双空间双索引检索器 ===")
    
    # 创建配置
    config = Config()
    config.reranker.enabled = False  # 暂时禁用重排序器以简化测试
    config.retriever.use_faiss = True
    config.retriever.retrieval_top_k = 5
    config.retriever.rerank_top_k = 3
    
    # 创建测试文档
    chinese_docs, english_docs = create_test_documents()
    
    # 创建增强检索器
    print("\n1. 初始化增强检索器...")
    retriever = EnhancedRetriever(
        config=config,
        chinese_documents=chinese_docs,
        english_documents=english_docs
    )
    
    # 测试中文查询
    print("\n2. 测试中文查询...")
    chinese_query = "安井食品主要生产什么产品？"
    chinese_results, chinese_scores = retriever.retrieve(
        text=chinese_query,
        top_k=2,
        return_scores=True
    )
    
    print(f"中文查询: {chinese_query}")
    print(f"检索到 {len(chinese_results)} 个文档:")
    for i, (doc, score) in enumerate(zip(chinese_results, chinese_scores)):
        print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:50]}...")
    
    # 测试英文查询
    print("\n3. 测试英文查询...")
    english_query = "What does Apple Inc. specialize in?"
    english_results, english_scores = retriever.retrieve(
        text=english_query,
        top_k=2,
        return_scores=True
    )
    
    print(f"英文查询: {english_query}")
    print(f"检索到 {len(english_results)} 个文档:")
    for i, (doc, score) in enumerate(zip(english_results, english_scores)):
        print(f"  {i+1}. 分数: {score:.4f}, 内容: {doc.content[:50]}...")
    
    # 测试语料库大小
    print("\n4. 语料库信息...")
    corpus_sizes = retriever.get_corpus_size()
    print(f"中文文档数量: {corpus_sizes['chinese']}")
    print(f"英文文档数量: {corpus_sizes['english']}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    # 导入EnhancedRetriever
    from xlm.components.retriever.enhanced_retriever import EnhancedRetriever
    
    test_dual_space_retriever() 