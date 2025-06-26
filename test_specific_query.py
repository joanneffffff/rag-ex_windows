#!/usr/bin/env python3
"""测试特定查询的检索质量"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_specific_query():
    """测试特定查询的检索质量"""
    print("=== 测试特定查询检索质量 ===")
    
    # 目标查询
    target_query = "德赛电池(000049)的下一季度收益预测如何？"
    print(f"目标查询: {target_query}")
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("\n1. 加载编码器...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,
        )
        print("   ✅ 编码器加载成功")
        
        print("\n2. 加载知识库（包含训练数据）...")
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,  # 加载所有数据
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True  # 包含评估数据
        )
        
        chinese_chunks = data_loader.chinese_docs
        english_chunks = data_loader.english_docs
        
        print(f"   ✅ 知识库加载成功:")
        print(f"      中文文档数: {len(chinese_chunks)}")
        print(f"      英文文档数: {len(english_chunks)}")
        
        # 检查知识库中是否有德赛电池相关的文档
        print(f"\n3. 搜索知识库中的德赛电池相关文档...")
        desay_docs = []
        for i, doc in enumerate(chinese_chunks):
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    content = doc_data.get('context', '') + doc_data.get('content', '')
                else:
                    content = doc.content
                
                if '德赛电池' in content or '000049' in content:
                    desay_docs.append((i, doc, content))
            except:
                pass
        
        print(f"   找到 {len(desay_docs)} 个德赛电池相关文档")
        
        if desay_docs:
            print(f"   前3个相关文档:")
            for i, (idx, doc, content) in enumerate(desay_docs[:3]):
                print(f"     {i+1}. 位置{idx}: {content[:200]}...")
        
        print(f"\n4. 创建检索器...")
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
        
        print(f"\n5. 测试不同检索参数...")
        
        # 测试不同的top_k值
        top_k_values = [5, 10, 20, 50, 100]
        
        for top_k in top_k_values:
            print(f"\n--- 测试 top_k={top_k} ---")
            
            # 检索
            retrieved_result = retriever.retrieve(
                text=target_query, 
                top_k=top_k, 
                return_scores=True, 
                language='zh'
            )
            
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            print(f"   检索到 {len(retrieved_docs)} 个文档")
            
            # 检查是否包含德赛电池相关文档
            desay_found = False
            desay_rank = 0
            
            for rank, doc in enumerate(retrieved_docs, 1):
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        content = doc_data.get('context', '') + doc_data.get('content', '')
                    else:
                        content = doc.content
                    
                    if '德赛电池' in content or '000049' in content:
                        desay_found = True
                        desay_rank = rank
                        score_info = f" (分数: {scores[rank-1]:.4f})" if scores and rank <= len(scores) else ""
                        print(f"   ✅ 在第{rank}位找到德赛电池相关文档{score_info}")
                        print(f"      内容: {content[:200]}...")
                        break
                except:
                    pass
            
            if not desay_found:
                print(f"   ❌ 未找到德赛电池相关文档")
                
                # 显示前3个检索结果
                print(f"   前3个检索结果:")
                for i, doc in enumerate(retrieved_docs[:3]):
                    try:
                        if doc.content.startswith('{'):
                            doc_data = json.loads(doc.content)
                            content = doc_data.get('context', '') + doc_data.get('content', '')
                        else:
                            content = doc.content
                        
                        score_info = f" (分数: {scores[i]:.4f})" if scores and i < len(scores) else ""
                        print(f"     {i+1}. {content[:100]}...{score_info}")
                    except:
                        print(f"     {i+1}. [解析失败] {doc.content[:100]}...")
        
        print(f"\n6. 尝试改进检索质量...")
        
        # 尝试不同的查询变体
        query_variants = [
            "德赛电池(000049)的下一季度收益预测如何？",
            "德赛电池 000049 下一季度收益预测",
            "德赛电池收益预测",
            "000049 收益预测",
            "德赛电池(000049)业绩预测"
        ]
        
        print(f"   测试不同查询变体...")
        
        for i, query in enumerate(query_variants):
            print(f"\n   变体 {i+1}: {query}")
            
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=20, 
                return_scores=True, 
                language='zh'
            )
            
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            # 检查是否包含德赛电池相关文档
            desay_found = False
            for rank, doc in enumerate(retrieved_docs, 1):
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        content = doc_data.get('context', '') + doc_data.get('content', '')
                    else:
                        content = doc.content
                    
                    if '德赛电池' in content or '000049' in content:
                        desay_found = True
                        score_info = f" (分数: {scores[rank-1]:.4f})" if scores and rank <= len(scores) else ""
                        print(f"     ✅ 在第{rank}位找到{score_info}")
                        break
                except:
                    pass
            
            if not desay_found:
                print(f"     ❌ 未找到")
        
        print(f"\n7. 分析结果...")
        
        if desay_docs:
            print(f"   ✅ 知识库中包含德赛电池相关文档")
            print(f"   💡 建议:")
            print(f"     1. 增加top_k值以提高召回率")
            print(f"     2. 尝试不同的查询变体")
            print(f"     3. 检查编码器是否针对这类查询进行了优化")
            print(f"     4. 考虑使用重排序器提高精度")
        else:
            print(f"   ❌ 知识库中不包含德赛电池相关文档")
            print(f"   💡 建议:")
            print(f"     1. 检查训练数据是否包含德赛电池相关信息")
            print(f"     2. 考虑扩充知识库")
            print(f"     3. 检查数据预处理是否遗漏了相关内容")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_query() 