#!/usr/bin/env python3
"""改进检索质量 - 针对德赛电池查询"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def improve_retrieval_quality():
    """改进检索质量"""
    print("=== 改进检索质量 - 德赛电池查询 ===")
    
    target_query = "德赛电池(000049)的下一季度收益预测如何？"
    print(f"目标查询: {target_query}")
    
    try:
        from config.parameters import Config
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        from xlm.utils.optimized_data_loader import OptimizedDataLoader
        
        config = Config()
        
        print("\n1. 加载系统...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,
        )
        
        data_loader = OptimizedDataLoader(
            data_dir="data",
            max_samples=-1,
            chinese_document_level=True,
            english_chunk_level=True,
            include_eval_data=True
        )
        
        chinese_chunks = data_loader.chinese_docs
        print(f"   知识库大小: {len(chinese_chunks)} 个中文文档")
        
        # 找到德赛电池文档
        desay_battery_docs = []
        desay_weir_docs = []
        
        for i, doc in enumerate(chinese_chunks):
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    content = doc_data.get('context', '') + doc_data.get('content', '')
                else:
                    content = doc.content
                
                if '德赛电池' in content or '000049' in content:
                    desay_battery_docs.append((i, content))
                elif '德赛西威' in content or '002920' in content:
                    desay_weir_docs.append((i, content))
            except:
                pass
        
        print(f"   德赛电池文档数: {len(desay_battery_docs)}")
        print(f"   德赛西威文档数: {len(desay_weir_docs)}")
        
        if desay_battery_docs:
            print(f"   德赛电池文档位置: {[pos for pos, _ in desay_battery_docs]}")
        
        retriever = BilingualRetriever(
            encoder_en=encoder_ch,  # 临时使用中文编码器
            encoder_ch=encoder_ch,
            corpus_documents_en=[],
            corpus_documents_ch=chinese_chunks,
            use_faiss=True,
            use_gpu=False,
            batch_size=8,
            cache_dir=config.encoder.cache_dir
        )
        
        print("\n2. 策略1: 增加top_k值...")
        
        # 测试更大的top_k值
        for top_k in [50, 100, 200, 500]:
            print(f"\n   测试 top_k={top_k}")
            
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
            
            # 检查是否找到德赛电池文档
            found_rank = None
            for rank, doc in enumerate(retrieved_docs, 1):
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        content = doc_data.get('context', '') + doc_data.get('content', '')
                    else:
                        content = doc.content
                    
                    if '德赛电池' in content or '000049' in content:
                        found_rank = rank
                        score_info = f" (分数: {scores[rank-1]:.4f})" if scores and rank <= len(scores) else ""
                        print(f"     ✅ 在第{rank}位找到德赛电池文档{score_info}")
                        break
                except:
                    pass
            
            if not found_rank:
                print(f"     ❌ 未找到德赛电池文档")
        
        print("\n3. 策略2: 优化查询...")
        
        # 尝试更精确的查询
        optimized_queries = [
            "德赛电池 000049 收益预测",
            "德赛电池 000049 业绩预测", 
            "德赛电池 000049 季度报告",
            "德赛电池 000049 财务预测",
            "德赛电池 000049 盈利预测",
            "德赛电池 000049 收入预测",
            "德赛电池 000049 利润预测",
            "德赛电池 000049 营收预测"
        ]
        
        for i, query in enumerate(optimized_queries):
            print(f"\n   查询变体 {i+1}: {query}")
            
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=100, 
                return_scores=True, 
                language='zh'
            )
            
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            # 检查是否找到德赛电池文档
            found_rank = None
            for rank, doc in enumerate(retrieved_docs, 1):
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        content = doc_data.get('context', '') + doc_data.get('content', '')
                    else:
                        content = doc.content
                    
                    if '德赛电池' in content or '000049' in content:
                        found_rank = rank
                        score_info = f" (分数: {scores[rank-1]:.4f})" if scores and rank <= len(scores) else ""
                        print(f"     ✅ 在第{rank}位找到{score_info}")
                        break
                except:
                    pass
            
            if not found_rank:
                print(f"     ❌ 未找到")
        
        print("\n4. 策略3: 分析编码器偏向性...")
        
        # 测试编码器对相似查询的响应
        test_queries = [
            "德赛电池 000049",  # 精确匹配
            "德赛西威 002920",  # 对比查询
            "德赛",             # 模糊匹配
            "电池",             # 行业匹配
            "000049",           # 代码匹配
        ]
        
        for query in test_queries:
            print(f"\n   测试查询: {query}")
            
            retrieved_result = retriever.retrieve(
                text=query, 
                top_k=10, 
                return_scores=True, 
                language='zh'
            )
            
            if isinstance(retrieved_result, tuple):
                retrieved_docs, scores = retrieved_result
            else:
                retrieved_docs = retrieved_result
                scores = []
            
            # 统计结果
            battery_count = 0
            weir_count = 0
            
            for doc in retrieved_docs:
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        content = doc_data.get('context', '') + doc_data.get('content', '')
                    else:
                        content = doc.content
                    
                    if '德赛电池' in content or '000049' in content:
                        battery_count += 1
                    elif '德赛西威' in content or '002920' in content:
                        weir_count += 1
                except:
                    pass
            
            print(f"     德赛电池文档: {battery_count}")
            print(f"     德赛西威文档: {weir_count}")
        
        print("\n5. 策略4: 手动验证德赛电池文档...")
        
        if desay_battery_docs:
            print(f"   德赛电池文档详情:")
            for i, (pos, content) in enumerate(desay_battery_docs):
                print(f"     文档 {i+1} (位置{pos}):")
                print(f"       内容: {content[:300]}...")
                
                # 测试这个文档是否能被检索到
                test_query = "德赛电池 000049"
                retrieved_result = retriever.retrieve(
                    text=test_query, 
                    top_k=500, 
                    return_scores=True, 
                    language='zh'
                )
                
                if isinstance(retrieved_result, tuple):
                    retrieved_docs, scores = retrieved_result
                else:
                    retrieved_docs = retrieved_result
                    scores = []
                
                # 检查这个文档是否在检索结果中
                found = False
                for rank, doc in enumerate(retrieved_docs, 1):
                    if doc.content == content or (doc.content.startswith('{') and json.loads(doc.content).get('context', '') in content):
                        found = True
                        score_info = f" (分数: {scores[rank-1]:.4f})" if scores and rank <= len(scores) else ""
                        print(f"       在检索结果第{rank}位找到{score_info}")
                        break
                
                if not found:
                    print(f"       未在检索结果中找到")
        
        print("\n6. 改进建议...")
        
        print("   问题分析:")
        print("   1. 编码器可能对'德赛西威'更熟悉，因为训练数据中德赛西威样本更多")
        print("   2. '德赛电池'和'德赛西威'都包含'德赛'，编码器可能混淆")
        print("   3. 德赛电池文档在知识库中的位置较后（27232），可能影响检索")
        
        print("\n   改进建议:")
        print("   1. 增加top_k值到500以上")
        print("   2. 使用更精确的查询词，如'000049'或'德赛电池'")
        print("   3. 考虑重新训练编码器，增加德赛电池相关样本")
        print("   4. 使用重排序器提高精度")
        print("   5. 考虑文档预处理，增强关键词权重")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    improve_retrieval_quality() 