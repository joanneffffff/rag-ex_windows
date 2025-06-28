#!/usr/bin/env python3
"""
查询优化脚本：实现实体识别和查询优化
1. 提取查询中的公司名和股票代码
2. 优化查询以提高检索精度
3. 测试优化后的检索效果
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

@dataclass
class EntityInfo:
    """实体信息"""
    company_name: Optional[str] = None
    stock_code: Optional[str] = None
    confidence: float = 0.0

class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        # 股票代码模式
        self.stock_code_patterns = [
            r'(\d{6})',  # 6位数字
            r'(\d{6}\.(SZ|SH))',  # 完整股票代码
            r'[（(](\d{6})[)）]',  # 括号中的股票代码
        ]
        
        # 公司名模式
        self.company_patterns = [
            r'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)(\d{6})',  # 公司名+股票代码
            r'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)(\d{6}\.(SZ|SH))',  # 公司名+完整股票代码
        ]
        
        # 常见公司名后缀
        self.company_suffixes = [
            '股份', '集团', '公司', '有限', '科技', '生物', '医药', '电子', '通信',
            '能源', '化工', '机械', '汽车', '地产', '银行', '证券', '保险'
        ]
    
    def extract_entities(self, query: str) -> EntityInfo:
        """提取查询中的实体信息"""
        entity = EntityInfo()
        
        # 1. 提取股票代码
        stock_code = self._extract_stock_code(query)
        if stock_code:
            entity.stock_code = stock_code
            entity.confidence += 0.4
        
        # 2. 提取公司名
        company_name = self._extract_company_name(query, stock_code)
        if company_name:
            entity.company_name = company_name
            entity.confidence += 0.4
        
        # 3. 基于股票代码查找公司名（如果还没找到）
        if stock_code and not company_name:
            company_name = self._lookup_company_by_code(stock_code)
            if company_name:
                entity.company_name = company_name
                entity.confidence += 0.2
        
        return entity
    
    def _extract_stock_code(self, query: str) -> Optional[str]:
        """提取股票代码"""
        for pattern in self.stock_code_patterns:
            match = re.search(pattern, query)
            if match:
                code = match.group(1)
                # 验证是否为有效的股票代码
                if self._is_valid_stock_code(code):
                    return code
        return None
    
    def _extract_company_name(self, query: str, stock_code: Optional[str] = None) -> Optional[str]:
        """提取公司名"""
        # 如果有股票代码，优先基于股票代码提取
        if stock_code:
            for pattern in self.company_patterns:
                match = re.search(pattern, query)
                if match and match.group(2) == stock_code:
                    return match.group(1).strip()
        
        # 通用公司名提取
        # 查找包含公司后缀的短语
        for suffix in self.company_suffixes:
            pattern = rf'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+{suffix})'
            match = re.search(pattern, query)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _is_valid_stock_code(self, code: str) -> bool:
        """验证股票代码是否有效"""
        # 基本验证：6位数字
        if not re.match(r'^\d{6}$', code):
            return False
        
        # 可以添加更多验证逻辑
        # 例如：检查是否在已知股票代码列表中
        return True
    
    def _lookup_company_by_code(self, stock_code: str) -> Optional[str]:
        """根据股票代码查找公司名"""
        # 这里可以维护一个股票代码到公司名的映射
        # 或者从知识库中查找
        known_companies = {
            '000049': '德赛电池',
            '000001': '平安银行',
            '000002': '万科A',
            # 可以添加更多映射
        }
        return known_companies.get(stock_code)
    
    def optimize_query(self, query: str, entity: EntityInfo) -> str:
        """优化查询"""
        optimized_parts = []
        
        # 1. 添加实体信息
        if entity.company_name and entity.stock_code:
            optimized_parts.append(f"{entity.company_name}({entity.stock_code})")
        elif entity.company_name:
            optimized_parts.append(entity.company_name)
        elif entity.stock_code:
            optimized_parts.append(f"股票代码{entity.stock_code}")
        
        # 2. 添加原始查询
        optimized_parts.append(query)
        
        # 3. 添加金融关键词（如果查询中没有）
        financial_keywords = ['收益', '预测', '股价', '财务', '业绩', '分析']
        has_financial_keyword = any(keyword in query for keyword in financial_keywords)
        
        if not has_financial_keyword and entity.confidence > 0.5:
            # 根据查询内容添加相关关键词
            if '预测' in query or '如何' in query:
                optimized_parts.append('收益预测 财务分析')
            elif '收益' in query:
                optimized_parts.append('财务数据 业绩分析')
        
        return ' '.join(optimized_parts)
    
    def create_multiple_queries(self, query: str, entity: EntityInfo) -> List[str]:
        """创建多个查询变体"""
        queries = []
        
        # 原始查询
        queries.append(query)
        
        # 基于实体的查询变体
        if entity.company_name and entity.stock_code:
            # 变体1：公司名+股票代码+原始查询
            queries.append(f"{entity.company_name}({entity.stock_code}) {query}")
            
            # 变体2：股票代码+原始查询
            queries.append(f"{entity.stock_code} {query}")
            
            # 变体3：公司名+原始查询
            queries.append(f"{entity.company_name} {query}")
        
        elif entity.company_name:
            queries.append(f"{entity.company_name} {query}")
        
        elif entity.stock_code:
            queries.append(f"{entity.stock_code} {query}")
        
        return queries

class RetrievalTester:
    """检索测试器"""
    
    def __init__(self, encoder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.encoder = SentenceTransformer(encoder_model_name)
        self.corpus_embeddings = None
        self.corpus_documents = None
    
    def load_corpus(self, corpus_file: str):
        """加载语料库"""
        print(f"加载语料库: {corpus_file}")
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.corpus_documents = []
        for item in data:
            if isinstance(item, dict):
                content = item.get('context', '')
                if content:
                    self.corpus_documents.append(content)
        
        print(f"加载了 {len(self.corpus_documents)} 个文档")
        
        # 编码语料库
        print("编码语料库...")
        self.corpus_embeddings = self.encoder.encode(
            self.corpus_documents, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_tensor=True
        )
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """搜索相关文档"""
        if self.corpus_embeddings is None or self.corpus_documents is None:
            raise ValueError("请先加载语料库")
        
        # 编码查询
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        
        # 计算相似度
        cos_scores = torch.nn.functional.cosine_similarity(
            query_embedding.unsqueeze(0), 
            self.corpus_embeddings
        )
        
        # 获取top-k结果
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus_documents)))
        
        results = []
        for score, idx in zip(top_results.values[0], top_results.indices[0]):
            results.append((self.corpus_documents[idx], score.item()))
        
        return results
    
    def evaluate_query_variants(self, query: str, entity: EntityInfo, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """评估查询变体的检索效果"""
        optimizer = QueryOptimizer()
        query_variants = optimizer.create_multiple_queries(query, entity)
        
        results = {}
        for i, variant in enumerate(query_variants):
            print(f"\n测试查询变体 {i+1}: {variant}")
            try:
                search_results = self.search(variant, top_k)
                results[f"variant_{i+1}"] = search_results
                
                # 打印前3个结果
                print("前3个检索结果:")
                for j, (doc, score) in enumerate(search_results[:3]):
                    print(f"  {j+1}. Score: {score:.4f}")
                    print(f"     Content: {doc[:100]}...")
            except Exception as e:
                print(f"查询变体 {i+1} 失败: {e}")
        
        return results

def main():
    """主函数"""
    # 测试查询
    test_query = "德赛电池(000049)的下一季度收益预测如何？"
    
    print("=== 查询优化测试 ===")
    print(f"原始查询: {test_query}")
    
    # 1. 实体提取
    optimizer = QueryOptimizer()
    entity = optimizer.extract_entities(test_query)
    
    print(f"\n提取的实体:")
    print(f"  公司名: {entity.company_name}")
    print(f"  股票代码: {entity.stock_code}")
    print(f"  置信度: {entity.confidence:.2f}")
    
    # 2. 查询优化
    optimized_query = optimizer.optimize_query(test_query, entity)
    print(f"\n优化后的查询: {optimized_query}")
    
    # 3. 创建查询变体
    query_variants = optimizer.create_multiple_queries(test_query, entity)
    print(f"\n查询变体:")
    for i, variant in enumerate(query_variants):
        print(f"  变体{i+1}: {variant}")
    
    # 4. 检索测试（如果有语料库文件）
    corpus_file = "data/alphafin/alphafin_rag_ready.json"
    try:
        tester = RetrievalTester()
        tester.load_corpus(corpus_file)
        
        print(f"\n=== 检索效果测试 ===")
        results = tester.evaluate_query_variants(test_query, entity, top_k=5)
        
        # 分析结果
        print(f"\n=== 结果分析 ===")
        for variant_name, search_results in results.items():
            print(f"\n{variant_name}:")
            if search_results:
                # 检查是否包含目标公司
                target_found = any(
                    '德赛电池' in doc or '000049' in doc 
                    for doc, _ in search_results
                )
                print(f"  找到目标公司: {'是' if target_found else '否'}")
                print(f"  最高分数: {search_results[0][1]:.4f}")
            else:
                print("  无检索结果")
    
    except FileNotFoundError:
        print(f"\n语料库文件 {corpus_file} 不存在，跳过检索测试")
    except Exception as e:
        print(f"\n检索测试失败: {e}")

if __name__ == "__main__":
    main() 