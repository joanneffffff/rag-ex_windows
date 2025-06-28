#!/usr/bin/env python3
"""
RAG系统查询优化改进脚本
集成实体识别和查询优化功能到现有RAG系统
"""

import re
import json
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np

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
        
        # 已知股票代码映射
        self.known_companies = {
            '000049': '德赛电池',
            '000001': '平安银行',
            '000002': '万科A',
            '000858': '五粮液',
            '000725': '京东方A',
            '000063': '中兴通讯',
            '000568': '泸州老窖',
            '000596': '古井贡酒',
            '000799': '酒鬼酒',
            '000858': '五粮液',
            '000876': '新希望',
            '000895': '双汇发展',
            '000938': '紫光股份',
            '000961': '中南建设',
            '000977': '浪潮信息',
            '000983': '西山煤电',
            '000998': '隆平高科',
            '001979': '招商蛇口',
            '002001': '新和成',
            '002007': '华兰生物',
            '002008': '大族激光',
            '002024': '苏宁易购',
            '002027': '分众传媒',
            '002142': '宁波银行',
            '002230': '科大讯飞',
            '002241': '歌尔股份',
            '002304': '洋河股份',
            '002415': '海康威视',
            '002594': '比亚迪',
            '002714': '牧原股份',
            '300059': '东方财富',
            '300122': '智飞生物',
            '300142': '沃森生物',
            '300274': '阳光电源',
            '300347': '泰格医药',
            '300498': '温氏股份',
            '300601': '康泰生物',
            '300750': '宁德时代',
            '300760': '迈瑞医疗',
            '600000': '浦发银行',
            '600009': '上海机场',
            '600036': '招商银行',
            '600048': '保利地产',
            '600104': '上汽集团',
            '600276': '恒瑞医药',
            '600309': '万华化学',
            '600519': '贵州茅台',
            '600585': '海螺水泥',
            '600690': '海尔智家',
            '600887': '伊利股份',
            '600900': '长江电力',
            '600958': '东方证券',
            '600999': '招商证券',
            '601012': '隆基绿能',
            '601088': '中国神华',
            '601166': '兴业银行',
            '601318': '中国平安',
            '601398': '工商银行',
            '601601': '中国太保',
            '601668': '中国建筑',
            '601857': '中国石油',
            '601888': '中国中免',
            '601899': '紫金矿业',
            '601919': '中远海控',
            '601988': '中国银行',
            '601989': '中国重工',
            '603259': '药明康德',
            '603288': '海天味业',
            '603501': '韦尔股份',
            '603986': '兆易创新',
            '688111': '金山办公',
            '688012': '中微公司',
            '688981': '中芯国际',
        }
    
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
        
        # 检查是否在已知股票代码列表中
        return code in self.known_companies
    
    def _lookup_company_by_code(self, stock_code: str) -> Optional[str]:
        """根据股票代码查找公司名"""
        return self.known_companies.get(stock_code)
    
    def create_optimized_queries(self, query: str, entity: EntityInfo) -> List[str]:
        """创建优化的查询列表"""
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
            
            # 变体4：精确匹配查询
            queries.append(f"{entity.company_name} {entity.stock_code} 收益预测 财务分析")
        
        elif entity.company_name:
            queries.append(f"{entity.company_name} {query}")
            queries.append(f"{entity.company_name} 收益预测 财务分析")
        
        elif entity.stock_code:
            queries.append(f"{entity.stock_code} {query}")
            queries.append(f"{entity.stock_code} 收益预测 财务分析")
        
        return queries

class EnhancedRetriever:
    """增强的检索器，支持查询优化"""
    
    def __init__(self, encoder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.encoder = SentenceTransformer(encoder_model_name)
        self.query_optimizer = QueryOptimizer()
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
    
    def search_with_optimization(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """使用查询优化进行搜索"""
        if self.corpus_embeddings is None or self.corpus_documents is None:
            raise ValueError("请先加载语料库")
        
        # 1. 实体提取
        entity = self.query_optimizer.extract_entities(query)
        print(f"提取的实体: 公司={entity.company_name}, 股票代码={entity.stock_code}, 置信度={entity.confidence:.2f}")
        
        # 2. 创建优化查询
        optimized_queries = self.query_optimizer.create_optimized_queries(query, entity)
        print(f"生成的查询变体数量: {len(optimized_queries)}")
        
        # 3. 多查询检索
        all_results = []
        for i, opt_query in enumerate(optimized_queries):
            print(f"执行查询变体 {i+1}: {opt_query}")
            
            # 编码查询
            query_embedding = self.encoder.encode(opt_query, convert_to_tensor=True)
            
            # 计算相似度
            cos_scores = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0), 
                self.corpus_embeddings
            )
            
            # 获取top-k结果
            top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus_documents)))
            
            # 添加查询变体信息
            for score, idx in zip(top_results.values[0], top_results.indices[0]):
                all_results.append({
                    'doc': self.corpus_documents[idx],
                    'score': score.item(),
                    'query_variant': i + 1,
                    'original_query': query,
                    'optimized_query': opt_query
                })
        
        # 4. 结果去重和重排序
        unique_results = self._deduplicate_and_rerank(all_results, top_k)
        
        return [(result['doc'], result['score']) for result in unique_results]
    
    def _deduplicate_and_rerank(self, all_results: List[Dict], top_k: int) -> List[Dict]:
        """去重和重排序"""
        # 按文档内容去重
        seen_docs = set()
        unique_results = []
        
        for result in all_results:
            doc_content = result['doc'][:200]  # 使用前200字符作为去重键
            if doc_content not in seen_docs:
                seen_docs.add(doc_content)
                unique_results.append(result)
        
        # 按分数降序排序
        unique_results.sort(key=lambda x: x['score'], reverse=True)
        
        return unique_results[:top_k]

def test_query_optimization():
    """测试查询优化功能"""
    # 测试查询
    test_queries = [
        "德赛电池(000049)的下一季度收益预测如何？",
        "000049的股价走势怎么样？",
        "平安银行的财务表现如何？",
        "贵州茅台的投资价值分析",
        "宁德时代的未来发展前景"
    ]
    
    print("=== 查询优化测试 ===")
    
    # 初始化检索器
    retriever = EnhancedRetriever()
    
    # 尝试加载语料库
    corpus_file = "data/alphafin/alphafin_rag_ready.json"
    try:
        retriever.load_corpus(corpus_file)
        
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"测试查询: {query}")
            print(f"{'='*60}")
            
            try:
                results = retriever.search_with_optimization(query, top_k=5)
                
                print(f"\n检索结果:")
                for i, (doc, score) in enumerate(results):
                    print(f"\n{i+1}. Score: {score:.4f}")
                    print(f"   Content: {doc[:150]}...")
                    
                    # 检查是否包含相关实体
                    if '德赛电池' in query or '000049' in query:
                        if '德赛电池' in doc or '000049' in doc:
                            print("   ✓ 找到目标公司")
                        else:
                            print("   ✗ 未找到目标公司")
                    elif '平安银行' in query or '000001' in query:
                        if '平安银行' in doc or '000001' in doc:
                            print("   ✓ 找到目标公司")
                        else:
                            print("   ✗ 未找到目标公司")
            
            except Exception as e:
                print(f"查询失败: {e}")
    
    except FileNotFoundError:
        print(f"语料库文件 {corpus_file} 不存在")
        print("请确保已生成alphafin_rag_ready.json文件")
    except Exception as e:
        print(f"加载语料库失败: {e}")

def main():
    """主函数"""
    test_query_optimization()

if __name__ == "__main__":
    main() 