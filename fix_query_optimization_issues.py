#!/usr/bin/env python3
"""
修复查询优化问题
1. 修复实体提取逻辑
2. 修复查询变体生成
3. 修复检索错误
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

class FixedQueryOptimizer:
    """修复后的查询优化器"""
    
    def __init__(self):
        # 股票代码模式 - 更精确的匹配
        self.stock_code_patterns = [
            r'(\d{6})',  # 6位数字
            r'(\d{6}\.(SZ|SH))',  # 完整股票代码
            r'[（(](\d{6})[)）]',  # 括号中的股票代码
        ]
        
        # 公司名模式 - 修复括号匹配
        self.company_patterns = [
            r'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)[（(](\d{6})[)）]',  # 公司名(股票代码)
            r'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)[（(](\d{6}\.(SZ|SH))[)）]',  # 公司名(完整股票代码)
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
        """提取查询中的实体信息 - 修复版本"""
        entity = EntityInfo()
        
        # 1. 优先提取完整的"公司名(股票代码)"模式
        for pattern in self.company_patterns:
            match = re.search(pattern, query)
            if match:
                company_name = match.group(1).strip()
                stock_code = match.group(2)
                
                # 验证股票代码
                if self._is_valid_stock_code(stock_code):
                    entity.company_name = company_name
                    entity.stock_code = stock_code
                    entity.confidence = 0.9
                    return entity
        
        # 2. 提取股票代码
        stock_code = self._extract_stock_code(query)
        if stock_code:
            entity.stock_code = stock_code
            entity.confidence += 0.4
        
        # 3. 基于股票代码查找公司名
        if stock_code and not entity.company_name:
            company_name = self._lookup_company_by_code(stock_code)
            if company_name:
                entity.company_name = company_name
                entity.confidence += 0.4
        
        # 4. 通用公司名提取（如果没有找到）
        if not entity.company_name:
            company_name = self._extract_company_name_generic(query)
            if company_name:
                entity.company_name = company_name
                entity.confidence += 0.3
        
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
    
    def _extract_company_name_generic(self, query: str) -> Optional[str]:
        """通用公司名提取"""
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
        """创建优化的查询列表 - 修复版本"""
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
        
        # 去重并返回
        unique_queries = []
        seen = set()
        for q in queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)
        
        return unique_queries

class FixedRetrievalTester:
    """修复后的检索测试器"""
    
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
                content = item.get('context', item.get('content', ''))
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
        """搜索相关文档 - 修复版本"""
        if self.corpus_embeddings is None or self.corpus_documents is None:
            raise ValueError("请先加载语料库")
        
        # 编码查询
        query_embedding = self.encoder.encode(query, convert_to_tensor=True)
        
        # 确保查询嵌入是2D张量
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # 计算相似度
        cos_scores = torch.nn.functional.cosine_similarity(
            query_embedding, 
            self.corpus_embeddings
        )
        
        # 获取top-k结果
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.corpus_documents)))
        
        results = []
        for score, idx in zip(top_results.values[0], top_results.indices[0]):
            results.append((self.corpus_documents[idx], score.item()))
        
        return results
    
    def evaluate_query_variants(self, query: str, entity: EntityInfo, top_k: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """评估查询变体的检索效果 - 修复版本"""
        optimizer = FixedQueryOptimizer()
        query_variants = optimizer.create_optimized_queries(query, entity)
        
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
                import traceback
                traceback.print_exc()
        
        return results

def test_fixed_optimization():
    """测试修复后的查询优化"""
    # 测试查询
    test_query = "德赛电池(000049)的下一季度收益预测如何？"
    
    print("=== 修复后的查询优化测试 ===")
    print(f"原始查询: {test_query}")
    
    # 1. 实体提取
    optimizer = FixedQueryOptimizer()
    entity = optimizer.extract_entities(test_query)
    
    print(f"\n提取的实体:")
    print(f"  公司名: {entity.company_name}")
    print(f"  股票代码: {entity.stock_code}")
    print(f"  置信度: {entity.confidence:.2f}")
    
    # 2. 创建查询变体
    query_variants = optimizer.create_optimized_queries(test_query, entity)
    print(f"\n查询变体:")
    for i, variant in enumerate(query_variants):
        print(f"  变体{i+1}: {variant}")
    
    # 3. 检索测试
    corpus_file = "data/alphafin/alphafin_rag_ready.json"
    try:
        tester = FixedRetrievalTester()
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
        print(f"语料库文件 {corpus_file} 不存在，跳过检索测试")
    except Exception as e:
        print(f"检索测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    test_fixed_optimization()

if __name__ == "__main__":
    main() 