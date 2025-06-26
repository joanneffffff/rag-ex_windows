#!/usr/bin/env python3
"""
简化的查询测试脚本
避免复杂的依赖，专注于核心功能测试
"""

import re
from typing import Optional, List
from dataclasses import dataclass

@dataclass
class EntityInfo:
    """实体信息"""
    company_name: Optional[str] = None
    stock_code: Optional[str] = None
    confidence: float = 0.0

class SimpleQueryOptimizer:
    """简化的查询优化器"""
    
    def __init__(self):
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
        """提取查询中的实体信息"""
        entity = EntityInfo()
        
        # 1. 优先提取完整的"公司名(股票代码)"模式
        pattern = r'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)[（(](\d{6})[)）]'
        match = re.search(pattern, query)
        
        if match:
            company_name = match.group(1).strip()
            stock_code = match.group(2)
            
            # 验证股票代码
            if stock_code in self.known_companies:
                entity.company_name = company_name
                entity.stock_code = stock_code
                entity.confidence = 0.9
                return entity
        
        # 2. 提取股票代码
        stock_code_match = re.search(r'(\d{6})', query)
        if stock_code_match:
            stock_code = stock_code_match.group(1)
            if stock_code in self.known_companies:
                entity.stock_code = stock_code
                entity.confidence += 0.4
                
                # 基于股票代码查找公司名
                entity.company_name = self.known_companies[stock_code]
                entity.confidence += 0.4
        
        return entity
    
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
        
        # 去重并返回
        unique_queries = []
        seen = set()
        for q in queries:
            if q not in seen:
                unique_queries.append(q)
                seen.add(q)
        
        return unique_queries

def test_simple_optimization():
    """测试简化的查询优化"""
    # 测试查询
    test_queries = [
        "德赛电池(000049)的下一季度收益预测如何？",
        "000049的股价走势怎么样？",
        "平安银行的财务表现如何？",
        "贵州茅台的投资价值分析",
        "宁德时代的未来发展前景"
    ]
    
    print("=== 简化查询优化测试 ===")
    
    optimizer = SimpleQueryOptimizer()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"测试查询 {i}: {query}")
        print(f"{'='*60}")
        
        # 1. 实体提取
        entity = optimizer.extract_entities(query)
        print(f"提取的实体:")
        print(f"  公司名: {entity.company_name}")
        print(f"  股票代码: {entity.stock_code}")
        print(f"  置信度: {entity.confidence:.2f}")
        
        # 2. 创建查询变体
        query_variants = optimizer.create_optimized_queries(query, entity)
        print(f"\n生成的查询变体:")
        for j, variant in enumerate(query_variants, 1):
            print(f"  变体{j}: {variant}")
        
        # 3. 验证变体质量
        print(f"\n变体质量检查:")
        for j, variant in enumerate(query_variants, 1):
            # 检查是否包含目标实体
            has_company = entity.company_name and entity.company_name in variant
            has_stock_code = entity.stock_code and entity.stock_code in variant
            print(f"  变体{j}: 包含公司名={has_company}, 包含股票代码={has_stock_code}")

def test_entity_extraction_edge_cases():
    """测试实体提取的边界情况"""
    print("\n=== 实体提取边界情况测试 ===")
    
    edge_cases = [
        "德赛电池(000049)的下一季度收益预测如何？",  # 标准格式
        "000049的股价走势怎么样？",  # 只有股票代码
        "德赛电池的财务表现如何？",  # 只有公司名
        "平安银行(000001)和招商银行(600036)哪个更好？",  # 多个实体
        "这个股票的收益预测如何？",  # 模糊指代
        "贵州茅台(600519)和五粮液(000858)的投资价值分析",  # 多个实体
        "没有实体的普通查询",  # 无实体
    ]
    
    optimizer = SimpleQueryOptimizer()
    
    for i, query in enumerate(edge_cases, 1):
        print(f"\n测试用例 {i}: {query}")
        entity = optimizer.extract_entities(query)
        print(f"  结果: 公司={entity.company_name}, 代码={entity.stock_code}, 置信度={entity.confidence:.2f}")

def main():
    """主函数"""
    test_simple_optimization()
    test_entity_extraction_edge_cases()
    
    print("\n=== 测试完成 ===")
    print("如果所有测试都通过，说明查询优化逻辑正确。")
    print("接下来可以集成到实际的RAG系统中。")

if __name__ == "__main__":
    main() 