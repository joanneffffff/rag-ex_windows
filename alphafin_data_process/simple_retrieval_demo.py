import json
from pathlib import Path
from typing import List, Dict, Optional
import re
from collections import defaultdict

class SimpleRetrievalDemo:
    """
    简化的多阶段检索演示系统
    不依赖外部库，用于演示检索逻辑
    """
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.data = []
        self.metadata_index = {}
        
        # 加载数据
        self._load_data()
        # 构建元数据索引
        self._build_metadata_index()
    
    def _load_data(self):
        """加载数据"""
        print("正在加载数据...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"加载了 {len(self.data)} 条记录")
    
    def _build_metadata_index(self):
        """构建元数据索引"""
        print("正在构建元数据索引...")
        
        # 按公司名称索引
        self.metadata_index['company_name'] = defaultdict(list)
        # 按股票代码索引
        self.metadata_index['stock_code'] = defaultdict(list)
        # 按报告日期索引
        self.metadata_index['report_date'] = defaultdict(list)
        
        for idx, record in enumerate(self.data):
            # 公司名称索引
            if record.get('company_name'):
                company_name = record['company_name'].strip().lower()
                self.metadata_index['company_name'][company_name].append(idx)
            
            # 股票代码索引
            if record.get('stock_code'):
                stock_code = str(record['stock_code']).strip().lower()
                self.metadata_index['stock_code'][stock_code].append(idx)
            
            # 报告日期索引
            if record.get('report_date'):
                report_date = record['report_date'].strip()
                self.metadata_index['report_date'][report_date].append(idx)
        
        print(f"元数据索引构建完成:")
        print(f"  - 公司名称: {len(self.metadata_index['company_name'])} 个")
        print(f"  - 股票代码: {len(self.metadata_index['stock_code'])} 个")
        print(f"  - 报告日期: {len(self.metadata_index['report_date'])} 个")
    
    def pre_filter(self, 
                   company_name: Optional[str] = None,
                   stock_code: Optional[str] = None,
                   report_date: Optional[str] = None) -> List[int]:
        """基于元数据进行预过滤"""
        candidates = set()
        
        # 如果提供了公司名称
        if company_name:
            company_name_lower = company_name.strip().lower()
            if company_name_lower in self.metadata_index['company_name']:
                candidates.update(self.metadata_index['company_name'][company_name_lower])
        
        # 如果提供了股票代码
        if stock_code:
            stock_code_lower = stock_code.strip().lower()
            if stock_code_lower in self.metadata_index['stock_code']:
                candidates.update(self.metadata_index['stock_code'][stock_code_lower])
        
        # 如果提供了报告日期
        if report_date:
            report_date_clean = report_date.strip()
            if report_date_clean in self.metadata_index['report_date']:
                candidates.update(self.metadata_index['report_date'][report_date_clean])
        
        # 如果没有提供任何元数据，返回所有记录
        if not candidates:
            candidates = set(range(len(self.data)))
        
        print(f"预过滤结果: {len(candidates)} 条候选记录")
        return list(candidates)
    
    def simple_text_search(self, query: str, candidate_indices: List[int], top_k: int = 10) -> List[Dict]:
        """简单的文本搜索（基于关键词匹配）"""
        query_terms = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', query.lower())
        
        results = []
        for idx in candidate_indices:
            if idx >= len(self.data):
                continue
                
            record = self.data[idx]
            score = 0
            
            # 在summary中搜索
            if record.get('summary'):
                summary = record['summary'].lower()
                for term in query_terms:
                    if term in summary:
                        score += 1
            
            # 在generated_question中搜索
            if record.get('generated_question'):
                question = record['generated_question'].lower()
                for term in query_terms:
                    if term in question:
                        score += 1
            
            # 在original_context中搜索
            if record.get('original_context'):
                context = record['original_context'].lower()
                for term in query_terms:
                    if term in context:
                        score += 0.5  # 权重较低
            
            if score > 0:
                results.append({
                    'index': idx,
                    'score': score,
                    'company_name': record.get('company_name'),
                    'stock_code': record.get('stock_code'),
                    'report_date': record.get('report_date'),
                    'summary': record.get('summary', '')[:200] + '...' if len(record.get('summary', '')) > 200 else record.get('summary', ''),
                    'generated_question': record.get('generated_question', ''),
                    'original_context': record.get('original_context', '')[:200] + '...' if len(record.get('original_context', '')) > 200 else record.get('original_context', '')
                })
        
        # 按分数排序
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def search(self, 
               query: str,
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               report_date: Optional[str] = None,
               top_k: int = 10) -> List[Dict]:
        """完整的多阶段检索流程"""
        print(f"\n开始多阶段检索...")
        print(f"查询: {query}")
        if company_name:
            print(f"公司名称: {company_name}")
        if stock_code:
            print(f"股票代码: {stock_code}")
        if report_date:
            print(f"报告日期: {report_date}")
        
        # 1. Pre-filtering
        candidate_indices = self.pre_filter(company_name, stock_code, report_date)
        
        # 2. 简单文本搜索
        results = self.simple_text_search(query, candidate_indices, top_k)
        
        print(f"检索完成，返回 {len(results)} 条结果")
        return results

def main():
    """主函数 - 演示简化检索系统"""
    # 数据文件路径
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    # 初始化检索系统
    print("正在初始化简化检索系统...")
    retrieval_system = SimpleRetrievalDemo(data_path)
    
    # 演示检索
    print("\n" + "="*50)
    print("检索演示")
    print("="*50)
    
    # 示例查询1：基于公司名称的检索
    print("\n示例1: 基于公司名称的检索")
    results1 = retrieval_system.search(
        query="业绩表现",
        company_name="中国宝武",
        top_k=5
    )
    
    for i, result in enumerate(results1):
        print(f"\n结果 {i+1} (分数: {result['score']}):")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  生成问题: {result['generated_question']}")
    
    # 示例查询2：通用检索
    print("\n示例2: 通用检索（无元数据过滤）")
    results2 = retrieval_system.search(
        query="钢铁行业",
        top_k=5
    )
    
    for i, result in enumerate(results2):
        print(f"\n结果 {i+1} (分数: {result['score']}):")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  生成问题: {result['generated_question']}")
    
    # 示例查询3：基于股票代码的检索
    print("\n示例3: 基于股票代码的检索")
    # 先查看有哪些股票代码
    stock_codes = list(retrieval_system.metadata_index['stock_code'].keys())[:5]
    if stock_codes:
        sample_stock_code = stock_codes[0]
        print(f"使用股票代码: {sample_stock_code}")
        
        results3 = retrieval_system.search(
            query="财务报告",
            stock_code=sample_stock_code,
            top_k=3
        )
        
        for i, result in enumerate(results3):
            print(f"\n结果 {i+1} (分数: {result['score']}):")
            print(f"  公司: {result['company_name']}")
            print(f"  股票代码: {result['stock_code']}")
            print(f"  摘要: {result['summary']}")

if __name__ == '__main__':
    main() 