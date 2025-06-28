import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict
import re
from datetime import datetime

# 需要安装的依赖：pip install faiss-cpu sentence-transformers rank-bm25
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"请安装必要的依赖: pip install faiss-cpu sentence-transformers rank-bm25")
    print(f"错误: {e}")
    exit(1)

class MultiStageRetrievalSystem:
    """
    多阶段检索系统：
    1. Pre-filtering: 基于元数据（公司名称、股票代码、报告日期）进行预过滤
    2. FAISS检索: 基于generate_question和summary生成统一嵌入索引
    3. Reranker: 基于original_context对检索结果进行重排序
    """
    
    def __init__(self, data_path: Path, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化检索系统
        
        Args:
            data_path: 数据文件路径
            model_name: 句子嵌入模型名称
        """
        self.data_path = data_path
        self.model_name = model_name
        self.data = []
        self.metadata_index = {}  # 元数据索引
        self.faiss_index = None
        self.embedding_model = None
        self.bm25_reranker = None
        self.contexts_for_rerank = []
        
        # 加载数据
        self._load_data()
        # 构建元数据索引
        self._build_metadata_index()
        # 初始化嵌入模型
        self._init_embedding_model()
        # 构建FAISS索引
        self._build_faiss_index()
        # 初始化reranker
        self._init_reranker()
    
    def _load_data(self):
        """加载数据"""
        print("正在加载数据...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"加载了 {len(self.data)} 条记录")
    
    def _build_metadata_index(self):
        """构建元数据索引用于pre-filtering"""
        print("正在构建元数据索引...")
        
        # 按公司名称索引
        self.metadata_index['company_name'] = defaultdict(list)
        # 按股票代码索引
        self.metadata_index['stock_code'] = defaultdict(list)
        # 按报告日期索引
        self.metadata_index['report_date'] = defaultdict(list)
        # 按公司名称+股票代码组合索引
        self.metadata_index['company_stock'] = defaultdict(list)
        
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
            
            # 公司名称+股票代码组合索引
            if record.get('company_name') and record.get('stock_code'):
                company_name = record['company_name'].strip().lower()
                stock_code = str(record['stock_code']).strip().lower()
                key = f"{company_name}_{stock_code}"
                self.metadata_index['company_stock'][key].append(idx)
        
        print(f"元数据索引构建完成:")
        print(f"  - 公司名称: {len(self.metadata_index['company_name'])} 个")
        print(f"  - 股票代码: {len(self.metadata_index['stock_code'])} 个")
        print(f"  - 报告日期: {len(self.metadata_index['report_date'])} 个")
        print(f"  - 公司+股票组合: {len(self.metadata_index['company_stock'])} 个")
    
    def _init_embedding_model(self):
        """初始化句子嵌入模型"""
        print(f"正在加载嵌入模型: {self.model_name}")
        self.embedding_model = SentenceTransformer(self.model_name)
        print("嵌入模型加载完成")
    
    def _build_faiss_index(self):
        """构建FAISS索引"""
        print("正在构建FAISS索引...")
        
        # 准备用于嵌入的文本
        texts_for_embedding = []
        valid_indices = []
        
        for idx, record in enumerate(self.data):
            # 组合generate_question和summary作为嵌入文本
            # 注意：字段名是generated_question而不是generate_question
            question = record.get('generated_question', '')
            summary = record.get('summary', '')
            
            if question and summary:
                combined_text = f"Question: {question} Summary: {summary}"
                texts_for_embedding.append(combined_text)
                valid_indices.append(idx)
            elif summary:  # 如果没有generate_question，只用summary
                texts_for_embedding.append(summary)
                valid_indices.append(idx)
            else:
                continue
        
        if not texts_for_embedding:
            print("警告：没有找到有效的文本用于嵌入")
            return
        
        # 生成嵌入向量
        print(f"正在为 {len(texts_for_embedding)} 条记录生成嵌入向量...")
        embeddings = self.embedding_model.encode(texts_for_embedding, show_progress_bar=True)
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        self.faiss_index.add(embeddings.astype('float32'))
        
        # 保存有效索引的映射
        self.valid_indices = valid_indices
        
        print(f"FAISS索引构建完成，维度: {dimension}")
    
    def _init_reranker(self):
        """初始化BM25 reranker"""
        print("正在初始化BM25 reranker...")
        
        # 收集所有original_context用于BM25
        contexts = []
        for record in self.data:
            if record.get('original_context'):
                # 分词处理（简单的中文分词）
                context = record['original_context']
                # 简单的分词：按空格、标点符号分割
                tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', context)
                contexts.append(tokens)
        
        self.bm25_reranker = BM25Okapi(contexts)
        self.contexts_for_rerank = contexts
        print("BM25 reranker初始化完成")
    
    def pre_filter(self, 
                   company_name: Optional[str] = None,
                   stock_code: Optional[str] = None,
                   report_date: Optional[str] = None,
                   max_candidates: int = 1000) -> List[int]:
        """
        基于元数据进行预过滤
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            report_date: 报告日期
            max_candidates: 最大候选数量
            
        Returns:
            候选记录的索引列表
        """
        candidates = set()
        
        # 如果提供了公司名称和股票代码，优先使用组合索引
        if company_name and stock_code:
            key = f"{company_name.strip().lower()}_{stock_code.strip().lower()}"
            if key in self.metadata_index['company_stock']:
                candidates.update(self.metadata_index['company_stock'][key])
        
        # 如果只提供了公司名称
        elif company_name:
            company_name_lower = company_name.strip().lower()
            if company_name_lower in self.metadata_index['company_name']:
                candidates.update(self.metadata_index['company_name'][company_name_lower])
        
        # 如果只提供了股票代码
        elif stock_code:
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
        
        # 限制候选数量
        candidates = list(candidates)[:max_candidates]
        
        print(f"预过滤结果: {len(candidates)} 条候选记录")
        return candidates
    
    def faiss_search(self, query: str, candidate_indices: List[int], top_k: int = 100) -> List[Tuple[int, float]]:
        """
        使用FAISS进行向量检索
        
        Args:
            query: 查询文本
            candidate_indices: 候选记录索引
            top_k: 返回前k个结果
            
        Returns:
            (索引, 相似度分数) 的列表
        """
        if self.faiss_index is None:
            print("FAISS索引未初始化")
            return []
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query])
        
        # 在FAISS中搜索
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        # 将FAISS索引映射回原始数据索引
        results = []
        for faiss_idx, score in zip(indices[0], scores[0]):
            if faiss_idx < len(self.valid_indices):
                original_idx = self.valid_indices[faiss_idx]
                # 检查是否在候选列表中
                if original_idx in candidate_indices:
                    results.append((original_idx, float(score)))
        
        print(f"FAISS检索结果: {len(results)} 条记录")
        return results
    
    def rerank(self, 
               query: str, 
               candidate_results: List[Tuple[int, float]], 
               top_k: int = 20) -> List[Tuple[int, float, float]]:
        """
        使用BM25对结果进行重排序
        
        Args:
            query: 查询文本
            candidate_results: FAISS检索结果 (索引, 相似度分数)
            top_k: 返回前k个结果
            
        Returns:
            (索引, FAISS分数, BM25分数) 的列表
        """
        if not candidate_results:
            return []
        
        # 提取候选记录的original_context
        candidate_contexts = []
        candidate_indices = []
        
        for idx, _ in candidate_results:
            if idx < len(self.data) and self.data[idx].get('original_context'):
                context = self.data[idx]['original_context']
                # 分词处理
                tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', context)
                candidate_contexts.append(tokens)
                candidate_indices.append(idx)
        
        if not candidate_contexts:
            return []
        
        # 计算BM25分数
        query_tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', query)
        bm25_scores = self.bm25_reranker.get_scores(query_tokens)
        
        # 组合FAISS和BM25分数
        final_results = []
        for i, (idx, faiss_score) in enumerate(candidate_results):
            if i < len(candidate_indices) and candidate_indices[i] == idx:
                bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
                # 简单的分数组合：加权平均
                combined_score = 0.7 * faiss_score + 0.3 * bm25_score
                final_results.append((idx, faiss_score, combined_score))
        
        # 按组合分数排序
        final_results.sort(key=lambda x: x[2], reverse=True)
        
        print(f"重排序结果: {len(final_results)} 条记录")
        return final_results[:top_k]
    
    def search(self, 
               query: str,
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               report_date: Optional[str] = None,
               top_k: int = 20) -> List[Dict]:
        """
        完整的多阶段检索流程
        
        Args:
            query: 查询文本
            company_name: 公司名称（可选）
            stock_code: 股票代码（可选）
            report_date: 报告日期（可选）
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表
        """
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
        
        # 2. FAISS检索
        faiss_results = self.faiss_search(query, candidate_indices, top_k=min(100, len(candidate_indices)))
        
        # 3. Reranker
        final_results = self.rerank(query, faiss_results, top_k)
        
        # 4. 格式化结果
        formatted_results = []
        for idx, faiss_score, combined_score in final_results:
            record = self.data[idx]
            result = {
                'index': idx,
                'faiss_score': faiss_score,
                'combined_score': combined_score,
                'company_name': record.get('company_name'),
                'stock_code': record.get('stock_code'),
                'report_date': record.get('report_date'),
                'original_context': record.get('original_context', '')[:200] + '...' if len(record.get('original_context', '')) > 200 else record.get('original_context', ''),
                'summary': record.get('summary', '')[:200] + '...' if len(record.get('summary', '')) > 200 else record.get('summary', ''),
                'generated_question': record.get('generated_question', ''),
                'original_question': record.get('original_question', ''),
                'original_answer': record.get('original_answer', '')
            }
            formatted_results.append(result)
        
        print(f"检索完成，返回 {len(formatted_results)} 条结果")
        return formatted_results
    
    def save_index(self, output_dir: Path):
        """保存索引到文件"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(output_dir / "faiss_index.bin"))
        
        # 保存元数据索引
        with open(output_dir / "metadata_index.pkl", 'wb') as f:
            pickle.dump(self.metadata_index, f)
        
        # 保存有效索引映射
        with open(output_dir / "valid_indices.pkl", 'wb') as f:
            pickle.dump(self.valid_indices, f)
        
        # 保存BM25模型
        if self.bm25_reranker:
            with open(output_dir / "bm25_model.pkl", 'wb') as f:
                pickle.dump(self.bm25_reranker, f)
        
        print(f"索引已保存到: {output_dir}")
    
    def load_index(self, index_dir: Path):
        """从文件加载索引"""
        # 加载FAISS索引
        faiss_path = index_dir / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        # 加载元数据索引
        metadata_path = index_dir / "metadata_index.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                self.metadata_index = pickle.load(f)
        
        # 加载有效索引映射
        valid_indices_path = index_dir / "valid_indices.pkl"
        if valid_indices_path.exists():
            with open(valid_indices_path, 'rb') as f:
                self.valid_indices = pickle.load(f)
        
        # 加载BM25模型
        bm25_path = index_dir / "bm25_model.pkl"
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                self.bm25_reranker = pickle.load(f)
        
        print(f"索引已从 {index_dir} 加载")

def main():
    """主函数 - 演示多阶段检索系统"""
    # 数据文件路径
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    index_dir = Path("data/alphafin/retrieval_index")
    
    # 初始化检索系统
    print("正在初始化多阶段检索系统...")
    retrieval_system = MultiStageRetrievalSystem(data_path)
    
    # 保存索引（可选）
    retrieval_system.save_index(index_dir)
    
    # 演示检索
    print("\n" + "="*50)
    print("检索演示")
    print("="*50)
    
    # 示例查询1：基于公司名称的检索
    print("\n示例1: 基于公司名称的检索")
    results1 = retrieval_system.search(
        query="公司业绩表现如何？",
        company_name="中国宝武",
        top_k=5
    )
    
    for i, result in enumerate(results1):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")
    
    # 示例查询2：通用检索
    print("\n示例2: 通用检索（无元数据过滤）")
    results2 = retrieval_system.search(
        query="钢铁行业发展趋势",
        top_k=5
    )
    
    for i, result in enumerate(results2):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")

if __name__ == '__main__':
    main() 