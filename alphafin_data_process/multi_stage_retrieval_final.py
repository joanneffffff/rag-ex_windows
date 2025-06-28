import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import re
from datetime import datetime
import sys

# 需要安装的依赖：pip install faiss-cpu sentence-transformers torch
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError as e:
    print(f"请安装必要的依赖: pip install faiss-cpu sentence-transformers torch")
    print(f"错误: {e}")
    exit(1)

# 导入现有的QwenReranker
try:
    sys.path.append(str(Path(__file__).parent.parent))
    from xlm.components.retriever.reranker import QwenReranker
    from config.parameters import Config, DEFAULT_CACHE_DIR
except ImportError as e:
    print(f"无法导入QwenReranker或Config: {e}")
    print("请确保xlm目录结构正确")
    exit(1)

class MultiStageRetrievalSystem:
    """
    多阶段检索系统：
    1. Pre-filtering: 基于元数据（仅中文数据支持）
    2. FAISS检索: 基于generated_question和summary生成统一嵌入索引
    3. Reranker: 基于original_context使用Qwen3-0.6B进行重排序
    
    支持英文和中文数据集，使用现有配置的模型
    - 中文数据（AlphaFin）：支持元数据预过滤 + FAISS + Qwen重排序
    - 英文数据（TatQA）：仅支持FAISS + Qwen重排序（无元数据）
    """
    
    def __init__(self, data_path: Path, dataset_type: str = "chinese", use_existing_config: bool = True):
        """
        初始化检索系统
        
        Args:
            data_path: 数据文件路径
            dataset_type: 数据集类型 ("chinese" 或 "english")
            use_existing_config: 是否使用现有配置
        """
        self.data_path = data_path
        self.dataset_type = dataset_type.lower()
        self.data = []
        self.metadata_index = {}  # 元数据索引（仅中文数据）
        self.faiss_index = None
        self.embedding_model = None
        self.qwen_reranker = None
        self.contexts_for_rerank = []
        
        # 使用现有配置
        if use_existing_config:
            self.config = Config()
            print("使用现有配置初始化多阶段检索系统")
        else:
            self.config = None
            print("使用默认配置初始化多阶段检索系统")
        
        # 根据数据集类型选择编码器（使用现有配置）
        if use_existing_config and self.config:
            if self.dataset_type == "chinese":
                # 使用现有的中文微调模型 - 转换为绝对路径
                model_path = self.config.encoder.chinese_model_path
                if not Path(model_path).is_absolute():
                    # 如果是相对路径，转换为绝对路径
                    model_path = str(Path(__file__).parent.parent / model_path)
                self.model_name = model_path
                print(f"使用现有中文编码器: {self.model_name}")
            else:  # english
                # 使用现有的英文微调模型 - 转换为绝对路径
                model_path = self.config.encoder.english_model_path
                if not Path(model_path).is_absolute():
                    # 如果是相对路径，转换为绝对路径
                    model_path = str(Path(__file__).parent.parent / model_path)
                self.model_name = model_path
                print(f"使用现有英文编码器: {self.model_name}")
        else:
            # 回退到默认模型
            if self.dataset_type == "chinese":
                self.model_name = "distiluse-base-multilingual-cased-v2"
            else:
                self.model_name = "all-MiniLM-L6-v2"
            print(f"使用默认编码器: {self.model_name}")
        
        # 加载数据
        self._load_data()
        # 构建元数据索引（仅中文数据）
        if self.dataset_type == "chinese":
            self._build_metadata_index()
        # 初始化嵌入模型
        self._init_embedding_model()
        # 构建FAISS索引
        self._build_faiss_index()
        # 初始化Qwen reranker
        self._init_qwen_reranker()
    
    def _load_data(self):
        """加载数据"""
        print("正在加载数据...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"加载了 {len(self.data)} 条记录")
        print(f"数据集类型: {self.dataset_type}")
        
        # 检查数据格式
        if self.data and isinstance(self.data[0], dict):
            sample_record = self.data[0]
            print(f"数据字段: {list(sample_record.keys())}")
            
            # 检查是否有元数据字段
            has_metadata = any(field in sample_record for field in ['company_name', 'stock_code', 'report_date'])
            print(f"包含元数据字段: {has_metadata}")
    
    def _build_metadata_index(self):
        """构建元数据索引用于pre-filtering（仅中文数据）"""
        if self.dataset_type != "chinese":
            print("非中文数据集，跳过元数据索引构建")
            return
            
        print("正在构建元数据索引...")
        
        # 检查是否有元数据字段
        if not self.data or not isinstance(self.data[0], dict):
            print("数据格式不支持元数据索引")
            return
            
        sample_record = self.data[0]
        has_metadata = any(field in sample_record for field in ['company_name', 'stock_code', 'report_date'])
        
        if not has_metadata:
            print("数据不包含元数据字段，跳过元数据索引构建")
            return
        
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
        print(f"模型类型: {'多语言编码器' if self.dataset_type == 'chinese' else '英文编码器'}")
        
        # 使用现有配置的缓存目录
        cache_dir = None
        if self.config:
            cache_dir = self.config.encoder.cache_dir
            print(f"使用配置的缓存目录: {cache_dir}")
        
        try:
            self.embedding_model = SentenceTransformer(self.model_name, cache_folder=cache_dir)
            print("嵌入模型加载完成")
        except Exception as e:
            print(f"嵌入模型加载失败: {e}")
            print("尝试使用默认模型...")
            try:
                # 回退到默认模型
                if self.dataset_type == "chinese":
                    fallback_model = "distiluse-base-multilingual-cased-v2"
                else:
                    fallback_model = "all-MiniLM-L6-v2"
                print(f"使用回退模型: {fallback_model}")
                self.embedding_model = SentenceTransformer(fallback_model)
                print("回退模型加载成功")
            except Exception as e2:
                print(f"回退模型也加载失败: {e2}")
                self.embedding_model = None
    
    def _build_faiss_index(self):
        """构建FAISS索引"""
        if self.embedding_model is None:
            print("嵌入模型未初始化，跳过FAISS索引构建")
            return
            
        print("正在构建FAISS索引...")
        
        # 准备用于嵌入的文本
        texts_for_embedding = []
        valid_indices = []
        
        for idx, record in enumerate(self.data):
            # 根据数据集类型选择不同的文本组合策略
            if self.dataset_type == "chinese":
                # 中文数据：组合generated_question和summary
                question = record.get('generated_question', '')
                summary = record.get('summary', '')
                
                if question and summary:
                    combined_text = f"Question: {question} Summary: {summary}"
                    texts_for_embedding.append(combined_text)
                    valid_indices.append(idx)
                elif summary:  # 如果没有generated_question，只用summary
                    texts_for_embedding.append(summary)
                    valid_indices.append(idx)
                else:
                    continue
            else:
                # 英文数据：使用context或content字段
                context = record.get('context', '') or record.get('content', '')
                if context:
                    texts_for_embedding.append(context)
                    valid_indices.append(idx)
                else:
                    continue
        
        if not texts_for_embedding:
            print("警告：没有找到有效的文本用于嵌入")
            return
        
        # 生成嵌入向量
        print(f"正在为 {len(texts_for_embedding)} 条记录生成嵌入向量...")
        try:
            # 使用配置的批处理大小
            batch_size = 32
            if self.config:
                batch_size = self.config.encoder.batch_size
            
            embeddings = self.embedding_model.encode(
                texts_for_embedding, 
                show_progress_bar=True,
                batch_size=batch_size
            )
            
            # 创建FAISS索引
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            self.faiss_index.add(embeddings.astype('float32'))
            
            # 保存有效索引的映射
            self.valid_indices = valid_indices
            
            print(f"FAISS索引构建完成，维度: {dimension}")
        except Exception as e:
            print(f"FAISS索引构建失败: {e}")
            self.faiss_index = None
    
    def _init_qwen_reranker(self):
        """初始化Qwen reranker"""
        print("正在初始化Qwen reranker...")
        try:
            # 使用现有配置
            model_name = "Qwen/Qwen3-Reranker-0.6B"
            cache_dir = DEFAULT_CACHE_DIR  # 使用DEFAULT_CACHE_DIR
            use_quantization = True
            quantization_type = "8bit"
            
            if self.config:
                model_name = self.config.reranker.model_name
                cache_dir = self.config.reranker.cache_dir or DEFAULT_CACHE_DIR  # 确保不为None
                use_quantization = self.config.reranker.use_quantization
                quantization_type = self.config.reranker.quantization_type
            
            print(f"使用配置的重排序器: {model_name}")
            print(f"缓存目录: {cache_dir}")
            print(f"量化: {use_quantization} ({quantization_type})")
            
            # 使用现有的QwenReranker
            self.qwen_reranker = QwenReranker(
                model_name=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                cache_dir=cache_dir,
                use_quantization=use_quantization,
                quantization_type=quantization_type
            )
            print("Qwen reranker初始化完成")
        except Exception as e:
            print(f"Qwen reranker初始化失败: {e}")
            self.qwen_reranker = None
    
    def pre_filter(self, 
                   company_name: Optional[str] = None,
                   stock_code: Optional[str] = None,
                   report_date: Optional[str] = None,
                   max_candidates: int = 1000) -> List[int]:
        """
        基于元数据进行预过滤（仅中文数据支持）
        
        Args:
            company_name: 公司名称
            stock_code: 股票代码
            report_date: 报告日期
            max_candidates: 最大候选数量
            
        Returns:
            候选记录的索引列表
        """
        if self.dataset_type != "chinese":
            print("非中文数据集，不支持元数据预过滤，返回所有记录")
            return list(range(len(self.data)))[:max_candidates]
        
        # 检查是否有元数据索引
        if not self.metadata_index:
            print("没有元数据索引，返回所有记录")
            return list(range(len(self.data)))[:max_candidates]
        
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
        if self.faiss_index is None or self.embedding_model is None:
            print("FAISS索引或嵌入模型未初始化")
            return []
        
        # 生成查询嵌入
        try:
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
        except Exception as e:
            print(f"FAISS检索失败: {e}")
            return []
    
    def rerank(self, 
               query: str, 
               candidate_results: List[Tuple[int, float]], 
               top_k: int = 20) -> List[Tuple[int, float, float]]:
        """
        使用Qwen重排序器对候选结果进行重排序
        
        Args:
            query: 查询文本
            candidate_results: 候选结果列表 [(doc_idx, faiss_score), ...]
            top_k: 返回前k个结果
            
        Returns:
            重排序后的结果列表 [(doc_idx, faiss_score, reranker_score), ...]
        """
        if not self.qwen_reranker or not candidate_results:
            print("重排序器不可用或没有候选结果")
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
        
        print(f"开始重排序 {len(candidate_results)} 条候选结果...")
        
        # 准备重排序的文档
        docs_for_rerank = []
        for doc_idx, faiss_score in candidate_results:
            if doc_idx < len(self.contexts_for_rerank):
                docs_for_rerank.append(self.contexts_for_rerank[doc_idx])
            else:
                print(f"警告: 文档索引 {doc_idx} 超出范围")
                docs_for_rerank.append("")
        
        try:
            # 使用较小的批处理大小以减少内存使用
            reranked_docs = self.qwen_reranker.rerank(
                query=query,
                documents=docs_for_rerank,
                batch_size=1  # 使用最小批处理大小
            )
            
            print(f"Qwen重排序结果: {len(reranked_docs)} 条记录")
            
            # 组合结果
            results = []
            for i, (doc_content, reranker_score) in enumerate(reranked_docs):
                if i < len(candidate_results):
                    doc_idx, faiss_score = candidate_results[i]
                    results.append((doc_idx, faiss_score, reranker_score))
            
            # 按重排序分数降序排序
            results.sort(key=lambda x: x[2], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Qwen重排序失败: {e}")
            print("回退到FAISS分数排序")
            # 回退到FAISS分数排序
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
    
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
            company_name: 公司名称（可选，仅中文数据）
            stock_code: 股票代码（可选，仅中文数据）
            report_date: 报告日期（可选，仅中文数据）
            top_k: 返回前k个结果
            
        Returns:
            检索结果列表
        """
        print(f"\n开始多阶段检索...")
        print(f"查询: {query}")
        print(f"数据集类型: {self.dataset_type}")
        
        if self.dataset_type == "chinese":
            if company_name:
                print(f"公司名称: {company_name}")
            if stock_code:
                print(f"股票代码: {stock_code}")
            if report_date:
                print(f"报告日期: {report_date}")
        else:
            print("英文数据集，不支持元数据过滤")
        
        # 使用配置的检索参数
        retrieval_top_k = 100
        rerank_top_k = top_k
        
        if self.config:
            retrieval_top_k = self.config.retriever.retrieval_top_k
            rerank_top_k = self.config.retriever.rerank_top_k
        
        # 1. Pre-filtering（仅中文数据支持）
        candidate_indices = self.pre_filter(company_name, stock_code, report_date)
        
        # 2. FAISS检索
        faiss_results = self.faiss_search(query, candidate_indices, top_k=min(retrieval_top_k, len(candidate_indices)))
        
        # 3. Qwen Reranker
        final_results = self.rerank(query, faiss_results, top_k=rerank_top_k)
        
        # 4. 格式化结果
        formatted_results = []
        for idx, faiss_score, combined_score in final_results:
            record = self.data[idx]
            
            # 根据数据集类型选择不同的字段
            if self.dataset_type == "chinese":
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
            else:
                result = {
                    'index': idx,
                    'faiss_score': faiss_score,
                    'combined_score': combined_score,
                    'context': record.get('context', '')[:200] + '...' if len(record.get('context', '')) > 200 else record.get('context', ''),
                    'content': record.get('content', '')[:200] + '...' if len(record.get('content', '')) > 200 else record.get('content', ''),
                    'uid': record.get('uid', ''),
                    'source': record.get('source', '')
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
        
        # 保存元数据索引（仅中文数据）
        if self.dataset_type == "chinese":
            with open(output_dir / "metadata_index.pkl", 'wb') as f:
                pickle.dump(self.metadata_index, f)
        
        # 保存有效索引映射
        with open(output_dir / "valid_indices.pkl", 'wb') as f:
            pickle.dump(self.valid_indices, f)
        
        # 保存数据集类型信息
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump({
                'dataset_type': self.dataset_type,
                'model_name': self.model_name
            }, f, indent=2)
        
        print(f"索引已保存到: {output_dir}")
    
    def load_index(self, index_dir: Path):
        """从文件加载索引"""
        # 加载数据集信息
        info_path = index_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.dataset_type = info.get('dataset_type', 'chinese')
                self.model_name = info.get('model_name', 'all-MiniLM-L6-v2')
        
        # 加载FAISS索引
        faiss_path = index_dir / "faiss_index.bin"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
        
        # 加载元数据索引（仅中文数据）
        if self.dataset_type == "chinese":
            metadata_path = index_dir / "metadata_index.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.metadata_index = pickle.load(f)
        
        # 加载有效索引映射
        valid_indices_path = index_dir / "valid_indices.pkl"
        if valid_indices_path.exists():
            with open(valid_indices_path, 'rb') as f:
                self.valid_indices = pickle.load(f)
        
        print(f"索引已从 {index_dir} 加载")
        print(f"数据集类型: {self.dataset_type}")

def main():
    """主函数 - 演示多阶段检索系统"""
    # 数据文件路径
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    index_dir = Path("data/alphafin/retrieval_index")
    
    # 初始化检索系统（中文数据）
    print("正在初始化多阶段检索系统（中文数据）...")
    retrieval_system = MultiStageRetrievalSystem(data_path, dataset_type="chinese")
    
    # 保存索引（可选）
    retrieval_system.save_index(index_dir)
    
    # 演示检索
    print("\n" + "="*50)
    print("检索演示")
    print("="*50)
    
    # 示例查询1：基于公司名称的检索（仅中文数据支持）
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