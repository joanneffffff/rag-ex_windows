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
        初始化多阶段检索系统
        
        Args:
            data_path: 数据文件路径
            dataset_type: 数据集类型 ("chinese" 或 "english")
            use_existing_config: 是否使用现有配置
        """
        self.data_path = data_path
        self.dataset_type = dataset_type
        self.data = []
        self.original_data = []
        self.doc_to_chunks_mapping = {}
        
        # 初始化组件
        self.embedding_model = None
        self.faiss_index = None
        self.qwen_reranker = None
        self.llm_generator = None  # 添加LLM生成器
        self.valid_indices = []
        self.metadata_index = defaultdict(dict)
        
        # 配置
        self.config = None
        self.model_name = "all-MiniLM-L6-v2"
        
        if use_existing_config:
            try:
                from config.parameters import Config
                self.config = Config()
                
                # 根据数据集类型选择编码器
                if self.dataset_type == "chinese":
                    # 使用中文编码器
                    self.model_name = self.config.encoder.chinese_model_path
                    print(f"使用中文编码器: {self.model_name}")
                else:
                    # 使用英文编码器
                    self.model_name = self.config.encoder.english_model_path
                    print(f"使用英文编码器: {self.model_name}")
                
                print("使用现有配置初始化多阶段检索系统")
            except Exception as e:
                print(f"加载配置失败: {e}")
                # 回退到默认模型
                if self.dataset_type == "chinese":
                    self.model_name = "distiluse-base-multilingual-cased-v2"
                    print(f"使用默认中文编码器: {self.model_name}")
                else:
                    self.model_name = "all-MiniLM-L6-v2"
                    print(f"使用默认英文编码器: {self.model_name}")
        
        # 加载数据
        self._load_data()
        
        # 构建索引
        self._build_metadata_index()
        self._init_embedding_model()
        self._build_faiss_index()
        self._init_qwen_reranker()
        self._init_llm_generator()  # 初始化LLM生成器
    
    def _load_data(self):
        """加载数据"""
        print("正在加载数据...")
        
        # 加载原始AlphaFin数据用于FAISS索引
        print("加载原始AlphaFin数据用于FAISS索引...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.original_data = json.load(f)
        print(f"加载了 {len(self.original_data)} 条原始记录")
        
        # 建立doc_id到chunks的映射
        print("建立doc_id到chunks的映射关系...")
        self.doc_to_chunks_mapping = {}
        
        for doc_idx, record in enumerate(self.original_data):
            if self.dataset_type == "chinese":
                # 对于中文数据，生成chunks
                original_context = record.get('original_context', '')
                company_name = record.get('company_name', '公司')
                
                if original_context:
                    # 使用convert_json_context_to_natural_language_chunks函数
                    from xlm.utils.optimized_data_loader import convert_json_context_to_natural_language_chunks
                    chunks = convert_json_context_to_natural_language_chunks(original_context, company_name)
                    
                    if chunks:
                        self.doc_to_chunks_mapping[doc_idx] = chunks
                    else:
                        # 如果没有chunks，使用summary作为fallback
                        self.doc_to_chunks_mapping[doc_idx] = [record.get('summary', '')]
                else:
                    # 如果没有original_context，使用summary
                    self.doc_to_chunks_mapping[doc_idx] = [record.get('summary', '')]
            else:
                # 英文数据，使用context或content
                context = record.get('context', '') or record.get('content', '')
                self.doc_to_chunks_mapping[doc_idx] = [context]
        
        print(f"建立了 {len(self.doc_to_chunks_mapping)} 个doc_id到chunks的映射")
        
        # 统计chunks总数
        total_chunks = sum(len(chunks) for chunks in self.doc_to_chunks_mapping.values())
        print(f"总共生成了 {total_chunks} 个chunks用于重排序")
        
        # 使用原始数据作为主要数据
        self.data = self.original_data
        
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
        if not self.data:
            print("数据格式不支持元数据索引")
            return
            
        # 检查数据格式
        if hasattr(self.data[0], 'content'):
            # DocumentWithMetadata格式
            print("使用DocumentWithMetadata格式，跳过元数据索引构建")
            print("注意：chunk级别的数据不支持元数据预过滤")
            return
        elif isinstance(self.data[0], dict):
            # 字典格式
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
            # 检查是否是微调模型路径
            if "finetuned" in self.model_name or "models/" in self.model_name:
                print("检测到微调模型，使用FinbertEncoder...")
                from xlm.components.encoder.finbert import FinbertEncoder
                self.embedding_model = FinbertEncoder(
                    model_name=self.model_name,
                    cache_dir=cache_dir,
                    device="cuda:0"  # 编码器使用cuda:0
                )
                print("微调模型加载完成 (cuda:0)")
            else:
                # 使用SentenceTransformer加载HuggingFace模型
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(self.model_name, cache_folder=cache_dir)
                # 将模型移动到cuda:0
                if hasattr(self.embedding_model, 'to'):
                    self.embedding_model.to('cuda:0')
                print("HuggingFace模型加载完成 (cuda:0)")
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
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(fallback_model)
                # 将模型移动到cuda:0
                if hasattr(self.embedding_model, 'to'):
                    self.embedding_model.to('cuda:0')
                print("回退模型加载成功 (cuda:0)")
            except Exception as e2:
                print(f"回退模型也加载失败: {e2}")
                self.embedding_model = None
    
    def _build_faiss_index(self):
        """构建FAISS索引"""
        if self.embedding_model is None:
            print("嵌入模型未初始化，跳过FAISS索引构建")
            return
            
        print("正在构建FAISS索引...")
        print("中文数据：使用summary字段进行向量编码")
        print("英文数据：使用context/content字段进行向量编码")
        
        # 准备用于嵌入的文本
        texts_for_embedding = []
        valid_indices = []
        
        for idx, record in enumerate(self.data):
            # 根据数据集类型选择不同的文本组合策略
            if self.dataset_type == "chinese":
                # 中文数据：只使用summary
                summary = record.get('summary', '')
                
                if summary:
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
            print("没有有效的文本用于嵌入")
            return
        
        # 生成嵌入
        print(f"正在编码 {len(texts_for_embedding)} 个文本...")
        embeddings = self.embedding_model.encode(texts_for_embedding, show_progress_bar=True)
        
        # 构建FAISS索引
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        self.faiss_index.add(embeddings.astype('float32'))
        
        # 保存有效索引的映射
        self.valid_indices = valid_indices
        
        print(f"FAISS索引构建完成，维度: {dimension}")
        print(f"有效索引数量: {len(self.valid_indices)}")
        print(f"基于summary构建索引，用于粗粒度检索")
    
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
                device="cuda:0",  # 重排序器使用cuda:0
                cache_dir=cache_dir,
                use_quantization=use_quantization,
                quantization_type=quantization_type
            )
            print("Qwen reranker初始化完成 (cuda:0)")
        except Exception as e:
            print(f"Qwen reranker初始化失败: {e}")
            self.qwen_reranker = None
    
    def _init_llm_generator(self):
        """初始化LLM生成器"""
        print("正在初始化LLM生成器...")
        try:
            # 重用现有的LocalLLMGenerator
            from xlm.components.generator.local_llm_generator import LocalLLMGenerator
            
            # 使用配置中的参数
            model_name = None  # 让LocalLLMGenerator从config读取
            cache_dir = None   # 让LocalLLMGenerator从config读取
            device = "cuda:1"  # 生成器使用cuda:1
            use_quantization = None  # 让LocalLLMGenerator从config读取
            quantization_type = None  # 让LocalLLMGenerator从config读取
            
            if self.config:
                # 如果config中有generator配置，使用它
                if hasattr(self.config, 'generator'):
                    model_name = self.config.generator.model_name
                    cache_dir = self.config.generator.cache_dir
                    use_quantization = self.config.generator.use_quantization
                    quantization_type = self.config.generator.quantization_type
            
            self.llm_generator = LocalLLMGenerator(
                model_name=model_name,
                cache_dir=cache_dir,
                device=device,
                use_quantization=use_quantization,
                quantization_type=quantization_type
            )
            print("LLM生成器初始化完成")
        except Exception as e:
            print(f"LLM生成器初始化失败: {e}")
            self.llm_generator = None
    
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
        
        # 准备重排序的文档 - 使用doc_id到chunks的映射
        docs_for_rerank = []
        doc_to_rerank_mapping = []  # 记录重排序结果到原始doc_id的映射
        
        for doc_idx, faiss_score in candidate_results:
            # 从映射中找到对应的chunks
            if doc_idx in self.doc_to_chunks_mapping:
                chunks = self.doc_to_chunks_mapping[doc_idx]
                
                # 对每个chunk进行重排序
                for chunk_idx, chunk_content in enumerate(chunks):
                    if chunk_content.strip():  # 跳过空chunk
                        docs_for_rerank.append(chunk_content)
                        doc_to_rerank_mapping.append((doc_idx, chunk_idx))
            else:
                # 如果找不到映射，使用原始数据
                if doc_idx < len(self.data):
                    record = self.data[doc_idx]
                    if self.dataset_type == "chinese":
                        content = record.get('summary', '')
                    else:
                        content = record.get('context', '') or record.get('content', '')
                    docs_for_rerank.append(content)
                    doc_to_rerank_mapping.append((doc_idx, -1))
                else:
                    docs_for_rerank.append("")
                    doc_to_rerank_mapping.append((doc_idx, -1))
        
        if not docs_for_rerank:
            print("没有有效的文档用于重排序")
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
        
        try:
            # 使用合理的批处理大小
            reranked_docs = self.qwen_reranker.rerank(
                query=query,
                documents=docs_for_rerank,
                batch_size=4  # 增加到4，避免batch_size=1的问题
            )
            
            print(f"Qwen重排序结果: {len(reranked_docs)} 条记录")
            
            # 组合结果，按doc_id分组，取最高分
            doc_scores = {}
            for i, (doc_content, reranker_score) in enumerate(reranked_docs):
                if i < len(doc_to_rerank_mapping):
                    doc_idx, chunk_idx = doc_to_rerank_mapping[i]
                    
                    # 为每个doc_id记录最高分
                    if doc_idx not in doc_scores or reranker_score > doc_scores[doc_idx]['score']:
                        doc_scores[doc_idx] = {
                            'score': reranker_score,
                            'chunk_idx': chunk_idx
                        }
            
            # 组合FAISS分数和重排序分数
            results = []
            for doc_idx, faiss_score in candidate_results:
                if doc_idx in doc_scores:
                    reranker_score = doc_scores[doc_idx]['score']
                    results.append((doc_idx, faiss_score, reranker_score))
                else:
                    # 如果没有重排序分数，使用0
                    results.append((doc_idx, faiss_score, 0.0))
            
            # 按重排序分数降序排序
            results.sort(key=lambda x: x[2], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            print(f"Qwen重排序失败: {e}")
            print("回退到FAISS分数排序")
            # 回退到FAISS分数排序
            return [(idx, score, 0.0) for idx, score in candidate_results[:top_k]]
    
    def generate_answer(self, query: str, candidate_results: List[Tuple[int, float, float]], top_k_for_context: int = 5) -> str:
        """
        生成LLM答案 - 将重排序后的Top-K1个chunks拼接作为上下文
        
        Args:
            query: 查询文本
            candidate_results: 候选结果列表 [(doc_idx, faiss_score, reranker_score), ...]
            top_k_for_context: 用于生成上下文的候选数量
            
        Returns:
            生成的LLM答案
        """
        if not candidate_results:
            print("没有候选结果，无法生成答案")
            return ""
        
        print(f"开始生成LLM答案...")
        
        # 获取重排序后的Top-K1个文档的chunks
        top_chunks = []
        for doc_idx, faiss_score, reranker_score in candidate_results[:top_k_for_context]:
            if doc_idx in self.doc_to_chunks_mapping:
                chunks = self.doc_to_chunks_mapping[doc_idx]
                # 添加所有chunks
                top_chunks.extend(chunks)
            else:
                # 如果找不到映射，使用原始数据
                if doc_idx < len(self.data):
                    record = self.data[doc_idx]
                    if self.dataset_type == "chinese":
                        content = record.get('summary', '')
                    else:
                        content = record.get('context', '') or record.get('content', '')
                    top_chunks.append(content)
        
        # 拼接chunks作为上下文
        context = "\n\n".join([chunk for chunk in top_chunks if chunk.strip()])
        
        print(f"上下文长度: {len(context)} 个字符")
        print(f"使用了 {len(top_chunks)} 个chunks")
        
        # 使用LLM生成器生成答案
        if self.llm_generator:
            try:
                # 构建prompt
                prompt = f"基于以下上下文信息，回答用户的问题。\n\n上下文：{context}\n\n问题：{query}\n\n回答："
                
                # 使用LocalLLMGenerator的generate方法
                generated_texts = self.llm_generator.generate([prompt])
                answer = generated_texts[0] if generated_texts else "抱歉，无法生成答案。"
            except Exception as e:
                print(f"LLM生成失败: {e}")
                answer = f"抱歉，生成答案时出现错误：{str(e)}"
        else:
            # 回退到默认的简单模板回答
            answer = f"基于检索到的信息，我可以回答您的问题：{query}\n\n相关上下文：{context[:500]}...\n\n这是一个基于检索结果的回答。"
        
        print(f"LLM答案生成完成")
        return answer
    
    def search(self, 
               query: str,
               company_name: Optional[str] = None,
               stock_code: Optional[str] = None,
               report_date: Optional[str] = None,
               top_k: int = 20) -> Dict:
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
        
        # 4. LLM答案生成 - 将重排序后的Top-K1个chunks拼接作为上下文
        llm_answer = self.generate_answer(query, final_results, top_k_for_context=5)
        
        # 5. 格式化结果
        formatted_results = []
        for idx, faiss_score, combined_score in final_results:
            record = self.data[idx]
            
            # 根据数据集类型选择不同的字段
            if hasattr(record, 'content'):
                # DocumentWithMetadata格式
                result = {
                    'index': idx,
                    'faiss_score': faiss_score,
                    'combined_score': combined_score,
                    'content': record.content[:200] + '...' if len(record.content) > 200 else record.content,
                    'source': record.metadata.source if hasattr(record.metadata, 'source') else 'unknown',
                    'language': record.metadata.language if hasattr(record.metadata, 'language') else 'unknown'
                }
            elif isinstance(record, dict):
                # 字典格式（兼容旧格式）
                if self.dataset_type == "chinese":
                    result = {
                        'index': idx,
                        'faiss_score': faiss_score,
                        'combined_score': combined_score,
                        'company_name': record.get('company_name'),
                        'stock_code': record.get('stock_code'),
                        'report_date': record.get('report_date'),
                        'original_context': record.get('original_context', ''),  # 返回完整的original_context
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
            else:
                # 其他格式
                result = {
                    'index': idx,
                    'faiss_score': faiss_score,
                    'combined_score': combined_score,
                    'content': str(record)[:200] + '...' if len(str(record)) > 200 else str(record)
                }
            
            formatted_results.append(result)
        
        # 添加LLM生成的答案到结果中
        final_output = {
            'retrieved_documents': formatted_results,
            'llm_answer': llm_answer,
            'query': query,
            'total_documents': len(formatted_results)
        }
        
        print(f"检索完成，返回 {len(formatted_results)} 条结果")
        print(f"LLM答案生成完成")
        return final_output
    
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
    
    for i, result in enumerate(results1['retrieved_documents']):
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
    
    for i, result in enumerate(results2['retrieved_documents']):
        print(f"\n结果 {i+1}:")
        print(f"  公司: {result['company_name']}")
        print(f"  股票代码: {result['stock_code']}")
        print(f"  摘要: {result['summary']}")
        print(f"  相似度分数: {result['combined_score']:.4f}")

if __name__ == '__main__':
    main() 