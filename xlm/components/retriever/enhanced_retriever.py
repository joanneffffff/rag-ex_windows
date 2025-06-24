"""
Enhanced retriever with dual embedding spaces and FAISS indexing for Chinese/English
"""

from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import torch
import faiss
from sentence_transformers.util import semantic_search
from tqdm import tqdm
from langdetect import detect, LangDetectException

from xlm.components.encoder.encoder import Encoder
from xlm.components.retriever.retriever import Retriever
from xlm.components.retriever.reranker import QwenReranker
from xlm.dto.dto import DocumentWithMetadata
from config.parameters import Config

class EnhancedRetriever(Retriever):
    def __init__(
        self,
        config: Config,
        chinese_documents: List[DocumentWithMetadata] = None,
        english_documents: List[DocumentWithMetadata] = None
    ):
        """
        初始化增强检索器 - 双空间双索引版本
        
        Args:
            config: 配置对象
            chinese_documents: 中文语料库文档
            english_documents: 英文语料库文档
        """
        self.config = config
        self.chinese_documents = chinese_documents or []
        self.english_documents = english_documents or []
        
        # 初始化编码器
        self._init_encoders()
        
        # 初始化重排序器
        self.reranker = None
        if config.reranker.enabled:
            self._init_reranker()
        
        # 初始化双FAISS索引
        self.chinese_index = None
        self.english_index = None
        if config.retriever.use_faiss:
            self._init_faiss_indices()
        
        # 编码双语料库
        self._encode_dual_corpus()
    
    def _init_encoders(self):
        """初始化中英文编码器"""
        print("初始化双编码器...")
        
        # 中文编码器
        chinese_model_path = self.config.encoder.chinese_model_path
        print(f"加载中文编码器: {chinese_model_path}")
        self.chinese_encoder = Encoder(
            model_name=chinese_model_path,
            device=self.config.encoder.device,
            cache_dir=self.config.encoder.cache_dir
        )
        
        # 英文编码器
        english_model_path = self.config.encoder.english_model_path
        print(f"加载英文编码器: {english_model_path}")
        self.english_encoder = Encoder(
            model_name=english_model_path,
            device=self.config.encoder.device,
            cache_dir=self.config.encoder.cache_dir
        )
    
    def _init_reranker(self):
        """初始化重排序器"""
        print("初始化重排序器...")
        self.reranker = QwenReranker(
            model_name=self.config.reranker.model_name,
            device=self.config.reranker.device,
            cache_dir=self.config.reranker.cache_dir,
            use_quantization=self.config.reranker.use_quantization,
            quantization_type=self.config.reranker.quantization_type
        )
    
    def _init_faiss_indices(self):
        """初始化双FAISS索引"""
        print("初始化双FAISS索引...")
        
        # 获取中文嵌入维度
        sample_text_zh = "样本文本用于维度检测"
        sample_embedding_zh = self.chinese_encoder.encode([sample_text_zh])[0]
        dimension_zh = len(sample_embedding_zh)
        
        # 获取英文嵌入维度
        sample_text_en = "Sample text for dimension detection"
        sample_embedding_en = self.english_encoder.encode([sample_text_en])[0]
        dimension_en = len(sample_embedding_en)
        
        print(f"中文嵌入维度: {dimension_zh}")
        print(f"英文嵌入维度: {dimension_en}")
        
        # 创建中文FAISS索引
        if self.chinese_documents:
            self.chinese_index = self._create_faiss_index(dimension_zh, len(self.chinese_documents))
        
        # 创建英文FAISS索引
        if self.english_documents:
            self.english_index = self._create_faiss_index(dimension_en, len(self.english_documents))
    
    def _create_faiss_index(self, dimension: int, corpus_size: int):
        """创建FAISS索引"""
        if self.config.retriever.use_gpu and faiss.get_num_gpus() > 0:
            # 使用GPU索引
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            return faiss.index_cpu_to_gpu(res, 0, index)
        else:
            # 根据语料库大小选择索引类型
            if corpus_size < 1000:
                # 小数据集使用简单平面索引
                return faiss.IndexFlatL2(dimension)
            else:
                # 大数据集使用IVF索引
                nlist = min(max(int(corpus_size / 100), 4), 1024)
                quantizer = faiss.IndexFlatL2(dimension)
                return faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    def _encode_dual_corpus(self):
        """编码双语料库"""
        # 编码中文语料库
        if self.chinese_documents:
            print(f"编码 {len(self.chinese_documents)} 个中文文档...")
            self.chinese_embeddings = self._encode_corpus_batch(
                self.chinese_documents, 
                self.chinese_encoder, 
                "中文语料库"
            )
            
            # 添加到中文FAISS索引
            if self.config.retriever.use_faiss and self.chinese_index:
                print("将中文嵌入添加到FAISS索引...")
                self._add_to_faiss(self.chinese_index, self.chinese_embeddings)
                print("中文FAISS索引构建完成")
        
        # 编码英文语料库
        if self.english_documents:
            print(f"编码 {len(self.english_documents)} 个英文文档...")
            self.english_embeddings = self._encode_corpus_batch(
                self.english_documents, 
                self.english_encoder, 
                "英文语料库"
            )
            
            # 添加到英文FAISS索引
            if self.config.retriever.use_faiss and self.english_index:
                print("将英文嵌入添加到FAISS索引...")
                self._add_to_faiss(self.english_index, self.english_embeddings)
                print("英文FAISS索引构建完成")
    
    def _encode_corpus_batch(self, documents: List[DocumentWithMetadata], encoder: Encoder, desc: str) -> np.ndarray:
        """批量编码语料库"""
        all_embeddings = []
        batch_size = self.config.encoder.batch_size
        
        for i in tqdm(range(0, len(documents), batch_size), desc=desc):
            batch = documents[i:i + batch_size]
            batch_texts = [doc.content for doc in batch]
            batch_embeddings = encoder.encode(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def _add_to_faiss(self, index, embeddings: np.ndarray):
        """添加嵌入到FAISS索引"""
        embeddings_np = embeddings.astype('float32')
        if not index.is_trained:
            index.train(embeddings_np)
        index.add(embeddings_np)
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        try:
            lang = detect(text)
            if lang.startswith('zh'):
                return 'chinese'
            else:
                return 'english'
        except LangDetectException:
            # 默认返回英文
            return 'english'
    
    def retrieve(
        self,
        text: str,
        top_k: int = None,
        return_scores: bool = False,
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]:
        """
        检索文档 - 根据查询语言自动选择编码器和索引
        
        Args:
            text: 查询文本
            top_k: 返回的文档数量
            return_scores: 是否返回分数
            
        Returns:
            检索到的文档列表，可选包含分数
        """
        if top_k is None:
            top_k = self.config.retriever.rerank_top_k
        
        # 检测查询语言
        query_language = self._detect_language(text)
        print(f"检测到查询语言: {query_language}")
        
        # 根据语言选择对应的编码器、索引和文档
        if query_language == 'chinese':
            encoder = self.chinese_encoder
            index = self.chinese_index
            embeddings = getattr(self, 'chinese_embeddings', None)
            documents = self.chinese_documents
        else:
            encoder = self.english_encoder
            index = self.english_index
            embeddings = getattr(self, 'english_embeddings', None)
            documents = self.english_documents
        
        # 检查是否有对应的语料库
        if not documents or embeddings is None:
            print(f"警告: 没有找到{query_language}语料库")
            if return_scores:
                return [], []
            else:
                return []
        
        # 1. 初始检索
        initial_results = self._initial_retrieval(
            text, 
            encoder, 
            embeddings, 
            index, 
            self.config.retriever.retrieval_top_k
        )
        
        if not initial_results:
            if return_scores:
                return [], []
            else:
                return []
        
        # 2. 重排序（如果启用）
        if self.reranker and len(initial_results) > 1:
            final_results = self._rerank_documents(text, initial_results, top_k)
        else:
            final_results = initial_results[:top_k]
        
        # 3. 提取文档和分数
        result_documents = [documents[result['corpus_id']] for result in final_results]
        scores = [result['score'] for result in final_results]
        
        if return_scores:
            return result_documents, scores
        else:
            return result_documents
    
    def _initial_retrieval(self, text: str, encoder: Encoder, embeddings: np.ndarray, index, top_k: int) -> List[Dict]:
        """初始检索"""
        # 编码查询
        query_embeddings = encoder.encode([text])
        
        if self.config.retriever.use_faiss and index:
            # 使用FAISS检索
            query_np = np.array(query_embeddings).astype('float32')
            distances, indices = index.search(query_np, top_k)
            
            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < embeddings.shape[0]:
                    results.append({
                        'corpus_id': int(idx),
                        'score': float(1.0 / (1.0 + score))
                    })
            return results
        else:
            # 使用语义搜索
            hits = semantic_search(
                torch.tensor(query_embeddings),
                torch.tensor(embeddings),
                top_k=top_k
            )
            return hits[0]
    
    def _rerank_documents(self, text: str, initial_results: List[Dict], top_k: int) -> List[Dict]:
        """重排序文档"""
        # 根据查询语言选择对应的文档
        query_language = self._detect_language(text)
        if query_language == 'chinese':
            documents = self.chinese_documents
        else:
            documents = self.english_documents
        
        # 提取文档文本
        doc_texts = []
        for result in initial_results:
            doc_id = result['corpus_id']
            if doc_id < len(documents):
                doc_text = documents[doc_id].content
                doc_texts.append(doc_text)
        
        if not doc_texts:
            return initial_results[:top_k]
        
        # 使用重排序器
        reranked_results = self.reranker.rerank(
            query=text,
            documents=doc_texts,
            batch_size=self.config.reranker.batch_size
        )
        
        # 将重排序结果映射回原始索引
        final_results = []
        for i, (doc_text, reranker_score) in enumerate(reranked_results):
            if i < len(initial_results):
                original_result = initial_results[i]
                # 结合原始分数和重排序分数
                combined_score = (original_result['score'] + reranker_score) / 2
                final_results.append({
                    'corpus_id': original_result['corpus_id'],
                    'score': combined_score,
                    'original_score': original_result['score'],
                    'reranker_score': reranker_score
                })
        
        return final_results[:top_k]
    
    def update_corpus(self, chinese_documents: List[DocumentWithMetadata] = None, english_documents: List[DocumentWithMetadata] = None):
        """更新语料库"""
        if chinese_documents is not None:
            self.chinese_documents = chinese_documents
        if english_documents is not None:
            self.english_documents = english_documents
        
        # 重新编码和构建索引
        self._init_faiss_indices()
        self._encode_dual_corpus()
    
    def get_corpus_size(self) -> Dict[str, int]:
        """获取语料库大小"""
        return {
            'chinese': len(self.chinese_documents),
            'english': len(self.english_documents)
        } 