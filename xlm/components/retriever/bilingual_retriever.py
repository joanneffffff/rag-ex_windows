from typing import List, Dict, Tuple, Union
import numpy as np
import torch
import faiss
import os
import pickle
import hashlib
from sentence_transformers.util import semantic_search
from langdetect import detect
from tqdm import tqdm

from xlm.components.encoder.finbert import FinbertEncoder
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class BilingualRetriever(Retriever):
    def __init__(
        self,
        encoder_en: FinbertEncoder,
        encoder_ch: FinbertEncoder,
        max_context_length: int = 100,
        num_threads: int = 4,
        corpus_documents_en: List[DocumentWithMetadata] = None,
        corpus_documents_ch: List[DocumentWithMetadata] = None,
        use_faiss: bool = False,
        batch_size: int = 32,
        use_gpu: bool = False,
        cache_dir: str = "cache",
        use_existing_embedding_index: bool = False
    ):
        self.encoder_en = encoder_en
        self.encoder_ch = encoder_ch
        self.max_context_length = max_context_length
        self.__num_threads = num_threads
        self.use_faiss = use_faiss
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.cache_dir = cache_dir
        self.use_existing_embedding_index = use_existing_embedding_index

        self.corpus_documents_en = corpus_documents_en or []
        self.corpus_embeddings_en = None
        self.index_en = None

        self.corpus_documents_ch = corpus_documents_ch or []
        self.corpus_embeddings_ch = None
        self.index_ch = None

        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)

        # 只在use_existing_embedding_index为True时尝试加载缓存，否则强制重新编码
        if self.use_existing_embedding_index:
            loaded = self._load_cached_embeddings()
            if loaded:
                print("Loaded cached embeddings successfully.")
            else:
                raise RuntimeError("use_existing_embedding_index=True但未找到有效缓存，请先生成embedding缓存！")
        else:
            self._compute_embeddings()

        if corpus_documents_en is not None:
            for i, doc in enumerate(corpus_documents_en):
                assert hasattr(doc, 'content') and isinstance(doc.content, str), f"corpus_documents_en污染: idx={i}, type={type(doc)}, 内容: {doc}"
        if corpus_documents_ch is not None:
            for i, doc in enumerate(corpus_documents_ch):
                assert hasattr(doc, 'content') and isinstance(doc.content, str), f"corpus_documents_ch污染: idx={i}, type={type(doc)}, 内容: {doc}"

    def _get_cache_key(self, documents: List[DocumentWithMetadata], encoder_name: str) -> str:
        """生成缓存键，基于文档内容和编码器名称"""
        # 创建文档内容的哈希
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.content.encode('utf-8'))
        
        # 只使用编码器名称的最后部分，避免路径问题
        encoder_basename = os.path.basename(encoder_name)
        
        # 结合编码器名称和文档数量
        cache_key = f"{encoder_basename}_{len(documents)}_{content_hash.hexdigest()[:16]}"
        return cache_key

    def _get_cache_path(self, cache_key: str, suffix: str) -> str:
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{cache_key}.{suffix}")

    def _load_cached_embeddings(self) -> bool:
        """尝试加载缓存的嵌入向量"""
        try:
            loaded_any = False
            
            # 检查英文文档缓存
            if self.corpus_documents_en:
                cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                index_path_en = self._get_cache_path(cache_key_en, "faiss")
                
                if os.path.exists(embeddings_path_en):
                    self.corpus_embeddings_en = np.load(embeddings_path_en)
                    loaded_any = True
                    
                    if self.use_faiss and os.path.exists(index_path_en):
                        self.index_en = faiss.read_index(index_path_en)
                    elif self.use_faiss:
                        self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
                        if self.corpus_embeddings_en is not None:
                            self._add_to_faiss(self.index_en, self.corpus_embeddings_en)
            else:
                # 英文文档为空，检查是否有任何英文缓存文件
                cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npy') and 'finetuned_finbert_tatqa' in f]
                if cache_files:
                    # 使用第一个找到的缓存文件
                    cache_file = cache_files[0]
                    embeddings_path_en = os.path.join(self.cache_dir, cache_file)
                    self.corpus_embeddings_en = np.load(embeddings_path_en)
                    loaded_any = True
                    
                    # 尝试加载对应的FAISS索引
                    index_file = cache_file.replace('.npy', '.faiss')
                    index_path_en = os.path.join(self.cache_dir, index_file)
                    if os.path.exists(index_path_en):
                        self.index_en = faiss.read_index(index_path_en)

            # 检查中文文档缓存
            if self.corpus_documents_ch:
                cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                
                if os.path.exists(embeddings_path_ch):
                    self.corpus_embeddings_ch = np.load(embeddings_path_ch)
                    loaded_any = True
                    
                    if self.use_faiss and os.path.exists(index_path_ch):
                        self.index_ch = faiss.read_index(index_path_ch)
                    elif self.use_faiss:
                        self.index_ch = self._init_faiss(self.encoder_ch, len(self.corpus_documents_ch))
                        if self.corpus_embeddings_ch is not None:
                            self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
            else:
                # 中文文档为空，检查是否有任何中文缓存文件
                cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npy') and 'finetuned_alphafin' in f]
                if cache_files:
                    # 使用第一个找到的缓存文件
                    cache_file = cache_files[0]
                    embeddings_path_ch = os.path.join(self.cache_dir, cache_file)
                    self.corpus_embeddings_ch = np.load(embeddings_path_ch)
                    loaded_any = True
                    
                    # 尝试加载对应的FAISS索引
                    index_file = cache_file.replace('.npy', '.faiss')
                    index_path_ch = os.path.join(self.cache_dir, index_file)
                    if os.path.exists(index_path_ch):
                        self.index_ch = faiss.read_index(index_path_ch)

            return loaded_any
        except Exception as e:
            return False

    def _save_cached_embeddings(self):
        """保存嵌入向量到缓存"""
        try:
            # 保存英文文档嵌入向量
            if self.corpus_documents_en and self.corpus_embeddings_en is not None:
                cache_key_en = self._get_cache_key(self.corpus_documents_en, str(self.encoder_en.model_name))
                embeddings_path_en = self._get_cache_path(cache_key_en, "npy")
                index_path_en = self._get_cache_path(cache_key_en, "faiss")
                # 确保目录存在
                os.makedirs(os.path.dirname(embeddings_path_en), exist_ok=True)
                os.makedirs(os.path.dirname(index_path_en), exist_ok=True)
                np.save(embeddings_path_en, self.corpus_embeddings_en)
                if self.use_faiss and self.index_en:
                    faiss.write_index(self.index_en, index_path_en)

            # 保存中文文档嵌入向量
            if self.corpus_documents_ch and self.corpus_embeddings_ch is not None:
                cache_key_ch = self._get_cache_key(self.corpus_documents_ch, str(self.encoder_ch.model_name))
                embeddings_path_ch = self._get_cache_path(cache_key_ch, "npy")
                index_path_ch = self._get_cache_path(cache_key_ch, "faiss")
                # 确保目录存在
                os.makedirs(os.path.dirname(embeddings_path_ch), exist_ok=True)
                os.makedirs(os.path.dirname(index_path_ch), exist_ok=True)
                np.save(embeddings_path_ch, self.corpus_embeddings_ch)
                if self.use_faiss and self.index_ch:
                    faiss.write_index(self.index_ch, index_path_ch)
        except Exception as e:
            pass

    def _compute_embeddings(self):
        """计算嵌入向量"""
        if self.use_faiss:
            if self.corpus_documents_en:
                self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
            if self.corpus_documents_ch:
                self.index_ch = self._init_faiss(self.encoder_ch, len(self.corpus_documents_ch))

        if self.corpus_documents_en:
            self.corpus_embeddings_en = self._batch_encode_corpus(self.corpus_documents_en, self.encoder_en)
            if self.use_faiss and self.corpus_embeddings_en is not None:
                self._add_to_faiss(self.index_en, self.corpus_embeddings_en)

        if self.corpus_documents_ch:
            self.corpus_embeddings_ch = self._batch_encode_corpus(self.corpus_documents_ch, self.encoder_ch)
            if self.use_faiss and self.corpus_embeddings_ch is not None:
                self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
        
        # 保存到缓存
        self._save_cached_embeddings()
        
        pass

    def _init_faiss(self, encoder, corpus_size):
        """Initialize FAISS index"""
        dimension = encoder.get_embedding_dimension()
        
        # Create FAISS index
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.IndexFlatL2(dimension)
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            return gpu_index
        else:
            if corpus_size < 1000:
                return faiss.IndexFlatL2(dimension)
            else:
                nlist = min(max(int(corpus_size / 100), 4), 1024)
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                return index

    def _batch_encode_corpus(self, documents: List[DocumentWithMetadata], encoder: FinbertEncoder) -> np.ndarray:
        """Encode corpus documents in batches with a progress bar."""
        batch_texts = [doc.content for doc in documents]
        return encoder.encode(texts=batch_texts, batch_size=self.batch_size, show_progress_bar=True)
    
    def _add_to_faiss(self, index, embeddings: np.ndarray):
        """Add embeddings to FAISS index in batches"""
        if embeddings is None or embeddings.shape[0] == 0:
            return
        
        embeddings_np = embeddings.astype('float32')
        if not index.is_trained:
            index.train(embeddings_np)
        index.add(embeddings_np)

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        return_scores: bool = False,
        language: str = None,
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]:
        # 检查self.corpus_documents_en/ch类型
        if hasattr(self, 'corpus_documents_en') and self.corpus_documents_en is not None:
            for i, doc in enumerate(self.corpus_documents_en):
                assert hasattr(doc, 'content') and isinstance(doc.content, str), f"self.corpus_documents_en污染: idx={i}, type={type(doc)}, 内容: {doc}"
        if hasattr(self, 'corpus_documents_ch') and self.corpus_documents_ch is not None:
            for i, doc in enumerate(self.corpus_documents_ch):
                assert hasattr(doc, 'content') and isinstance(doc.content, str), f"self.corpus_documents_ch污染: idx={i}, type={type(doc)}, 内容: {doc}"
        if language is None:
            lang = detect(text)
            language = 'zh' if lang.startswith('zh') else 'en'
        if language == 'zh':
            encoder = self.encoder_ch
            corpus_embeddings = self.corpus_embeddings_ch
            corpus_documents = self.corpus_documents_ch
            index = self.index_ch
        else:
            encoder = self.encoder_en
            corpus_embeddings = self.corpus_embeddings_en
            corpus_documents = self.corpus_documents_en
            index = self.index_en
        if corpus_embeddings is None or corpus_embeddings.shape[0] == 0:
            if return_scores:
                return [], []
            else:
                return []
        query_embeddings = encoder.encode([text])
        if self.use_faiss and index:
            distances, indices = index.search(query_embeddings.astype('float32'), top_k)
            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx != -1:
                    results.append({'corpus_id': idx, 'score': 1 - score / 2})
        else:
            hits = semantic_search(
                torch.tensor(query_embeddings),
                torch.tensor(corpus_embeddings),
                top_k=top_k
            )
            results = hits[0]
        doc_indices = [hit['corpus_id'] for hit in results]
        scores = [hit['score'] for hit in results]
        raw_documents = [corpus_documents[i] for i in doc_indices]
        
        # 确保返回的是DocumentWithMetadata对象，统一使用content字段
        documents = []
        for doc in raw_documents:
            if isinstance(doc, dict):
                content = doc.get('content', doc.get('context', ''))
                if not isinstance(content, str):
                    # 如果content不是字符串，尝试取context字段或转为字符串
                    content = content.get('context', '') if isinstance(content, dict) and 'context' in content else str(content)
                metadata = DocumentMetadata(
                    source=doc.get('source', 'unknown'),
                    created_at=doc.get('created_at', ''),
                    author=doc.get('author', ''),
                    language=language or 'unknown'
                )
                documents.append(DocumentWithMetadata(content=content, metadata=metadata))
            else:
                documents.append(doc)
        
        if return_scores:
            return documents, scores
        else:
            return documents 