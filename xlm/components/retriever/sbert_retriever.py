from typing import List, Dict, Tuple, Union
import numpy as np
import torch
import faiss
from sentence_transformers.util import semantic_search

from xlm.components.encoder.encoder import Encoder
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata


class SBERTRetriever(Retriever):
    def __init__(
        self,
        encoder: Encoder,
        max_context_length: int = 100,
        num_threads: int = 4,
        corpus_embeddings: list = None,
        corpus_documents: list = None,
        use_faiss: bool = False,
        batch_size: int = 32,
        use_gpu: bool = False
    ):
        self.encoder = encoder
        self.max_context_length = max_context_length
        self.__num_threads = num_threads
        self.corpus_documents = corpus_documents or []
        self.use_faiss = use_faiss
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.corpus_embeddings = np.array([])
        
        # Initialize FAISS if needed
        self.index = None
        if use_faiss:
            self._init_faiss()
        
        # Process initial corpus if provided
        if corpus_documents:
            if corpus_embeddings is not None:
                self.corpus_embeddings = np.array(corpus_embeddings)
            else:
                self.corpus_embeddings = self._batch_encode_corpus(documents=corpus_documents)
            
            # Add embeddings to FAISS if using it
            if self.use_faiss and self.corpus_embeddings is not None and hasattr(self.corpus_embeddings, 'shape') and self.corpus_embeddings.shape[0] > 0:
                self._add_to_faiss(self.corpus_embeddings)
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        # Get embedding dimension from encoder
        sample_text = "Sample text for dimension detection"
        sample_embedding = self.encoder.encode([sample_text])[0]
        dimension = len(sample_embedding)
        
        # Create FAISS index
        if self.use_gpu and faiss.get_num_gpus() > 0:
            # Use GPU index if available
            self.index = faiss.IndexFlatL2(dimension)
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(),
                0,
                self.index
            )
        else:
            # Determine index type based on corpus size
            corpus_size = len(self.corpus_documents)
            if corpus_size < 100:
                # For small datasets, use simple flat index
                self.index = faiss.IndexFlatL2(dimension)
            else:
                # For larger datasets, use IVF index
                nlist = min(max(int(corpus_size / 10), 4), corpus_size)  # Adaptive number of clusters
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
                # Train index if needed
                if not self.index.is_trained and self.corpus_embeddings is not None and self.corpus_embeddings.shape[0] > 0:
                    train_data = self.corpus_embeddings.astype('float32')
                    self.index.train(train_data)
    
    def _batch_encode_corpus(self, documents: List[DocumentWithMetadata]) -> np.ndarray:
        """Encode corpus documents in batches"""
        all_embeddings = []
        batch_size = 32  # 可以根据需要调整
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [doc.content for doc in batch]
            # 使用encode方法而不是encode_batch
            batch_embeddings = self.encoder.encode(texts=batch_texts)
            all_embeddings.extend(batch_embeddings)
        return np.array(all_embeddings)
    
    def _add_to_faiss(self, embeddings: List[List[float]]):
        """Add embeddings to FAISS index in batches"""
        if not embeddings:
            return
        
        # Process in batches
        for i in range(0, len(embeddings), self.batch_size):
            batch = embeddings[i:i+self.batch_size]
            batch_np = np.array(batch).astype('float32')
            
            if not self.index.is_trained:
                self.index.train(batch_np)
            self.index.add(batch_np)
    
    def encode_corpus(self, documents: List[DocumentWithMetadata]) -> List[List[float]]:
        """Encode corpus documents"""
        texts = [str(doc.content) for doc in documents]
        return self.encoder.encode(texts=texts)

    def encode_queries(self, text: str):
        # 兼容原始Encoder和MultiModalEncoder
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode([text])
        elif hasattr(self.encoder, "encode_batch"):
            result = self.encoder.encode_batch(texts=[text])
            return result['text']
        else:
            raise AttributeError("Encoder does not support encode or encode_batch methods.")

    def update_corpus(self, documents: list, embeddings: list = None):
        """Update corpus documents and embeddings"""
        self.corpus_documents = documents
        if embeddings is not None:
            self.corpus_embeddings = np.array(embeddings)
        else:
            self.corpus_embeddings = self._batch_encode_corpus(documents)
            
        if self.use_faiss:
            self._init_faiss()  # Reinitialize FAISS with new corpus
            self._add_to_faiss(self.corpus_embeddings)

    def search(
        self,
        query_embeddings: List[List[float]],
        corpus_embeddings: List[List[float]],
        top_k: int = 3,
    ) -> List[Dict]:
        """Search for similar documents"""
        if self.use_faiss and self.index:
            return self._faiss_search(query_embeddings, top_k)
        else:
            return self._sbert_search(query_embeddings, corpus_embeddings, top_k)
    
    def _faiss_search(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 3
    ) -> List[Dict]:
        """Search using FAISS"""
        query_np = np.array(query_embeddings).astype('float32')
        distances, indices = self.index.search(query_np, top_k)
        
        results = []
        for query_idx, (query_distances, query_indices) in enumerate(zip(distances, indices)):
            query_results = []
            for score, idx in zip(query_distances, query_indices):
                if idx < len(self.corpus_documents):
                    query_results.append({
                        'corpus_id': int(idx),
                        'score': float(1.0 / (1.0 + score))
                    })
            results.append(query_results)
        
        return results
    
    def _sbert_search(
        self,
        query_embeddings: List[List[float]],
        corpus_embeddings: List[List[float]],
        top_k: int = 3
    ) -> List[Dict]:
        """Search using sentence-transformers"""
        query_embeddings = torch.tensor(query_embeddings)
        corpus_embeddings = torch.tensor(corpus_embeddings)
        
        return semantic_search(
            query_embeddings,
            corpus_embeddings,
            top_k=top_k
        )

    def retrieve_documents_with_scores(
        self,
        text: str,
        top_k: int = 5,
    ) -> list:
        query_embeddings = self.encode_queries(text=text)
        
        # Ensure we have corpus embeddings
        if self.corpus_embeddings is None or not hasattr(self.corpus_embeddings, 'shape') or self.corpus_embeddings.shape[0] == 0:
            return [], []
        
        results = self.search(
            query_embeddings=query_embeddings,
            corpus_embeddings=self.corpus_embeddings,
            top_k=top_k,
        )
        return results

    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        return_scores: bool = False,
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]:
        result = self.retrieve_documents_with_scores(text=text, top_k=top_k)
        idxes = []
        scores = []
        for item in result[0]:
            idxes.append(item["corpus_id"])
            scores.append(item["score"])
        documents = [self.corpus_documents[idx] for idx in idxes]

        if return_scores:
            return documents, scores
        else:
            return documents
