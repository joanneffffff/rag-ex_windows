from typing import List, Dict, Tuple, Union
import numpy as np
import torch
import faiss
from sentence_transformers.util import semantic_search
from langdetect import detect
from tqdm import tqdm

from xlm.components.encoder.finbert import FinbertEncoder
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import DocumentWithMetadata

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
        use_gpu: bool = False
    ):
        self.encoder_en = encoder_en
        self.encoder_ch = encoder_ch
        self.max_context_length = max_context_length
        self.__num_threads = num_threads
        self.use_faiss = use_faiss
        self.batch_size = batch_size
        self.use_gpu = use_gpu

        self.corpus_documents_en = corpus_documents_en or []
        self.corpus_embeddings_en = None
        self.index_en = None

        self.corpus_documents_ch = corpus_documents_ch or []
        self.corpus_embeddings_ch = None
        self.index_ch = None

        if use_faiss:
            if self.corpus_documents_en:
                print("Initializing FAISS index for English documents...")
                self.index_en = self._init_faiss(self.encoder_en, len(self.corpus_documents_en))
            if self.corpus_documents_ch:
                print("Initializing FAISS index for Chinese documents...")
                self.index_ch = self._init_faiss(self.encoder_ch, len(self.corpus_documents_ch))

        if self.corpus_documents_en:
            print(f"Start encoding {len(self.corpus_documents_en)} English documents...")
            self.corpus_embeddings_en = self._batch_encode_corpus(self.corpus_documents_en, self.encoder_en)
            if self.use_faiss and self.corpus_embeddings_en is not None:
                print("Adding English document embeddings to FAISS index...")
                self._add_to_faiss(self.index_en, self.corpus_embeddings_en)
                print("English documents indexed.")

        if self.corpus_documents_ch:
            print(f"Start encoding {len(self.corpus_documents_ch)} Chinese documents...")
            self.corpus_embeddings_ch = self._batch_encode_corpus(self.corpus_documents_ch, self.encoder_ch)
            if self.use_faiss and self.corpus_embeddings_ch is not None:
                print("Adding Chinese document embeddings to FAISS index...")
                self._add_to_faiss(self.index_ch, self.corpus_embeddings_ch)
                print("Chinese documents indexed.")
        
        print("Retriever initialization complete.")

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
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]:
        lang = detect(text)
        
        if lang == 'zh-cn' or lang == 'zh-tw':
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
        documents = [corpus_documents[i] for i in doc_indices]

        if return_scores:
            return documents, scores
        else:
            return documents 