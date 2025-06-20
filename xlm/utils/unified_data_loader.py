"""
Unified data loader for financial data
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import json
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datasets import load_dataset
from xlm.dto.dto import DocumentWithMetadata

class UnifiedDataLoader:
    def __init__(
        self,
        data_dir: str = "data",
        # cache_dir: str = "D:/AI/huggingface",
        cache_dir: str = "M:/huggingface",
        use_faiss: bool = True,
        batch_size: int = 32,
        max_samples: int = 1000,  # 限制每个数据源的最大样本数
        use_cached_index: bool = True  # 是否使用缓存的索引
    ):
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.use_faiss = use_faiss
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.use_cached_index = use_cached_index
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        print("准备加载 SentenceTransformer ...")
        # 推荐多语言模型 paraphrase-multilingual-MiniLM-L12-v2
        self.encoder = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=cache_dir
        )
        print("SentenceTransformer 加载完成")
        
        print("准备加载数据 ...")
        self.documents: List[DocumentWithMetadata] = []
        self._load_data()
        print("数据加载完成")
        
        # Initialize retriever attributes
        self.corpus_embeddings = None
        self.index = None
        
        print("准备构建/加载索引 ...")
        self.build_unified_index(save_dir="data/processed")
        print("索引构建/加载完成")
    
    def _load_data(self):
        """Load and process all data sources"""
        # Load TatQA data
        tatqa_docs = self._load_tatqa_data()
        if len(tatqa_docs) > self.max_samples:
            print(f"Sampling {self.max_samples} documents from TatQA dataset...")
            tatqa_docs = tatqa_docs[:self.max_samples]
        self.documents.extend(tatqa_docs)
        print(f"Loaded {len(tatqa_docs)} TatQA documents\n")
        
        # Load AlphaFin data
        try:
            print("Loading AlphaFin data from local file...")
            alphafin_docs = self._load_local_alphafin_data()
            if not alphafin_docs:
                print("No local AlphaFin data found, trying HuggingFace...")
                alphafin_docs = self._load_hf_alphafin_data()
            
            if len(alphafin_docs) > self.max_samples:
                print(f"Sampling {self.max_samples} documents from AlphaFin dataset...")
                alphafin_docs = alphafin_docs[:self.max_samples]
        except Exception as e:
            self.logger.error(f"Error loading AlphaFin data: {str(e)}")
            alphafin_docs = []
        
        self.documents.extend(alphafin_docs)
        print(f"Loaded {len(alphafin_docs)} AlphaFin documents\n")
    
    def _load_tatqa_data(self) -> List[DocumentWithMetadata]:
        """Load and process TatQA data"""
        tatqa_docs = []
        tatqa_path = Path(self.data_dir) / "tatqa_dataset_raw/tatqa_dataset_train.json"
        
        if not tatqa_path.exists():
            self.logger.warning(f"TatQA data file not found at {tatqa_path}")
            return tatqa_docs
        
        print(f"Processing TatQA data from {tatqa_path}")
        
        try:
            with open(tatqa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("TatQA data should be a list")
                
            for item in tqdm(data, desc="Processing TatQA items"):
                if not isinstance(item, dict):
                    continue
                    
                # Process tables with context
                if 'table' in item and isinstance(item['table'], dict):
                    table_text = self._process_table(item['table'])
                    paragraphs = item.get('paragraphs', [])
                    qa_pairs = item.get('qa', {})
                    
                    # 构建基础上下文
                    base_context = []
                    if table_text:
                        base_context.append(f"Table:\n{table_text}")
                    if paragraphs:
                        base_context.append("Context:\n" + "\n".join(str(p) for p in paragraphs))
                    
                    # 合并上下文
                    context = "\n\n".join(base_context)
                    
                    # 处理问答对
                    if qa_pairs:
                        question = str(qa_pairs.get('question', ''))
                        answer = str(qa_pairs.get('answer', ''))
                        derivation = qa_pairs.get('derivation', [])
                        scale = str(qa_pairs.get('scale', ''))
                        answer_type = str(qa_pairs.get('answer_type', ''))
                        
                        # 构建答案解释
                        answer_explanation = []
                        if scale:
                            answer_explanation.append(f"Scale: {scale}")
                        if answer_type:
                            answer_explanation.append(f"Answer Type: {answer_type}")
                        if derivation:
                            answer_explanation.append(f"Derivation: {' -> '.join(str(d) for d in derivation)}")
                        
                        # 构建完整的问答文本
                        qa_text = [
                            "Question & Answer:",
                            f"Q: {question}",
                            f"A: {answer}"
                        ]
                        if answer_explanation:
                            qa_text.append("Details:")
                            qa_text.extend(answer_explanation)
                        
                        # 创建问答文档
                        qa_content = f"{context}\n\n{chr(10).join(qa_text)}"
                        tatqa_docs.append(DocumentWithMetadata(
                            content=qa_content,
                            metadata={
                                "source": "tatqa_qa",
                                "id": str(item.get('id', '')),
                                "question": question,
                                "answer": answer,
                                "scale": scale,
                                "answer_type": answer_type
                            }
                        ))
                    
                    # 存储独立的表格和上下文
                    if context.strip():
                        tatqa_docs.append(DocumentWithMetadata(
                            content=context,
                            metadata={
                                "source": "tatqa_table",
                                "id": str(item.get('id', ''))
                            }
                        ))
                
                # 处理独立段落
                if 'paragraphs' in item and isinstance(item['paragraphs'], list):
                    for para in item['paragraphs']:
                        if para and str(para).strip():
                            # 统计词数和句数
                            word_count = len(str(para).split())
                            sentence_count = str(para).count('.') + str(para).count('。')
                            tatqa_docs.append(DocumentWithMetadata(
                                content=str(para),
                                metadata={
                                    "source": "tatqa_paragraph",
                                    "id": str(item.get('id', '')),
                                    "word_count": word_count,
                                    "sentence_count": sentence_count
                                }
                            ))
        
        except Exception as e:
            self.logger.error(f"Error processing TatQA data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return tatqa_docs
    
    def _process_table(self, table: Dict[str, Any]) -> str:
        """Convert table to text format"""
        if not isinstance(table, dict) or 'header' not in table or 'table' not in table:
            return ""
        
        try:
            headers = table['header']
            rows = table['table']
            
            # Process headers
            header_str = " | ".join(str(h).strip() for h in headers)
            
            # Process rows
            processed_rows = []
            for row in rows:
                # Ensure row has same length as headers
                if len(row) != len(headers):
                    row = row[:len(headers)] if len(row) > len(headers) else row + [''] * (len(headers) - len(row))
                
                # Clean and format row values
                row_values = []
                for value in row:
                    if isinstance(value, (int, float)):
                        # Format numbers with appropriate precision
                        row_values.append(f"{value:,}" if isinstance(value, int) else f"{value:,.2f}")
                    else:
                        row_values.append(str(value).strip())
                
                processed_rows.append(" | ".join(row_values))
            
            # Combine headers and rows
            table_str = f"{header_str}\n" + "\n".join(processed_rows)
            
            return table_str
        
        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")
            return ""
    
    def _load_local_alphafin_data(self) -> List[DocumentWithMetadata]:
        """Load AlphaFin data from local file, prefer QCA format, support dedup and smart question generation"""
        alphafin_docs = []
        qca_path = Path(self.data_dir) / "alphafin/alphafin_qca.json"
        raw_path = Path(self.data_dir) / "alphafin/sample_data.json"
        seen_questions = set()
        if qca_path.exists():
            try:
                with open(qca_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    question = item.get('question', '').strip()
                    context = item.get('context', '').strip()
                    answer = item.get('answer', '').strip()
                    # 智能问句生成
                    if not question:
                        question = context[:30] + '……请简要分析。'
                    # 去重
                    if not question or not context or not answer or question in seen_questions:
                        continue
                    seen_questions.add(question)
                    alphafin_docs.append(DocumentWithMetadata(
                        content=f"Question: {question}\nContext: {context}\nAnswer: {answer}",
                        metadata={
                            "source": "alphafin",
                            "question": question
                        }
                    ))
            except Exception as e:
                self.logger.error(f"Error loading QCA AlphaFin data: {str(e)}")
        elif raw_path.exists():
            # fallback to old logic
            try:
                with open(raw_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data:
                    instruction = item.get('instruction', '').strip()
                    input_text = item.get('input', '').strip()
                    output = item.get('output', '').strip()
                    question = instruction or input_text[:30] + '……请简要分析。'
                    if not question or not input_text or not output or question in seen_questions:
                        continue
                    seen_questions.add(question)
                    alphafin_docs.append(DocumentWithMetadata(
                        content=f"Question: {question}\nContext: {input_text}\nAnswer: {output}",
                        metadata={
                            "source": "alphafin",
                            "question": question
                        }
                    ))
            except Exception as e:
                self.logger.error(f"Error loading local AlphaFin data: {str(e)}")
        return alphafin_docs
    
    def _load_hf_alphafin_data(self) -> List[DocumentWithMetadata]:
        """Load AlphaFin data from HuggingFace"""
        alphafin_docs = []
        
        try:
            dataset = load_dataset(
                "C1em/alphafin",
                cache_dir=self.cache_dir
            )
            
            if 'train' in dataset:
                for item in dataset['train']:
                    content = item.get('content', '')
                    if content:
                        alphafin_docs.append(DocumentWithMetadata(
                            content=content,
                            metadata={
                                "source": "alphafin",
                                "id": item.get('id', ''),
                                "type": item.get('type', ''),
                                "category": item.get('category', '')
                            }
                        ))
        except Exception as e:
            self.logger.error(f"Error loading AlphaFin data from HuggingFace: {str(e)}")
        
        return alphafin_docs
    
    def build_unified_index(self, save_dir: Optional[str] = None):
        """Build unified search index"""
        print("\nBuilding unified index...")
        
        if not self.documents:
            print("No documents to index!")
            return
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            index_path = save_path / "unified.index"
            embeddings_path = save_path / "embeddings.npy"
            
            # Try to load cached index
            if self.use_cached_index and index_path.exists() and embeddings_path.exists():
                try:
                    print("Loading cached index...")
                    self.index = faiss.read_index(str(index_path))
                    self.corpus_embeddings = np.load(str(embeddings_path))
                    print("Successfully loaded cached index")
                    return
                except Exception as e:
                    print(f"Failed to load cached index: {e}")
        
        # Generate embeddings if needed
        print("Generating document embeddings...")
        texts = [doc.content for doc in self.documents]
        self.corpus_embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        if self.use_faiss:
            # Initialize FAISS index
            print("Building FAISS index...")
            self.index = faiss.IndexFlatL2(self.corpus_embeddings.shape[1])
            self.index.add(self.corpus_embeddings.astype('float32'))
        
        if save_dir:
            self._save_index(save_dir)
            print("Index cached for future use")
    
    def _save_index(self, save_dir: str):
        """Save index and related data"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_faiss and self.index is not None:
            faiss.write_index(self.index, str(save_path / "unified.index"))
        
        if self.corpus_embeddings is not None:
            np.save(str(save_path / "embeddings.npy"), self.corpus_embeddings)
    
    def retrieve(
        self,
        text: str,
        top_k: int = 3,
        return_scores: bool = False
    ) -> Union[List[DocumentWithMetadata], Tuple[List[DocumentWithMetadata], List[float]]]:
        """Retrieve most relevant documents"""
        if not text or not self.documents:
            return ([], []) if return_scores else []
        
        # Encode query
        query_embedding = self.encoder.encode(text)
        
        if self.use_faiss and self.index is not None:
            # Use FAISS for retrieval
            scores, indices = self.index.search(
                np.array([query_embedding]).astype('float32'),
                min(top_k, len(self.documents))
            )
            scores = scores[0]
            indices = indices[0]
        else:
            # Fallback to numpy
            scores = np.dot(self.corpus_embeddings, query_embedding)
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        # Get documents
        retrieved_docs = [self.documents[idx] for idx in indices]
        
        # Normalize scores to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        
        return (retrieved_docs, scores.tolist()) if return_scores else retrieved_docs 