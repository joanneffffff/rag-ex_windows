"""
优化的数据加载器
对中文数据使用文档级别chunking，对英文数据保持原有逻辑
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
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
import pandas as pd
from config.parameters import Config

class OptimizedDataLoader:
    def __init__(
        self,
        data_dir: str,
        max_samples: int,
        cache_dir: Optional[str] = None,
        chinese_document_level: bool = True,  # 中文使用文档级别
        english_chunk_level: bool = True,     # 英文保持chunk级别
    ):
        self.data_dir = Path(data_dir)
        if cache_dir is None:
            cache_dir = Config().cache_dir
        self.cache_dir = Path(cache_dir)
        self.max_samples = max_samples
        self.chinese_document_level = chinese_document_level
        self.english_chunk_level = english_chunk_level
        
        self.logger = logging.getLogger(__name__)
        
        print("准备加载优化数据 ...")
        self.english_docs: List[DocumentWithMetadata] = []
        self.chinese_docs: List[DocumentWithMetadata] = []
        self._load_data()
        print("数据加载完成")
    
    def _load_data(self):
        """Load and process all data sources with optimized chunking"""
        # Load TatQA data (English) - 保持原有逻辑
        tatqa_docs = self._load_tatqa_data()
        if self.max_samples > 0 and len(tatqa_docs) > self.max_samples:
            print(f"Sampling {self.max_samples} documents from TatQA dataset...")
            self.english_docs = tatqa_docs[:self.max_samples]
        else:
            self.english_docs = tatqa_docs
        print(f"Loaded {len(self.english_docs)} TatQA documents (English)\n")
        
        # Load AlphaFin data (Chinese) - 使用文档级别处理
        try:
            print("Loading AlphaFin data with document-level chunking...")
            alphafin_docs = self._load_optimized_alphafin_data()
            
            if self.max_samples > 0 and len(alphafin_docs) > self.max_samples:
                print(f"Sampling {self.max_samples} documents from AlphaFin dataset...")
                self.chinese_docs = alphafin_docs[:self.max_samples]
            else:
                self.chinese_docs = alphafin_docs

        except Exception as e:
            self.logger.error(f"Error loading AlphaFin data: {str(e)}")
            self.chinese_docs = []
        
        print(f"Loaded {len(self.chinese_docs)} AlphaFin documents (Chinese)\n")
    
    def _load_tatqa_data(self) -> List[DocumentWithMetadata]:
        """Load and process TatQA data - 保持原有逻辑"""
        documents = []
        
        # 使用配置文件中的路径
        config = Config()
        tatqa_path = Path(config.data.english_data_path)
        
        if not tatqa_path.exists():
            self.logger.warning(f"TatQA data file not found at {tatqa_path}")
            return documents
        
        print(f"Processing TatQA data from {tatqa_path}")
        
        try:
            # 检查文件格式
            if tatqa_path.suffix == '.jsonl':
                # 处理JSONL格式
                data = []
                with open(tatqa_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
            else:
                # 处理JSON格式
                with open(tatqa_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("TatQA data should be a list")
                
            for i, item in enumerate(tqdm(data, desc="Loading TAT-QA data")):
                # 处理不同的数据格式
                if isinstance(item, dict):
                    if 'paragraphs' in item and 'tables' in item:
                        # 原始TatQA格式
                        full_context_parts = []
                        
                        # Add all paragraphs
                        paragraphs = item.get("paragraphs", [])
                        for para in paragraphs:
                            if para_text := para.get("text", "").strip():
                                full_context_parts.append(para_text)

                        # Add all tables, converting them to text
                        tables = item.get("tables", [])
                        for table_data in tables:
                            if table_data.get('table'):
                                table_df = pd.DataFrame(table_data['table'], columns=table_data['header'])
                                table_text = self._table_to_text(table_df, table_data.get('caption', ''))
                                full_context_parts.append(table_text)
                        
                        # Join all parts into a single page_content string
                        final_content = "\n\n---\n\n".join(full_context_parts)
                        
                        # Create metadata for the entire document
                        source_file = tatqa_path.name
                        doc_id = item.get("uid", f"doc_{i}")
                        metadata = DocumentMetadata(
                            source=source_file,
                            created_at="",
                            author="",
                            language="english"
                        )
                    elif 'question' in item and 'context' in item:
                        # 问答格式：使用context作为内容
                        final_content = item.get("context", "")
                        metadata = DocumentMetadata(
                            source=tatqa_path.name,
                            created_at="",
                            author="",
                            language="english"
                        )
                    else:
                        # 其他格式：尝试找到文本内容
                        final_content = str(item)
                        metadata = DocumentMetadata(
                            source=tatqa_path.name,
                            created_at="",
                            author="",
                            language="english"
                        )
                else:
                    final_content = str(item)
                    metadata = DocumentMetadata(
                        source=tatqa_path.name,
                        created_at="",
                        author="",
                        language="english"
                    )

                if final_content:
                    documents.append(DocumentWithMetadata(
                        content=final_content,
                        metadata=metadata
                    ))

        except Exception as e:
            self.logger.error(f"Error processing TatQA data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return documents

    def _load_optimized_alphafin_data(self) -> List[DocumentWithMetadata]:
        """Load AlphaFin data with document-level chunking"""
        alphafin_docs = []
        
        # 使用配置文件中的路径
        config = Config()
        alphafin_path = Path(config.data.chinese_data_path)
        
        if not alphafin_path.exists():
            self.logger.warning(f"AlphaFin data file not found at {alphafin_path}")
            return alphafin_docs
        
        print(f"Loading optimized AlphaFin data from {alphafin_path}")
        
        try:
            # 检查文件格式
            if alphafin_path.suffix == '.jsonl':
                # 处理JSONL格式
                alphafin_data = []
                with open(alphafin_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            alphafin_data.append(json.loads(line))
            else:
                # 处理JSON格式
                with open(alphafin_path, 'r', encoding='utf-8') as f:
                    alphafin_data = json.load(f)
            
            print(f"Found {len(alphafin_data)} documents in AlphaFin data")
            
            for record in tqdm(alphafin_data, desc="Processing Optimized AlphaFin Data"):
                # 处理不同的数据格式
                if isinstance(record, dict):
                    if 'content' in record:
                        # 新格式：直接有content字段
                        content = record.get("content")
                        metadata = record.get("metadata", {})
                    elif 'question' in record and 'context' in record:
                        # 问答格式：使用context作为内容
                        content = record.get("context")
                        metadata = {
                            "source": "alphafin_qa",
                            "question": record.get("question", ""),
                            "language": "chinese"
                        }
                    else:
                        # 其他格式：尝试找到文本内容
                        content = str(record)
                        metadata = {"source": "alphafin", "language": "chinese"}
                else:
                    content = str(record)
                    metadata = {"source": "alphafin", "language": "chinese"}
                
                if content and isinstance(content, str):
                    # 文档级别处理：直接使用整个文档作为chunk
                    if self.chinese_document_level:
                        # 检查文档长度，如果太长则适当分割
                        if len(content) > 8192:  # 8K字符限制
                            # 按段落分割长文档
                            paragraphs = content.split('\n\n')
                            if len(paragraphs) > 1:
                                # 合并段落直到达到合理长度
                                merged_chunks = self._merge_paragraphs_to_chunks(paragraphs, max_length=8192)
                                for i, chunk_content in enumerate(merged_chunks):
                                    chunk_metadata = DocumentMetadata(
                                        source=f"{metadata.get('source', 'alphafin')}_doc_{i}",
                                        created_at=metadata.get('created_at', ''),
                                        author=metadata.get('author', ''),
                                        language="chinese"
                                    )
                                    
                                    chunk_doc = DocumentWithMetadata(
                                        content=chunk_content,
                                        metadata=chunk_metadata
                                    )
                                    alphafin_docs.append(chunk_doc)
                            else:
                                # 单个长段落，按句子分割
                                sentences = content.split('。')
                                merged_chunks = self._merge_sentences_to_chunks(sentences, max_length=8192)
                                for i, chunk_content in enumerate(merged_chunks):
                                    chunk_metadata = DocumentMetadata(
                                        source=f"{metadata.get('source', 'alphafin')}_doc_{i}",
                                        created_at=metadata.get('created_at', ''),
                                        author=metadata.get('author', ''),
                                        language="chinese"
                                    )
                                    
                                    chunk_doc = DocumentWithMetadata(
                                        content=chunk_content,
                                        metadata=chunk_metadata
                                    )
                                    alphafin_docs.append(chunk_doc)
                        else:
                            # 文档长度适中，直接使用
                            doc_metadata = DocumentMetadata(
                                source=metadata.get('source', 'alphafin'),
                                created_at=metadata.get('created_at', ''),
                                author=metadata.get('author', ''),
                                language="chinese"
                            )
                            
                            doc = DocumentWithMetadata(
                                content=content,
                                metadata=doc_metadata
                            )
                            alphafin_docs.append(doc)
                    else:
                        # 使用原有的chunk级别处理
                        doc_metadata = DocumentMetadata(
                            source=metadata.get('source', 'alphafin'),
                            created_at=metadata.get('created_at', ''),
                            author=metadata.get('author', ''),
                            language="chinese"
                        )
                        
                        doc = DocumentWithMetadata(
                            content=content,
                            metadata=doc_metadata
                        )
                        alphafin_docs.append(doc)
        
        except Exception as e:
            self.logger.error(f"Error loading optimized AlphaFin data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        print(f"Processed {len(alphafin_docs)} Chinese documents/chunks")
        return alphafin_docs
    
    def _merge_paragraphs_to_chunks(self, paragraphs: List[str], max_length: int = 8192) -> List[str]:
        """合并段落到chunks"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            if current_length + para_length <= max_length:
                current_chunk.append(para)
                current_length += para_length
            else:
                # 保存当前chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # 开始新的chunk
                if para_length <= max_length:
                    current_chunk = [para]
                    current_length = para_length
                else:
                    # 段落太长，需要分割
                    sub_chunks = self._split_long_paragraph(para, max_length)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
        
        # 保存最后一个chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _merge_sentences_to_chunks(self, sentences: List[str], max_length: int = 8192) -> List[str]:
        """合并句子到chunks"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_length = len(sentence)
            
            if current_length + sentence_length <= max_length:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # 保存当前chunk
                if current_chunk:
                    chunks.append('。'.join(current_chunk) + '。')
                
                # 开始新的chunk
                if sentence_length <= max_length:
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    # 句子太长，需要分割
                    sub_chunks = self._split_long_sentence(sentence, max_length)
                    chunks.extend(sub_chunks)
                    current_chunk = []
                    current_length = 0
        
        # 保存最后一个chunk
        if current_chunk:
            chunks.append('。'.join(current_chunk) + '。')
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, max_length: int) -> List[str]:
        """分割长段落"""
        if len(paragraph) <= max_length:
            return [paragraph]
        
        chunks = []
        start = 0
        
        while start < len(paragraph):
            end = start + max_length
            
            # 尝试在句号处截断
            if end < len(paragraph):
                for i in range(end, max(start + max_length - 100, start), -1):
                    if paragraph[i] in '。！？；':
                        end = i + 1
                        break
            
            chunk = paragraph[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end
        
        return chunks
    
    def _split_long_sentence(self, sentence: str, max_length: int) -> List[str]:
        """分割长句子"""
        if len(sentence) <= max_length:
            return [sentence]
        
        chunks = []
        start = 0
        
        while start < len(sentence):
            end = start + max_length
            chunk = sentence[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        return chunks

    def _table_to_text(self, table: pd.DataFrame, caption: str = "") -> str:
        """Convert a table to a text representation."""
        return f"Table: {caption}\n{table.to_markdown()}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        chinese_lengths = [len(doc.content) for doc in self.chinese_docs]
        english_lengths = [len(doc.content) for doc in self.english_docs]
        
        return {
            "chinese_docs": len(self.chinese_docs),
            "english_docs": len(self.english_docs),
            "chinese_avg_length": sum(chinese_lengths) / len(chinese_lengths) if chinese_lengths else 0,
            "english_avg_length": sum(english_lengths) / len(english_lengths) if english_lengths else 0,
            "chinese_min_length": min(chinese_lengths) if chinese_lengths else 0,
            "english_min_length": min(english_lengths) if english_lengths else 0,
            "chinese_max_length": max(chinese_lengths) if chinese_lengths else 0,
            "english_max_length": max(english_lengths) if english_lengths else 0,
        } 