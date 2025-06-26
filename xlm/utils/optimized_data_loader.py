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
import re
import ast

class OptimizedDataLoader:
    def __init__(
        self,
        data_dir: str,
        max_samples: int,
        cache_dir: Optional[str] = None,
        chinese_document_level: bool = True,  # 中文使用文档级别
        english_chunk_level: bool = True,     # 英文保持chunk级别
        include_eval_data: bool = True,       # 是否包含评估数据到知识库
    ):
        self.data_dir = Path(data_dir)
        if cache_dir is None:
            cache_dir = Config().cache_dir
        self.cache_dir = Path(cache_dir)
        self.max_samples = max_samples
        self.chinese_document_level = chinese_document_level
        self.english_chunk_level = english_chunk_level
        self.include_eval_data = include_eval_data
        
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
        
        # 根据参数决定是否添加评估数据到知识库
        if self.include_eval_data:
            print("Loading evaluation data to knowledge base...")
            try:
                # 加载中文评估数据
                alphafin_eval_docs = self._load_eval_data("evaluate_mrr/alphafin_eval.jsonl", "chinese")
                self.chinese_docs.extend(alphafin_eval_docs)
                print(f"Added {len(alphafin_eval_docs)} AlphaFin evaluation documents to Chinese knowledge base")
                
                # 加载英文评估数据
                tatqa_eval_docs = self._load_eval_data("evaluate_mrr/tatqa_eval.jsonl", "english")
                self.english_docs.extend(tatqa_eval_docs)
                print(f"Added {len(tatqa_eval_docs)} TatQA evaluation documents to English knowledge base")
                
            except Exception as e:
                self.logger.error(f"Error loading evaluation data: {str(e)}")
                print("Warning: Could not load evaluation data, continuing with training data only")
        else:
            print("Skipping evaluation data (not included in knowledge base for fair evaluation)")
        
        print(f"Final knowledge base size: {len(self.chinese_docs)} Chinese docs, {len(self.english_docs)} English docs")
    
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
                    # 文档级别处理：保持原始格式，避免过度分割
                    if self.chinese_document_level:
                        # 直接使用原始内容，不进行JSON转换
                        # 这样可以保持文档的完整性，避免过度分割
                        
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
                            # 文档长度适中，直接使用原始内容
                            doc_metadata = DocumentMetadata(
                                source=metadata.get('source', 'alphafin'),
                                created_at=metadata.get('created_at', ''),
                                author=metadata.get('author', ''),
                                language="chinese"
                            )
                            
                            alphafin_doc = DocumentWithMetadata(
                                content=content,  # 使用原始内容
                                metadata=doc_metadata
                            )
                            alphafin_docs.append(alphafin_doc)
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

    def _load_eval_data(self, eval_file: str, language: str) -> List[DocumentWithMetadata]:
        """Load evaluation data and convert to documents"""
        eval_docs = []
        
        try:
            eval_path = Path(eval_file)
            if not eval_path.exists():
                self.logger.warning(f"Evaluation file not found at {eval_path}")
                return eval_docs
            
            print(f"Loading evaluation data from {eval_path}")
            
            # 检查文件格式
            if eval_path.suffix == '.jsonl':
                # 处理JSONL格式
                eval_data = []
                with open(eval_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            eval_data.append(json.loads(line))
            else:
                # 处理JSON格式
                with open(eval_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)
            
            print(f"Found {len(eval_data)} evaluation samples")
            
            for record in tqdm(eval_data, desc=f"Processing {language} evaluation data"):
                # 处理不同的数据格式
                if isinstance(record, dict):
                    if 'context' in record:
                        # 使用context作为文档内容
                        content = record.get("context")
                        question = record.get("question", "")
                        metadata = {
                            "source": f"eval_{language}",
                            "question": question,
                            "language": language,
                            "eval_data": True
                        }
                    else:
                        # 其他格式：尝试找到文本内容
                        content = str(record)
                        metadata = {"source": f"eval_{language}", "language": language, "eval_data": True}
                else:
                    content = str(record)
                    metadata = {"source": f"eval_{language}", "language": language, "eval_data": True}
                
                if content and isinstance(content, str):
                    # 文档级别处理：直接使用整个文档作为chunk
                    if language == "chinese" and self.chinese_document_level:
                        # 检查文档长度，如果太长则适当分割
                        if len(content) > 8192:  # 8K字符限制
                            # 按段落分割长文档
                            paragraphs = content.split('\n\n')
                            if len(paragraphs) > 1:
                                # 合并段落直到达到合理长度
                                merged_chunks = self._merge_paragraphs_to_chunks(paragraphs, max_length=8192)
                                for i, chunk_content in enumerate(merged_chunks):
                                    chunk_metadata = DocumentMetadata(
                                        source=f"{metadata.get('source', 'eval_chinese')}_doc_{i}",
                                        created_at="",
                                        author="",
                                        language="chinese"
                                    )
                                    
                                    chunk_doc = DocumentWithMetadata(
                                        content=chunk_content,
                                        metadata=chunk_metadata
                                    )
                                    eval_docs.append(chunk_doc)
                            else:
                                # 单个长段落，按句子分割
                                sentences = content.split('。')
                                merged_chunks = self._merge_sentences_to_chunks(sentences, max_length=8192)
                                for i, chunk_content in enumerate(merged_chunks):
                                    chunk_metadata = DocumentMetadata(
                                        source=f"{metadata.get('source', 'eval_chinese')}_doc_{i}",
                                        created_at="",
                                        author="",
                                        language="chinese"
                                    )
                                    
                                    chunk_doc = DocumentWithMetadata(
                                        content=chunk_content,
                                        metadata=chunk_metadata
                                    )
                                    eval_docs.append(chunk_doc)
                        else:
                            # 文档长度适中，直接使用
                            doc_metadata = DocumentMetadata(
                                source=metadata.get('source', 'eval_chinese'),
                                created_at="",
                                author="",
                                language="chinese"
                            )
                            
                            eval_doc = DocumentWithMetadata(
                                content=content,
                                metadata=doc_metadata
                            )
                            eval_docs.append(eval_doc)
                    else:
                        # 英文或其他语言，使用简单chunking
                        if language == "english" and self.english_chunk_level:
                            # 英文使用与训练数据相同的chunk策略
                            if len(content) > 1500:  # 英文chunk长度限制
                                # 按段落分割长文档
                                paragraphs = content.split('\n\n')
                                if len(paragraphs) > 1:
                                    # 合并段落直到达到合理长度
                                    merged_chunks = self._merge_paragraphs_to_chunks(paragraphs, max_length=1500)
                                    for i, chunk_content in enumerate(merged_chunks):
                                        chunk_metadata = DocumentMetadata(
                                            source=f"{metadata.get('source', f'eval_{language}')}_chunk_{i}",
                                            created_at="",
                                            author="",
                                            language=language
                                        )
                                        
                                        chunk_doc = DocumentWithMetadata(
                                            content=chunk_content,
                                            metadata=chunk_metadata
                                        )
                                        eval_docs.append(chunk_doc)
                                else:
                                    # 单个长段落，按句子分割
                                    sentences = content.split('. ')
                                    merged_chunks = self._merge_sentences_to_chunks(sentences, max_length=1500)
                                    for i, chunk_content in enumerate(merged_chunks):
                                        chunk_metadata = DocumentMetadata(
                                            source=f"{metadata.get('source', f'eval_{language}')}_chunk_{i}",
                                            created_at="",
                                            author="",
                                            language=language
                                        )
                                        
                                        chunk_doc = DocumentWithMetadata(
                                            content=chunk_content,
                                            metadata=chunk_metadata
                                        )
                                        eval_docs.append(chunk_doc)
                            else:
                                # 文档长度适中，直接使用
                                doc_metadata = DocumentMetadata(
                                    source=metadata.get('source', f'eval_{language}'),
                                    created_at="",
                                    author="",
                                    language=language
                                )
                                
                                eval_doc = DocumentWithMetadata(
                                    content=content,
                                    metadata=doc_metadata
                                )
                                eval_docs.append(eval_doc)
                        else:
                            # 其他语言，使用简单处理
                            doc_metadata = DocumentMetadata(
                                source=metadata.get('source', f'eval_{language}'),
                                created_at="",
                                author="",
                                language=language
                            )
                            
                            eval_doc = DocumentWithMetadata(
                                content=content,
                                metadata=doc_metadata
                            )
                            eval_docs.append(eval_doc)
            
        except Exception as e:
            self.logger.error(f"Error processing evaluation data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return eval_docs 

def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    """
    Parses a JSON string context from AlphaFin and converts it into a list of
    natural language chunks, handling various formats and cleaning.
    """
    chunks = []
    
    if not json_str_context or not json_str_context.strip():
        return chunks

    processed_str_context = json_str_context.replace("\\n", "\n")

    cleaned_initial = re.sub(re.escape("【问题】:"), "", processed_str_context)
    cleaned_initial = re.sub(re.escape("【答案】:"), "", cleaned_initial).strip()
    
    cleaned_initial = cleaned_initial.replace('，', ',')
    cleaned_initial = cleaned_initial.replace('：', ':')
    cleaned_initial = cleaned_initial.replace('【', '') 
    cleaned_initial = cleaned_initial.replace('】', '') 
    cleaned_initial = cleaned_initial.replace('\u3000', ' ')
    cleaned_initial = cleaned_initial.replace('\xa0', ' ').strip()
    cleaned_initial = re.sub(r'\s+', ' ', cleaned_initial).strip()

    report_match = re.match(
        r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
        cleaned_initial, 
        re.DOTALL
    )
    
    if report_match:
        report_title_full = report_match.group(1).strip()
        report_date = report_match.group(2).strip()
        report_raw_content = report_match.group(3).strip() 

        content_after_second_title_match = re.match(r"研报题目是:(.+)", report_raw_content, re.DOTALL)
        if content_after_second_title_match:
            report_content_preview = content_after_second_title_match.group(1).strip()
        else:
            report_content_preview = report_raw_content 
            
        report_content_preview = re.sub(re.escape("【问题】:"), "", report_content_preview)
        report_content_preview = re.sub(re.escape("【答案】:"), "", report_content_preview).strip()
        report_content_preview = re.sub(r'\s+', ' ', report_content_preview).strip() 

        company_stock_match = re.search(r"(.+?)（(\d{6}\.\w{2})）", report_title_full)
        company_info = ""
        if company_stock_match:
            report_company_name = company_stock_match.group(1).strip()
            report_stock_code = company_stock_match.group(2).strip()
            company_info = f"，公司名称：{report_company_name}，股票代码：{report_stock_code}"
            report_title_main = re.sub(r"（\d{6}\.\w{2}）", "", report_title_full).strip()
        else:
            report_title_main = report_title_full

        chunk_text = f"一份发布日期为 {report_date} 的研究报告，其标题是：{report_title_main}{company_info}。报告摘要内容：{report_content_preview.rstrip('...') if report_content_preview.endswith('...') else report_content_preview}。"
        chunks.append(chunk_text)
        return chunks 

    extracted_dict_str = None
    parsed_data = None 

    temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_initial) 
    all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 

    for potential_dict_str in all_dict_matches:
        cleaned_potential_dict_str = potential_dict_str.strip()
        
        json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
        try:
            parsed_data_temp = json.loads(json_compatible_str_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break 
        except json.JSONDecodeError:
            pass 

        fixed_for_ast_eval_temp = re.sub(
            r"(?<!['\"\w.])\b(0[1-9]\d*)\b(?![\d.]|['\"\w.])", 
            r"'\1'", 
            cleaned_potential_dict_str
        )
        try:
            parsed_data_temp = ast.literal_eval(fixed_for_ast_eval_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break 
        except (ValueError, SyntaxError):
            pass 

    if extracted_dict_str is not None and isinstance(parsed_data, dict):
        for metric_name, time_series_data in parsed_data.items():
            if not isinstance(metric_name, str):
                metric_name = str(metric_name)

            cleaned_metric_name = re.sub(r'（.*?）', '', metric_name).strip()
            
            if not isinstance(time_series_data, dict):
                if time_series_data is not None and str(time_series_data).strip():
                    chunks.append(f"{company_name}的{cleaned_metric_name}数据为：{time_series_data}。")
                continue
            if not time_series_data:
                continue
            
            try:
                sorted_dates = sorted(time_series_data.keys(), key=str)
            except TypeError:
                sorted_dates = [str(k) for k in time_series_data.keys()]
                
            description_parts = []
            for date in sorted_dates:
                value = time_series_data[date]
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                description_parts.append(f"在{date}为{formatted_value}")
            
            if description_parts:
                if len(description_parts) <= 3:
                    full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                else:
                    first_part = "，".join(description_parts[:3])
                    last_part = "，".join(description_parts[-3:])
                    if len(sorted_dates) > 6:
                        full_description = f"{company_name}的{cleaned_metric_name}数据从{sorted_dates[0]}到{sorted_dates[-1]}，主要变化为：{first_part}，...，{last_part}。"
                    else:
                        full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                chunks.append(full_description)
        return chunks 

    pure_text = cleaned_initial
    pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", pure_text, 1).strip()
    pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()
    pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()
    pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()

    if pure_text: 
        chunks.append(pure_text)
    else:
        print(f"警告：未能在 context 字符串中找到有效结构 (字典、研报或纯文本)。原始字符串（前100字符）：{json_str_context[:100]}...")
        chunks.append(f"原始格式，解析失败或无有效结构：{json_str_context.strip()[:100]}...")

    return chunks 