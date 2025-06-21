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
        data_dir: str,
        max_samples: int,
        cache_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("M:/huggingface")
        self.max_samples = max_samples
        
        self.logger = logging.getLogger(__name__)
        
        print("准备加载数据 ...")
        self.english_docs: List[DocumentWithMetadata] = []
        self.chinese_docs: List[DocumentWithMetadata] = []
        self._load_data()
        print("数据加载完成")
    
    def _load_data(self):
        """Load and process all data sources"""
        # Load TatQA data
        tatqa_docs = self._load_tatqa_data()
        if len(tatqa_docs) > self.max_samples:
            print(f"Sampling {self.max_samples} documents from TatQA dataset...")
            self.english_docs = tatqa_docs[:self.max_samples]
        else:
            self.english_docs = tatqa_docs
        print(f"Loaded {len(self.english_docs)} TatQA documents (English)\n")
        
        # Load AlphaFin data
        try:
            print("Loading AlphaFin data...")
            alphafin_docs = self._load_local_alphafin_data()
            if not alphafin_docs:
                print("No local AlphaFin data found, trying HuggingFace...")
                alphafin_docs = self._load_hf_alphafin_data()
            
            if len(alphafin_docs) > self.max_samples:
                print(f"Sampling {self.max_samples} documents from AlphaFin dataset...")
                self.chinese_docs = alphafin_docs[:self.max_samples]
            else:
                self.chinese_docs = alphafin_docs

        except Exception as e:
            self.logger.error(f"Error loading AlphaFin data: {str(e)}")
            self.chinese_docs = []
        
        print(f"Loaded {len(self.chinese_docs)} AlphaFin documents (Chinese)\n")
    
    def _load_tatqa_data(self) -> List[DocumentWithMetadata]:
        """Load and process TatQA data"""
        tatqa_docs = []
        tatqa_path = self.data_dir / "tatqa_dataset_raw/tatqa_dataset_train.json"
        
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
                    
                table = item.get('table', {})
                paragraphs = item.get('paragraphs', [])
                questions = item.get('questions', [])

                # Process table into a string format
                table_str = ""
                if 'header' in table and 'rows' in table:
                    table_str = "Table: " + ", ".join(table['header']) + "\n"
                    for row in table['rows']:
                        table_str += ", ".join(map(str, row)) + "\n"

                # Process paragraphs into a single string
                para_texts = []
                for para in paragraphs:
                    if isinstance(para, dict) and 'text' in para:
                        para_texts.append(para['text'])
                    elif isinstance(para, str):
                        para_texts.append(para)
                para_str = "\n".join(para_texts)

                # Combine table and paragraphs for context
                full_context = f"{table_str}\n{para_str}".strip()
                
                # Create documents for each question
                for q in questions:
                    question_text = q.get('question', '')
                    answer = q.get('answer', '')
                    
                    if question_text and full_context:
                        # Storing the full context in the document content
                        doc_content = f"Question: {question_text}\nContext: {full_context}"
                        
                        tatqa_docs.append(DocumentWithMetadata(
                            content=doc_content,
                            metadata={
                                "source": "tatqa",
                                "id": str(item.get('uid', '')),
                                "question": question_text,
                                "answer": answer
                            }
                        ))

        except Exception as e:
            self.logger.error(f"Error processing TatQA data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return tatqa_docs

    def _load_local_alphafin_data(self) -> List[DocumentWithMetadata]:
        """Load AlphaFin data from the final processed local file."""
        alphafin_docs = []
        alphafin_path = self.data_dir / "alphafin" / "alphafin_final.json"
        if alphafin_path.exists():
            self.logger.info(f"Loading data from {alphafin_path}")
            with open(alphafin_path, 'r', encoding='utf-8') as f:
                alphafin_data = json.load(f)
            
            for record in tqdm(alphafin_data, desc="Processing Final AlphaFin Data"):
                content = record.get("content")
                metadata = record.get("metadata")

                if content and metadata:
                    alphafin_docs.append(DocumentWithMetadata(
                        content=content,
                        metadata=metadata
                    ))
        else:
            self.logger.warning(f"Final AlphaFin data file not found at {alphafin_path}")
        
        return alphafin_docs
    
    def _load_hf_alphafin_data(self) -> List[DocumentWithMetadata]:
        """Load AlphaFin data from HuggingFace as a fallback."""
        alphafin_docs = []
        try:
            dataset = load_dataset("C1em/alphafin", cache_dir=str(self.cache_dir))
            if 'train' in dataset:
                for item in tqdm(dataset['train'], desc="Processing HF AlphaFin"):
                    content = item.get('content', '')
                    if content:
                        alphafin_docs.append(DocumentWithMetadata(
                            content=content,
                            metadata={
                                "source": "alphafin_hf",
                                "id": item.get('id', ''),
                                "type": item.get('type', ''),
                                "category": item.get('category', '')
                            }
                        ))
        except Exception as e:
            self.logger.error(f"Error loading AlphaFin data from HuggingFace: {str(e)}")
        
        return alphafin_docs 