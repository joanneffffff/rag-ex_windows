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
import pandas as pd
from config.parameters import Config

class UnifiedDataLoader:
    def __init__(
        self,
        data_dir: str,
        max_samples: int,
        cache_dir: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        if cache_dir is None:
            cache_dir = Config().cache_dir
        self.cache_dir = Path(cache_dir)
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
        documents = []
        tatqa_path = self.data_dir / "tatqa_dataset_raw/tatqa_dataset_train.json"
        
        if not tatqa_path.exists():
            self.logger.warning(f"TatQA data file not found at {tatqa_path}")
            return documents
        
        print(f"Processing TatQA data from {tatqa_path}")
        
        try:
            with open(tatqa_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                raise ValueError("TatQA data should be a list")
                
            for i, item in enumerate(tqdm(data, desc="Loading TAT-QA data")):
                # Combine all paragraphs and tables from a single item into one document
                # to preserve context. This is the fix for the context fragmentation issue.
                
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
                metadata = {"source": source_file, "doc_id": doc_id, "language": "english"}

                if final_content:
                    documents.append(DocumentWithMetadata(
                        page_content=final_content,
                        metadata=metadata
                    ))

        except Exception as e:
            self.logger.error(f"Error processing TatQA data: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return documents

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

    def _table_to_text(self, table: pd.DataFrame, caption: str = "") -> str:
        """
        Convert a table to a text representation.
        """
        return f"Table: {caption}\n{table.to_markdown()}" 