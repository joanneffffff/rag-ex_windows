"""
Specialized encoders for table and time series data
"""

from typing import List, Dict, Union, Optional
import torch
import numpy as np
from torch import nn
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

class SmartChunker:
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
    
    def chunk_text(self, text: str) -> List[str]:
        """Smart text chunking that preserves semantic units"""
        # Split into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If single sentence is too long, split it
            if sentence_length > self.max_length:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence into smaller parts
                words = sentence.split()
                for i in range(0, len(words), self.max_length):
                    chunk = " ".join(words[i:i + self.max_length])
                    chunks.append(chunk)
            else:
                # Add sentence to current chunk if it fits
                if current_length + sentence_length <= self.max_length:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    # Start new chunk
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks

class TableEncoder(nn.Module):
    def __init__(self, 
        # base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_model: str = "paraphrase-multilingual-MiniLM-L12-v2", 
        device: str = "cpu"):
        super().__init__()
        self.device = device
        self.base_model = SentenceTransformer(base_model, device=device)
        self.table_processor = TableProcessor()
        self.chunker = SmartChunker()
        
    def forward(self, tables: List[Dict]) -> torch.Tensor:
        # Process tables to text
        processed_tables = [self.table_processor.process(table) for table in tables]
        
        # Chunk long table descriptions
        chunked_tables = []
        for table in processed_tables:
            chunks = self.chunker.chunk_text(table)
            chunked_tables.extend(chunks)
        
        # Encode using base model
        return self.base_model.encode(chunked_tables, convert_to_tensor=True)
    
    def encode(self, tables: List[Dict]) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.forward(tables)
            return embeddings.cpu().numpy()

class TimeSeriesEncoder(nn.Module):
    def __init__(self, 
        # base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        base_model: str = "paraphrase-multilingual-MiniLM-L12-v2", 
        device: str = "cpu"):
        super().__init__()
        self.device = device
        self.base_model = SentenceTransformer(base_model, device=device)
        self.ts_processor = TimeSeriesProcessor()
        self.chunker = SmartChunker()
        
    def forward(self, time_series: List[Dict]) -> torch.Tensor:
        # Process time series to text
        processed_ts = [self.ts_processor.process(ts) for ts in time_series]
        
        # Chunk long time series descriptions
        chunked_ts = []
        for ts in processed_ts:
            chunks = self.chunker.chunk_text(ts)
            chunked_ts.extend(chunks)
        
        # Encode using base model
        return self.base_model.encode(chunked_ts, convert_to_tensor=True)
    
    def encode(self, time_series: List[Dict]) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.forward(time_series)
            return embeddings.cpu().numpy()

class TableProcessor:
    def process(self, table: Dict) -> str:
        """Convert table to structured text format"""
        if not isinstance(table, dict) or 'header' not in table or 'table' not in table:
            return ""
        
        headers = table['header']
        rows = table['table']
        
        # Process headers
        header_str = " | ".join(str(h).strip() for h in headers)
        
        # Process rows with type information
        processed_rows = []
        for row in rows:
            row_values = []
            for value in row:
                if isinstance(value, (int, float)):
                    # Format numbers with type information
                    row_values.append(f"{value:,}" if isinstance(value, int) else f"{value:,.2f}")
                else:
                    row_values.append(str(value).strip())
            processed_rows.append(" | ".join(row_values))
        
        # Add table structure information
        table_str = f"Table Structure:\n"
        table_str += f"Headers: {header_str}\n"
        table_str += f"Number of rows: {len(rows)}\n"
        table_str += f"Number of columns: {len(headers)}\n"
        table_str += "\nTable Content:\n"
        table_str += f"{header_str}\n"
        table_str += "\n".join(processed_rows)
        
        return table_str

class TimeSeriesProcessor:
    def process(self, time_series: Dict) -> str:
        """Convert time series to structured text format"""
        if not isinstance(time_series, dict):
            return ""
        
        # Convert to pandas Series for analysis
        series = pd.Series(time_series)
        
        # Calculate statistics
        stats = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'trend': 'increasing' if series.iloc[-1] > series.iloc[0] else 'decreasing'
        }
        
        # Create structured text
        text = "Time Series Analysis:\n"
        text += f"Period: {series.index[0]} to {series.index[-1]}\n"
        text += f"Number of points: {len(series)}\n"
        text += f"Statistics:\n"
        text += f"- Mean: {stats['mean']:.4f}\n"
        text += f"- Median: {stats['median']:.4f}\n"
        text += f"- Standard Deviation: {stats['std']:.4f}\n"
        text += f"- Range: {stats['min']:.4f} to {stats['max']:.4f}\n"
        text += f"- Trend: {stats['trend']}\n"
        
        # Add raw data
        text += "\nRaw Data:\n"
        for date, value in series.items():
            text += f"{date}: {value}\n"
        
        return text 