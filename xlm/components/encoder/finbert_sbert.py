"""
FinBERT and Sentence-BERT fusion for financial text encoding
"""

from typing import List, Dict, Union, Optional
import torch
import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer

class FinBertSBERT(nn.Module):
    def __init__(
        self,
        finbert_model: str = "ProsusAI/finbert",
        # sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        sbert_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
        fusion_method: str = "concat"
    ):
        super().__init__()
        self.finbert = AutoModel.from_pretrained(finbert_model)
        self.sbert = SentenceTransformer(sbert_model)
        self.fusion_method = fusion_method
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion_layer = nn.Linear(
                self.finbert.config.hidden_size + self.sbert.get_sentence_embedding_dimension(),
                self.sbert.get_sentence_embedding_dimension()
            )
        elif fusion_method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=self.finbert.config.hidden_size,
                num_heads=8
            )
            self.fusion_layer = nn.Linear(
                self.finbert.config.hidden_size,
                self.sbert.get_sentence_embedding_dimension()
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(finbert_model)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Get FinBERT embeddings
        finbert_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        finbert_outputs = self.finbert(**finbert_inputs)
        finbert_embeddings = finbert_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Get SBERT embeddings
        sbert_embeddings = self.sbert.encode(texts, convert_to_tensor=True)
        
        # Fuse embeddings
        if self.fusion_method == "concat":
            combined = torch.cat([finbert_embeddings, sbert_embeddings], dim=1)
            fused = self.fusion_layer(combined)
        elif self.fusion_method == "attention":
            # Use attention to weight FinBERT features
            attn_output, _ = self.attention(
                finbert_embeddings.unsqueeze(0),
                finbert_embeddings.unsqueeze(0),
                finbert_embeddings.unsqueeze(0)
            )
            fused = self.fusion_layer(attn_output.squeeze(0))
        
        return fused
    
    def encode(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            embeddings = self.forward(texts)
            return embeddings.cpu().numpy()
    
    def get_embedding_dimension(self) -> int:
        return self.sbert.get_sentence_embedding_dimension()

class FinancialTextProcessor:
    def __init__(self):
        self.sentiment_map = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
    
    def process(self, text: str) -> str:
        """Process financial text with additional context"""
        # Add financial context markers
        processed = "Financial Text Analysis:\n"
        processed += f"Content: {text}\n"
        
        # Add common financial terms
        financial_terms = [
            "revenue", "profit", "loss", "earnings", "dividend",
            "stock", "market", "price", "value", "growth"
        ]
        
        # Check for financial terms
        found_terms = [term for term in financial_terms if term.lower() in text.lower()]
        if found_terms:
            processed += f"\nFinancial Terms Found: {', '.join(found_terms)}\n"
        
        return processed 