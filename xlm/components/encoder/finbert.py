from typing import List
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tqdm import tqdm
from config.parameters import Config

class FinbertEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        cache_dir: str = None,
        device: str = None,
        batch_size: int = 32,
        max_length: int = 512
    ):
        # 使用config中的平台感知配置
        if cache_dir is None:
            config = Config()
            cache_dir = config.cache_dir
            
        super().__init__()
        self.model_name = model_name
        self.device = device if device else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        print(f"FinbertEncoder '{model_name}' loaded on {self.device}.")

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: List[str], batch_size: int = 32, show_progress_bar: bool = False) -> np.ndarray:
        all_embeddings = []
        
        # Create a tqdm iterator if requested
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Encoding Batches")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i:i+batch_size]
                encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                model_output = self.model(**encoded_input)
                sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                all_embeddings.append(sentence_embeddings.cpu().numpy())
        
        return np.concatenate(all_embeddings, axis=0)

    def get_embedding_dimension(self) -> int:
        return self.model.config.hidden_size 