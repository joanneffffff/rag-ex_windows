from typing import List, Dict, Union
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from config.parameters import Config


class Encoder:
    def __init__(
        self,
        # model_name: str = "all-MiniLM-L6-v2",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: Union[str, None] = None,
        # cache_dir: str = "D:/AI/huggingface",
        cache_dir: str = None,
    ):
        """
        初始化编码器
        Args:
            model_name: 模型名称（推荐 paraphrase-multilingual-MiniLM-L12-v2 支持多语言）
            # model_name: str = "all-MiniLM-L6-v2"
            device: 设备 (cpu/cuda)
            cache_dir: 模型缓存目录
        """
        if cache_dir is None:
            cache_dir = Config().cache_dir
        self.cache_dir = cache_dir
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        print(f"\n加载编码器模型: {model_name}")
        print(f"- 设备: {self.device}")
        print(f"- 缓存目录: {cache_dir}")
        
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_dir,
            device=self.device
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        编码文本
        Args:
            texts: 文本列表
        Returns:
            编码向量numpy数组
        """
        # 使用模型编码
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=False,  # 直接返回numpy数组
            device=self.device
        )
        
        return embeddings

    def encode_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        """
        批量编码文本（兼容性方法）
        Args:
            texts: 文本列表
        Returns:
            包含'text'键的字典，值为编码向量numpy数组
        """
        embeddings = self.encode(texts)
        return {'text': embeddings}
