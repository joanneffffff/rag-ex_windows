from typing import List
import torch
from sentence_transformers import SentenceTransformer


class Encoder:
    def __init__(
        self,
        # model_name: str = "all-MiniLM-L6-v2",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str = None,
        # cache_dir: str = "D:/AI/huggingface",
        cache_dir: str = "M:/huggingface",
    ):
        """
        初始化编码器
        Args:
            model_name: 模型名称（推荐 paraphrase-multilingual-MiniLM-L12-v2 支持多语言）
            # model_name: str = "all-MiniLM-L6-v2"
            device: 设备 (cpu/cuda)
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # 加载模型
        print(f"\n加载编码器模型: {model_name}")
        print(f"- 设备: {self.device}")
        print(f"- 缓存目录: {cache_dir}")
        
        self.model = SentenceTransformer(
            model_name_or_path=model_name,
            cache_folder=cache_dir,
            device=self.device
        )

    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        编码文本
        Args:
            texts: 文本列表
        Returns:
            编码向量列表
        """
        # 使用模型编码
        embeddings = self.model.encode(
            sentences=texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device
        )
        
        # 转换为列表
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
            
        return embeddings.tolist()
