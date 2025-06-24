"""
Configuration parameters for the RAG system.
"""

import os
import platform
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List

# --- Platform-Aware Path Configuration ---
# Set the default Hugging Face cache directory based on the operating system.
# You can modify the Windows path here if needed (e.g., "D:/AI/huggingface").
WINDOWS_CACHE_DIR = "M:/huggingface"
LINUX_CACHE_DIR = "/users/sgjfei3/data/huggingface"

DEFAULT_CACHE_DIR = WINDOWS_CACHE_DIR if platform.system() == "Windows" else LINUX_CACHE_DIR

@dataclass
class EncoderConfig:
    # 默认使用多语言模型，支持中英文
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # 中文微调模型路径
    chinese_model_path: str = "models/finetuned_alphafin_zh"
    # 英文微调模型路径
    english_model_path: str = "models/finetuned_finbert_tatqa"
    cache_dir: str = DEFAULT_CACHE_DIR
    device: Optional[str] = None  # Will auto-detect if None
    batch_size: int = 32
    max_length: int = 512

@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    cache_dir: str = DEFAULT_CACHE_DIR
    device: Optional[str] = None  # Will auto-detect if None
    use_quantization: bool = True
    quantization_type: str = "8bit"  # "8bit" or "4bit"
    batch_size: int = 4
    enabled: bool = True  # 是否启用重排序器

@dataclass
class RetrieverConfig:
    use_faiss: bool = True  # Changed default to True for efficiency
    num_threads: int = 4
    batch_size: int = 32
    use_gpu: bool = torch.cuda.is_available() # Dynamically set default based on hardware
    max_context_length: int = 100
    # 重排序相关配置
    retrieval_top_k: int = 100  # FAISS检索的top-k
    rerank_top_k: int = 10      # 重排序后的top-k

@dataclass
class DataConfig:
    data_dir: str = "data"  # Unified root data directory
    max_samples: int = 500 # Max samples to load from each dataset

@dataclass
class ModalityConfig:
    text_weight: float = 1.0
    table_weight: float = 1.0
    time_series_weight: float = 1.0
    combine_method: str = "weighted_sum"  # or "concatenate", "attention"

@dataclass
class SystemConfig:
    memory_limit: int = 16  # in GB
    log_level: str = "INFO"
    temp_dir: str = "temp"

@dataclass
class GeneratorConfig:
    model_name: str = "Qwen/Qwen2-1.5B-Instruct"
    # model_name: str = "Qwen/Qwen3-8B"
    cache_dir: str = DEFAULT_CACHE_DIR

@dataclass
class Config:
    cache_dir: str = DEFAULT_CACHE_DIR # Global cache directory
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    def __post_init__(self):
        # Propagate the global cache_dir to other configs if they have it
        if hasattr(self.encoder, 'cache_dir'):
            self.encoder.cache_dir = self.cache_dir
        if hasattr(self.reranker, 'cache_dir'):
            self.reranker.cache_dir = self.cache_dir
        if hasattr(self.generator, 'cache_dir'):
            self.generator.cache_dir = self.cache_dir # Bug fix: Correctly assign global cache_dir

    @classmethod
    def load_environment_config(cls) -> 'Config':
        """Load configuration based on environment"""
        # Example of environment-based config loading
        if os.getenv("PRODUCTION") == "1":
            return cls(
                encoder=EncoderConfig(
                    model_name="all-mpnet-base-v2",
                    batch_size=64
                ),
                retriever=RetrieverConfig(
                    use_faiss=True,
                    num_threads=8,
                    use_gpu=True
                ),
                reranker=RerankerConfig(
                    enabled=True,
                    use_quantization=True
                ),
                system=SystemConfig(
                    memory_limit=32,
                    log_level="WARNING"
                )
            )
        return cls()  # Default development config

# Default configuration instance
config = Config() 