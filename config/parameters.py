"""
Configuration parameters for the RAG system.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class EncoderConfig:
    # model_name: str = "all-MiniLM-L6-v2"
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # cache_dir: str = "M:/huggingface"
    cache_dir: str = "D:/AI/huggingface"
    device: Optional[str] = None  # Will auto-detect if None
    batch_size: int = 32
    max_length: int = 512

@dataclass
class RetrieverConfig:
    use_faiss: bool = False
    num_threads: int = 4
    batch_size: int = 32
    use_gpu: bool = False
    max_context_length: int = 100

@dataclass
class DataConfig:
    news_data_path: str = "data/news"
    stock_data_path: str = "data/stock"
    table_data_path: str = "data/tables"
    time_series_path: str = "data/timeseries"

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
    model_name: str = "Qwen/Qwen3-0.6B"
    # cache_dir: str = "M:/huggingface"
    cache_dir: str = "D:/AI/huggingface"

@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    data: DataConfig = field(default_factory=DataConfig)
    modality: ModalityConfig = field(default_factory=ModalityConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

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
                system=SystemConfig(
                    memory_limit=32,
                    log_level="WARNING"
                )
            )
        return cls()  # Default development config

# Default configuration instance
config = Config() 