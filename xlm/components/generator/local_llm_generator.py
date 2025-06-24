import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from xlm.components.generator.generator import Generator
from config.parameters import Config


class LocalLLMGenerator(Generator):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        cache_dir: str = None,
        device: str = None,
        use_quantization: bool = True,
        quantization_type: str = "8bit",
        use_flash_attention: bool = False
    ):
        super().__init__(model_name=model_name)
        self.device = device
        self.temperature = 0.7
        self.max_new_tokens = 100
        self.top_p = 0.9
        
        # 使用config中的平台感知配置
        if cache_dir is None:
            config = Config()
            cache_dir = config.cache_dir
        
        self.cache_dir = cache_dir  # 关键修正，确保属性存在
        self.use_quantization = use_quantization
        self.quantization_type = quantization_type
        self.use_flash_attention = use_flash_attention
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置环境变量
        # os.environ['HF_HOME'] = 'D:/AI/huggingface'
        os.environ['HF_HOME'] = self.cache_dir
        # os.environ['TRANSFORMERS_CACHE'] = os.path.join('D:/AI/huggingface', 'transformers')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, 'transformers')
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        print(f"LocalLLMGenerator '{model_name}' loaded on {self.device}.")
        
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Prepare model loading arguments
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "low_cpu_mem_usage": True,
            "use_cache": True,
        }

        # Only apply quantization if on a CUDA GPU
        if self.device == 'cuda':
            print("CUDA device detected. Applying 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=False,
            )
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto" # Recommended for quantization
        else:
            print("CPU device detected. Loading model without quantization.")
            model_kwargs["device_map"] = "cpu"
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
    def generate(self, texts: List[str]) -> List[str]:
        responses = []
        for text in texts:
            # 优化：直接用tokenizer.__call__处理padding和truncation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length"
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate with increased length
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,  # Increased from a smaller default
                do_sample=True,
                top_p=0.9,
                temperature=0.6,
                pad_token_id=self.tokenizer.eos_token_id  # Use eos_token_id for padding
            )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response.strip())
            
        return responses 