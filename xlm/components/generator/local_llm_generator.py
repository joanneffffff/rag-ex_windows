import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

from xlm.components.generator.generator import Generator
from config.parameters import Config


class LocalLLMGenerator(Generator):
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_quantization: Optional[bool] = None,
        quantization_type: Optional[str] = None,
        use_flash_attention: bool = False
    ):
        # 使用config中的平台感知配置
        config = Config()
        
        # 如果没有提供model_name，从config读取
        if model_name is None:
            model_name = config.generator.model_name
        
        # 如果没有提供量化参数，从config读取
        if use_quantization is None:
            use_quantization = config.generator.use_quantization
        if quantization_type is None:
            quantization_type = config.generator.quantization_type
        
        super().__init__(model_name=model_name)
        self.device = device
        self.temperature = config.generator.temperature
        self.max_new_tokens = config.generator.max_new_tokens
        self.top_p = config.generator.top_p
        
        # 使用config中的平台感知配置
        if cache_dir is None:
            cache_dir = config.generator.cache_dir  # 使用generator的缓存目录
        
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
        print(f"LocalLLMGenerator '{model_name}' loaded on {self.device} with quantization: {self.use_quantization} ({self.quantization_type}).")
        
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

        # 根据配置应用量化
        if self.use_quantization and self.device and self.device.startswith('cuda'):
            print(f"CUDA device detected. Applying {self.quantization_type} quantization...")
            
            if self.quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
            elif self.quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                print(f"Unknown quantization type: {self.quantization_type}, falling back to 4bit")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=False,
                )
            
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = self.device  # 明确指定设备
        else:
            if self.device and self.device.startswith('cuda'):
                print("CUDA device detected but quantization disabled. Loading model without quantization.")
                model_kwargs["device_map"] = self.device  # 明确指定设备
                model_kwargs["torch_dtype"] = torch.float16
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
            # 调试信息：打印设备状态
            print(f"Generator device: {self.device}")
            print(f"Model device: {next(self.model.parameters()).device if hasattr(self.model, 'parameters') else 'Unknown'}")
            
            # 确保tokenizer在正确的设备上
            if hasattr(self.tokenizer, 'device') and self.tokenizer.device != self.device:
                print(f"Warning: Tokenizer device mismatch. Moving to {self.device}")
            
            # 优化：直接用tokenizer.__call__处理padding和truncation
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding="max_length"
            )
            
            # 确保所有输入都在正确的设备上
            model_device = next(self.model.parameters()).device
            print(f"Model device: {model_device}")
            
            if model_device.type == 'cuda':
                input_ids = inputs["input_ids"].to(model_device)
                attention_mask = inputs["attention_mask"].to(model_device)
                print(f"Input tensors moved to: {input_ids.device}")
            else:
                input_ids = inputs["input_ids"].cpu()
                attention_mask = inputs["attention_mask"].cpu()
                print(f"Input tensors moved to: {input_ids.device}")
            
            # Generate with increased length
            with torch.no_grad():  # 添加no_grad来节省内存
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,  # 使用配置文件中的值
                    do_sample=True,
                    top_p=self.top_p,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,  # Use eos_token_id for padding
                    repetition_penalty=1.1,  # 添加重复惩罚
                    length_penalty=1.0  # 添加长度惩罚
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(response.strip())
            
        return responses 