import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xlm.components.generator.generator import Generator


class LocalLLMGenerator(Generator):
    def __init__(
        self,
        model_name: str = "facebook/opt-125m",
        device: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
        top_p: float = 0.9,
        # cache_dir: str = "D:/AI/huggingface"
        cache_dir: str = "M:/huggingface"
    ):
        super().__init__(model_name=model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.cache_dir = cache_dir
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置环境变量
        # os.environ['HF_HOME'] = 'D:/AI/huggingface'
        os.environ['HF_HOME'] = self.cache_dir
        # os.environ['TRANSFORMERS_CACHE'] = os.path.join('D:/AI/huggingface', 'transformers')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, 'transformers')
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """加载模型和tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
    def generate(self, texts: List[str]) -> List[str]:
        """
        生成回答
        Args:
            texts: 输入提示列表
        Returns:
            生成的回答列表
        """
        responses = []
        
        for prompt in texts:
            # 准备输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # 记录输入长度
            input_length = inputs["input_ids"].shape[1]
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    num_beams=4,  # 使用beam search
                    no_repeat_ngram_size=3,  # 避免重复
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=False  # 使用确定性生成
                )
            
            # 只解码新生成的token
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # 如果响应为空或不完整，尝试重新生成
            if not response or len(response) < 5:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens * 2,  # 增加生成长度
                    temperature=self.temperature * 2,  # 增加随机性
                    top_p=self.top_p,
                    num_beams=1,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    do_sample=True
                )
                new_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            responses.append(response)
            
        return responses 