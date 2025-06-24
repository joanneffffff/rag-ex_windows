"""
Reranker component using Qwen3-0.6B model for document reranking
Following official implementation guidelines
"""

from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

class QwenReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        device: str = None,
        cache_dir: str = None,
        use_quantization: bool = True,
        quantization_type: str = "8bit",  # "8bit" or "4bit"
        use_flash_attention: bool = False
    ):
        """
        初始化Qwen重排序器
        
        Args:
            model_name: 模型名称或路径
            device: 设备 (cpu/cuda)
            cache_dir: 模型缓存目录
            use_quantization: 是否使用量化
            quantization_type: 量化类型 ("8bit" 或 "4bit")
            use_flash_attention: 是否使用flash attention
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        print(f"\n加载重排序器模型: {model_name}")
        print(f"- 设备: {self.device}")
        print(f"- 缓存目录: {cache_dir}")
        print(f"- 量化: {use_quantization} ({quantization_type})")
        print(f"- Flash Attention: {use_flash_attention}")
        
        # 配置量化参数
        quantization_config = None
        if use_quantization:
            if quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side='left'
        )
        
        # 加载模型
        model_kwargs = {
            "torch_dtype": torch.float16,
            "cache_dir": cache_dir
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # 移动到设备
        if self.device == "cuda":
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        # 获取特殊token ID
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # 设置最大长度
        self.max_length = 8192
        
        # 设置提示模板（按照官方实现）
        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        # 预编码prefix和suffix tokens（按照官方实现）
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        
        print("重排序器模型加载完成")
    
    def format_instruction(self, instruction: str, query: str, document: str) -> str:
        """
        格式化指令（按照官方实现）
        
        Args:
            instruction: 指令文本
            query: 查询文本
            document: 文档文本
            
        Returns:
            格式化后的指令字符串
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    
    def process_inputs(self, pairs: List[str]) -> Dict[str, torch.Tensor]:
        """
        处理输入（按照官方实现）
        
        Args:
            pairs: 格式化后的指令字符串列表
            
        Returns:
            分词器输出字典
        """
        # 首先进行基础分词
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # 为每个输入添加prefix和suffix tokens（按照官方实现）
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        # 进行padding
        inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        # 移动到正确的设备
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        
        return inputs
    
    @torch.no_grad()
    def compute_logits(self, inputs: Dict[str, torch.Tensor]) -> List[float]:
        """
        计算logits（按照官方实现）
        
        Args:
            inputs: 分词器输出
            
        Returns:
            分数列表
        """
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 4
    ) -> List[Tuple[str, float]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            batch_size: 批处理大小
            
        Returns:
            重排序后的(文档, 分数)列表
        """
        if not documents:
            return []
        
        # 格式化所有文档
        formatted_pairs = []
        for doc in documents:
            formatted_text = self.format_instruction(None, query, doc)
            formatted_pairs.append((formatted_text, doc))
        
        # 批处理重排序
        all_scores = []
        for i in range(0, len(formatted_pairs), batch_size):
            batch_pairs = formatted_pairs[i:i + batch_size]
            batch_texts = [pair[0] for pair in batch_pairs]
            
            # 处理输入
            inputs = self.process_inputs(batch_texts)
            
            # 计算分数
            batch_scores = self.compute_logits(inputs)
            all_scores.extend(batch_scores)
        
        # 组合文档和分数
        results = []
        for i, (formatted_text, doc) in enumerate(formatted_pairs):
            results.append((doc, all_scores[i]))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def rerank_with_metadata(
        self,
        query: str,
        documents_with_metadata: List[Dict],
        batch_size: int = 4
    ) -> List[Dict]:
        """
        对带元数据的文档进行重排序
        
        Args:
            query: 查询文本
            documents_with_metadata: 带元数据的文档列表
            batch_size: 批处理大小
            
        Returns:
            重排序后的文档元数据列表
        """
        if not documents_with_metadata:
            return []
        
        # 提取文档文本
        documents = [doc.get('content', doc.get('text', '')) for doc in documents_with_metadata]
        
        # 进行重排序
        reranked_results = self.rerank(query, documents, batch_size)
        
        # 将分数添加回元数据
        results = []
        for i, (doc_text, score) in enumerate(reranked_results):
            # 找到对应的原始元数据
            original_metadata = documents_with_metadata[i]
            updated_metadata = original_metadata.copy()
            updated_metadata['reranker_score'] = score
            results.append(updated_metadata)
        
        return results 