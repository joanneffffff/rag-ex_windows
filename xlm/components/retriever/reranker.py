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
        
        # 移动到设备 - 量化模型不需要手动移动
        if quantization_config:
            # 量化模型已经自动设置到正确设备，不需要手动移动
            print("量化模型已自动设置到设备，跳过手动移动")
        else:
            # 非量化模型需要手动移动到设备
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
        处理输入（优化版本，直接使用tokenizer.__call__方法）
        
        Args:
            pairs: 格式化后的指令字符串列表
            
        Returns:
            分词器输出字典
        """
        # 为每个输入添加prefix和suffix tokens
        processed_pairs = []
        for pair in pairs:
            # 预编码prefix和suffix tokens
            prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
            suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
            
            # 编码主要内容
            content_tokens = self.tokenizer.encode(pair, add_special_tokens=False, 
                                                  max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens),
                                                  truncation=True)
            
            # 组合所有tokens
            full_tokens = prefix_tokens + content_tokens + suffix_tokens
            processed_pairs.append(full_tokens)
        
        # 直接使用tokenizer.__call__方法进行padding（更高效）
        inputs = self.tokenizer(
            processed_pairs,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
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
        batch_size: int = 2  # 减少批处理大小以节省内存
    ) -> List[Tuple[str, float]]:
        """
        对文档进行重排序（优化内存使用）
        
        Args:
            query: 查询文本
            documents: 文档列表
            batch_size: 批处理大小（默认2以减少内存使用）
            
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
        
        # 批处理重排序（优化内存使用）
        all_scores = []
        for i in range(0, len(formatted_pairs), batch_size):
            batch_pairs = formatted_pairs[i:i + batch_size]
            batch_texts = [pair[0] for pair in batch_pairs]
            
            try:
                # 处理输入
                inputs = self.process_inputs(batch_texts)
                
                # 计算分数
                batch_scores = self.compute_logits(inputs)
                all_scores.extend(batch_scores)
                
                # 清理GPU内存
                del inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU内存不足，尝试减小批处理大小...")
                    # 如果内存不足，尝试更小的批处理
                    if batch_size > 1:
                        # 递归调用，使用更小的批处理大小
                        return self.rerank(query, documents, batch_size=batch_size // 2)
                    else:
                        print("批处理大小已最小化，仍内存不足，尝试CPU处理...")
                        # 最后尝试CPU处理
                        return self._rerank_on_cpu(query, documents)
                else:
                    raise e
        
        # 组合文档和分数
        results = []
        for i, (formatted_text, doc) in enumerate(formatted_pairs):
            results.append((doc, all_scores[i]))
        
        # 按分数降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _rerank_on_cpu(self, query: str, documents: List[str]) -> List[Tuple[str, float]]:
        """
        CPU回退重排序（当GPU内存不足时使用）
        
        Args:
            query: 查询文本
            documents: 文档列表
            
        Returns:
            重排序后的(文档, 分数)列表
        """
        print("使用CPU进行重排序...")
        
        # 临时将模型移动到CPU
        original_device = next(self.model.parameters()).device
        self.model = self.model.cpu()
        
        try:
            # 格式化所有文档
            formatted_pairs = []
            for doc in documents:
                formatted_text = self.format_instruction(None, query, doc)
                formatted_pairs.append((formatted_text, doc))
            
            # 逐个处理（CPU模式）
            all_scores = []
            for formatted_text, doc in formatted_pairs:
                inputs = self.process_inputs([formatted_text])
                score = self.compute_logits(inputs)[0]
                all_scores.append(score)
                del inputs
            
            # 组合文档和分数
            results = []
            for i, (formatted_text, doc) in enumerate(formatted_pairs):
                results.append((doc, all_scores[i]))
            
            # 按分数降序排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        finally:
            # 恢复模型到原始设备
            self.model = self.model.to(original_device)
    
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