from xlm.components.generator.llm_generator import LLMGenerator
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.registry import DEFAULT_LMS_ENDPOINT
from config.parameters import config, DEFAULT_CACHE_DIR
from typing import Optional


def load_generator(
    generator_model_name: str,
    split_lines: bool = False,
    use_local_llm: bool = False,
    use_gpu: bool = False,
    gpu_device: str = "cuda:1",  # 默认使用GPU 1
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
    cache_dir: Optional[str] = None,
):
    """
    加载生成器
    Args:
        generator_model_name: 模型名称（支持如"facebook/opt-125m"、"llama2-7b-chat"、"qwen3-8b"等）
        split_lines: 是否按行分割
        use_local_llm: 是否使用本地LLM
        use_gpu: 是否使用GPU
        gpu_device: GPU设备名称（如"cuda:0", "cuda:1"）
        lms_endpoint: LMS服务端点
        cache_dir: 缓存目录
    Returns:
        Generator实例
    Note:
        现在支持qwen3-8b作为生成器模型，只需将generator_model_name设为"qwen3-8b"。
    """
    if use_local_llm:
        import torch
        
        # 检查GPU可用性和内存
        if use_gpu and torch.cuda.is_available():
            try:
                # 解析GPU设备ID
                if ":" in gpu_device:
                    gpu_id = int(gpu_device.split(":")[1])
                else:
                    gpu_id = 0
                
                # 检查GPU数量
                if gpu_id < torch.cuda.device_count():
                    # 检查GPU内存
                    gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory
                    allocated_memory = torch.cuda.memory_allocated(gpu_id)
                    free_memory = gpu_memory - allocated_memory
                    
                    print(f"GPU {gpu_id} 总内存: {gpu_memory / 1024**3:.1f}GB")
                    print(f"GPU {gpu_id} 已用内存: {allocated_memory / 1024**3:.1f}GB")
                    print(f"GPU {gpu_id} 可用内存: {free_memory / 1024**3:.1f}GB")
                    
                    # 如果可用内存少于4GB，回退到CPU
                    if free_memory < 4 * 1024**3:  # 4GB
                        print(f"GPU {gpu_id} 内存不足，回退到CPU")
                        device = 'cpu'
                        use_quantization = False
                    else:
                        device = gpu_device
                        use_quantization = True
                else:
                    print(f"GPU {gpu_id} 不存在，回退到CPU")
                    device = 'cpu'
                    use_quantization = False
            except Exception as e:
                print(f"GPU检查失败: {e}，回退到CPU")
                device = 'cpu'
                use_quantization = False
        else:
            device = 'cpu'
            use_quantization = False
        
        print(f"生成器设备: {device}")
        print(f"量化: {use_quantization}")
        
        # 确保cache_dir有有效值
        final_cache_dir = cache_dir or config.generator.cache_dir or DEFAULT_CACHE_DIR
        
        return LocalLLMGenerator(
            model_name=generator_model_name,
            device=device,
            cache_dir=final_cache_dir,
            use_quantization=use_quantization,
            quantization_type="8bit",
            use_flash_attention=False
        )
    else:
        return LLMGenerator(
            model_name=generator_model_name,
            endpoint=lms_endpoint,
            split_lines=split_lines,
        )
