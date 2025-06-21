from xlm.components.generator.llm_generator import LLMGenerator
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.registry import DEFAULT_LMS_ENDPOINT
from config.parameters import config


def load_generator(
    generator_model_name: str,
    split_lines: bool = False,
    use_local_llm: bool = False,
    use_gpu: bool = False,
    lms_endpoint: str = DEFAULT_LMS_ENDPOINT,
    cache_dir: str = None,
):
    """
    加载生成器
    Args:
        generator_model_name: 模型名称（支持如"facebook/opt-125m"、"llama2-7b-chat"、"qwen3-8b"等）
        split_lines: 是否按行分割
        use_local_llm: 是否使用本地LLM
        use_gpu: 是否使用GPU
        lms_endpoint: LMS服务端点
        cache_dir: 缓存目录
    Returns:
        Generator实例
    Note:
        现在支持qwen3-8b作为生成器模型，只需将generator_model_name设为"qwen3-8b"。
    """
    if use_local_llm:
        return LocalLLMGenerator(
            model_name=generator_model_name,
            device='cuda' if use_gpu else 'cpu',
            cache_dir=cache_dir or config.generator.cache_dir,
            temperature=0.1,  # 降低温度以获得更确定性的答案
            max_new_tokens=150,  # Increase token limit for more complete answers
            top_p=0.9
        )
    else:
        return LLMGenerator(
            model_name=generator_model_name,
            endpoint=lms_endpoint,
            split_lines=split_lines,
        )
