from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
from xlm.modules.comparator.generic_comparator import (
    LevenshteinComparator,
    JaroWinklerComparator,
)
from xlm.modules.comparator.n_gram_overlap_comparator import NGramOverlapComparator
from xlm.modules.comparator.score_comaprator import ScoreComparator
from xlm.components.encoder.encoder import Encoder
from config.parameters import Config

# 初始化基本比较器
levenshtein_comparator = LevenshteinComparator()
jaro_winkler_comparator = JaroWinklerComparator()
n_gram_comparator = NGramOverlapComparator()
score_comparator = ScoreComparator()

# 初始化基于编码的比较器
sentence_transformers_based_comparator = EmbeddingComparator(
    encoder=Encoder(
        # model_name="all-MiniLM-L6-v2",
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        # cache_dir="D:/AI/huggingface",
        cache_dir=Config().cache_dir
    )
)

# 注册所有比较器
COMPARATORS = {
    "sentence_transformers_based_comparator": sentence_transformers_based_comparator,
    "levenshtein_comparator": levenshtein_comparator,
    "jaro_winkler_comparator": jaro_winkler_comparator,
    "n_gram_comparator": n_gram_comparator,
    "score_comparator": score_comparator,
}


def load_comparator(comparator_name: str, model_name: str = None):
    """
    加载比较器
    Args:
        comparator_name: 比较器名称
        model_name: 模型名称（用于基于LLM的比较器）
    Returns:
        比较器实例
    """
    if comparator_name == "base_llm_based_comparator":
        return EmbeddingComparator(
            encoder=Encoder(
                model_name=model_name,
                # cache_dir="D:/AI/huggingface",
                cache_dir=Config().cache_dir
            )
        )
    
    if comparator_name not in COMPARATORS:
        raise Exception(
            f"未找到指定的比较器！可用的比较器有: {list(COMPARATORS.keys())}"
        )
    
    return COMPARATORS.get(comparator_name)
