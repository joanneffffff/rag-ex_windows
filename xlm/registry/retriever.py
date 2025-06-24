from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.retriever.enhanced_retriever import EnhancedRetriever
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.unified_data_loader import UnifiedDataLoader
from xlm.utils.dual_language_loader import DualLanguageLoader
from config.parameters import Config

def load_bilingual_retriever(
    data_loader: UnifiedDataLoader,
    use_faiss: bool = True,
    use_gpu: bool = False,
    batch_size: int = 32,
    cache_dir: str = None,
):
    """
    Loads the bilingual retriever.
    """
    print("Loading English encoder (ProsusAI/finbert)...")
    encoder_en = FinbertEncoder(
        model_name="ProsusAI/finbert",
        cache_dir=cache_dir,
    )

    print("Loading Chinese encoder (Langboat/mengzi-bert-base-fin)...")
    encoder_ch = FinbertEncoder(
        model_name="Langboat/mengzi-bert-base-fin",
        cache_dir=cache_dir,
    )

    retriever = BilingualRetriever(
        encoder_en=encoder_en,
        encoder_ch=encoder_ch,
        corpus_documents_en=data_loader.english_docs,
        corpus_documents_ch=data_loader.chinese_docs,
        use_faiss=use_faiss,
        use_gpu=use_gpu,
        batch_size=batch_size,
    )

    return retriever

def load_enhanced_retriever(
    config: Config,
    chinese_data_path: str = None,
    english_data_path: str = None,
    jsonl_data_path: str = None
):
    """
    Loads the enhanced retriever with dual embedding spaces and Qwen reranker.
    
    Args:
        config: Configuration object
        chinese_data_path: Path to Chinese data file
        english_data_path: Path to English data file
        jsonl_data_path: Path to JSONL data file
        
    Returns:
        EnhancedRetriever instance
    """
    print("Loading enhanced retriever with dual embedding spaces...")
    
    # 加载双语言数据
    data_loader = DualLanguageLoader()
    chinese_docs, english_docs = data_loader.load_dual_language_data(
        chinese_data_path=chinese_data_path,
        english_data_path=english_data_path,
        jsonl_data_path=jsonl_data_path
    )
    
    # 创建增强检索器
    retriever = EnhancedRetriever(
        config=config,
        chinese_documents=chinese_docs,
        english_documents=english_docs
    )
    
    print("Enhanced retriever loaded successfully!")
    return retriever
