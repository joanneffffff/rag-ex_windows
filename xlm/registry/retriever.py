from xlm.components.retriever.bilingual_retriever import BilingualRetriever
from xlm.components.encoder.finbert import FinbertEncoder
from xlm.utils.unified_data_loader import UnifiedDataLoader

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
