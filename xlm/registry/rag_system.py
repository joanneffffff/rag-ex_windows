from xlm.components.generator.generator import Generator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.registry.retriever import load_bilingual_retriever
from xlm.utils.unified_data_loader import UnifiedDataLoader
from xlm.registry.generator import load_generator
from config.parameters import config # Use the global config instance


def load_bilingual_rag_system(
    use_gpu: bool = False,
    prompt_template: str = "Context: {context}\nQuestion: {question}\n\nAnswer:",
):
    """
    A self-contained function to load the complete bilingual RAG system.
    It handles data loading and generator initialization internally.
    """
    print(f"Initializing components for Bilingual RAG System (GPU enabled: {use_gpu})...")
    
    # 1. Load data using parameters from the global config
    data_loader = UnifiedDataLoader(
        data_dir=config.data.data_dir,
        cache_dir=config.cache_dir,
        max_samples=config.data.max_samples
    )

    # 2. Load generator
    generator = load_generator(
        generator_model_name=config.generator.model_name,
        use_local_llm=True,
        use_gpu=use_gpu,
        cache_dir=config.cache_dir
    )
    
    # 3. Load bilingual retriever, passing the use_gpu flag
    retriever = load_bilingual_retriever(
        data_loader=data_loader,
        use_faiss=True,
        use_gpu=use_gpu,
        cache_dir=config.cache_dir
    )
    
    # 4. Create and return the RAG system
    system = RagSystem(
        retriever=retriever,
        generator=generator,
        retriever_top_k=3,
    )
    return system
