import sys

from xlm.registry.generator import load_generator
from xlm.registry.rag_system import load_rag_system
from xlm.registry.retriever import load_retriever
from xlm.ui.rag_explainer_ui import RagExplainerUI
from xlm.utils.visualizer import Visualizer


def load_visualizer() -> Visualizer:
    visualizer = Visualizer(show_mid_features=True, show_low_features=True)
    return visualizer


if __name__ == "__main__":
    # interface = ExplainerUI(
    #     logo_path="xlm/ui/images/iais.svg",
    #     css_path="xlm/ui/css/demo.css",
    #     visualizer=load_visualizer(),
    #     window_title="RAG-Ex",
    #     title="✳️ RAG-Ex: Towards Generic Explainability",
    # )
    encoder_model_name = "sentence-transformers"
    generator_model_name = "facebook/opt-125m"
    data_path = "data/rise_of_ai.txt"
    prompt_template = "Context: {context}\nQuestion: {question}\n\nAnswer:"

    retriever = load_retriever(
        encoder_model_name=encoder_model_name,
        lms_endpoint="http://localhost:9985",
        data_path=data_path,
    )
    generator = load_generator(
        generator_model_name=generator_model_name,
        split_lines=False,
        use_local_llm=True
    )
    # --- To use Qwen3-8B as the generator model, uncomment below ---
    # generator = load_generator(
    #     generator_model_name="Qwen/Qwen3-8B",
    #     split_lines=False,
    #     use_local_llm=True
    # )
    # --- To use Fin-R1 as the generator model, uncomment below ---
    # generator = load_generator(
    #     generator_model_name="SUFE-AIFLM-Lab/Fin-R1",
    #     split_lines=False,
    #     use_local_llm=True
    # )
    rag_system = load_rag_system(
        retriever=retriever, generator=generator, prompt_template=prompt_template
    )

    interface = RagExplainerUI(
        logo_path="xlm/ui/images/iais.svg",
        css_path="xlm/ui/css/demo.css",
        visualizer=load_visualizer(),
        window_title="RAG-Ex 2.0 (Local LLM)",
        title="✳️ RAG-Ex 2.0: Towards Generic Explainability (Local LLM)",
        rag_system=rag_system,
    )
    app = interface.build_app()
    app.queue()

    platform = sys.platform
    print("Platform: ", platform)
    host = "127.0.0.1" if "win" in platform else "0.0.0.0"
    port = 8023
    app.launch(server_name=host, server_port=port)
