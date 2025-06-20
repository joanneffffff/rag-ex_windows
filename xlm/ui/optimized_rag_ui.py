import os
import gradio as gr
from typing import List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gradio.components import Markdown

from xlm.registry.generator import load_generator
from xlm.registry.retriever import load_retriever
from xlm.registry.rag_system import load_rag_system
from xlm.utils.visualizer import Visualizer
from xlm.dto.dto import DocumentWithMetadata
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig

class OptimizedRagUI:
    def __init__(
        self,
        # encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        # generator_model_name: str = "facebook/opt-125m",
        generator_model_name: str = "Qwen/Qwen3-8B",
        # cache_dir: str = "D:/AI/huggingface",
        cache_dir: str = "M:/huggingface",
        data_path: str = "data/rise_of_ai.txt",
        use_faiss: bool = True,
        window_title: str = "RAG System with FAISS",
        title: str = "RAG System with FAISS",
        examples: Optional[List[List[str]]] = None,
    ):
        self.encoder_model_name = encoder_model_name
        self.generator_model_name = generator_model_name
        self.cache_dir = cache_dir
        self.data_path = data_path
        self.use_faiss = use_faiss
        self.window_title = window_title
        self.title = title
        self.examples = examples or [
            ["What are the key challenges in AI development?"],
            ["How does machine learning impact business?"],
            ["What are the ethical concerns in AI?"],
            ["Explain the concept of deep learning."]
        ]
        
        # Set environment variables for model caching
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
        os.environ['HF_HOME'] = cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
        
        # Initialize system components
        self._init_components()
        
        # Create Gradio interface
        self.interface = self._create_interface()
    
    def _init_components(self):
        """Initialize RAG system components"""
        print("\n1. Loading retriever...")
        from xlm.components.encoder.multimodal_encoder import MultiModalEncoder
        
        # Initialize with enhanced encoder configuration
        config = Config(
            encoder=EncoderConfig(
                model_name=self.encoder_model_name,
                device="cpu",
                batch_size=8
            ),
            retriever=RetrieverConfig(num_threads=2),
            modality=ModalityConfig(
                combine_method="weighted_sum",
                text_weight=0.4,
                table_weight=0.3,
                time_series_weight=0.3
            )
        )
        
        encoder = MultiModalEncoder(
            config=config,
            use_enhanced_encoders=True  # Enable enhanced encoder by default
        )
        
        self.retriever = load_retriever(
            encoder_model_name=self.encoder_model_name,
            data_path=self.data_path,
            encoder=encoder  # Pass the enhanced encoder
        )
        
        if self.use_faiss:
            print("Initializing FAISS index...")
            self._init_faiss()
        
        print("\n2. Loading generator...")
        self.generator = load_generator(
            generator_model_name=self.generator_model_name,
            use_local_llm=True
        )
        
        print("\n3. Initializing RAG system...")
        self.prompt_template = "Context: {context}\nQuestion: {question}\nAnswer:"
        self.rag_system = load_rag_system(
            retriever=self.retriever,
            generator=self.generator,
            prompt_template=self.prompt_template
        )
        
        print("\n4. Loading visualizer...")
        self.visualizer = Visualizer(show_mid_features=True)
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        corpus_embeddings = self.retriever.corpus_embeddings
        self.dimension = len(corpus_embeddings[0])
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(corpus_embeddings).astype('float32'))
    
    def _create_interface(self) -> gr.Blocks:
        """Create optimized Gradio interface"""
        with gr.Blocks(
            theme=gr.themes.Monochrome().set(
                button_primary_background_fill="#009374",
                button_primary_background_fill_hover="#009374C4",
                checkbox_label_background_fill_selected="#028A6EFF",
            ),
            title=self.window_title
        ) as interface:
            # Title
            with gr.Row():
                with gr.Column(scale=1):
                    Markdown(
                        f'<p style="text-align: center; font-size:200%; font-weight: bold">'
                        f"{self.title}"
                        f"</p>"
                    )
            
            # Input area
            with gr.Row():
                with gr.Column(scale=1):
                    question_input = gr.Textbox(
                        placeholder="Type your question here and press Enter.",
                        label="Question",
                        lines=3
                    )
            
            # Control buttons
            with gr.Row():
                submit_btn = gr.Button(
                    value="🔍 Ask",
                    variant="secondary",
                    elem_id="button"
                )
            
            # Output area with tabs
            with gr.Row():
                with gr.Tabs():
                    with gr.TabItem("Answer"):
                        answer_output = gr.Textbox(
                            label="Generated Response",
                            lines=5,
                            interactive=False
                        )
                    
                    with gr.TabItem("Retrieved Context"):
                        context_output = gr.Dataframe(
                            headers=["Relevance", "Content"],
                            label="Retrieved Documents"
                        )
                    
                    with gr.TabItem("Visualization"):
                        visualization_output = gr.Plot(
                            label="Retrieval Scores"
                        )
            
            # Example questions
            gr.Examples(
                examples=self.examples,
                inputs=[question_input]
            )
            
            # Event handlers
            submit_btn.click(
                fn=self._process_question,
                inputs=[question_input],
                outputs=[answer_output, context_output, visualization_output]
            )
        
        return interface
    
    def _process_question(
        self,
        question: str
    ) -> tuple[str, List[List[str]], Optional[gr.Plot]]:
        """Process user question and return results"""
        if not question or question.strip() == "":
            return "Please enter a question", [], None
        
        # Run RAG system
        rag_output = self.rag_system.run(user_input=question)
        
        # Prepare answer
        answer = rag_output.generated_responses[0] if rag_output.generated_responses else "Unable to generate answer"
        
        # Prepare context data
        context_data = []
        for doc, score in zip(rag_output.retrieved_documents, rag_output.retriever_scores):
            context_data.append([f"{score:.4f}", doc.content])
        
        # Generate visualization
        try:
            plot = self.visualizer.plot_retrieval_scores(
                documents=[doc.content for doc in rag_output.retrieved_documents],
                scores=rag_output.retriever_scores
            )
        except Exception:
            plot = None
        
        return answer, context_data, plot
    
    def launch(self, share: bool = False):
        """Launch UI interface"""
        self.interface.launch(share=share) 