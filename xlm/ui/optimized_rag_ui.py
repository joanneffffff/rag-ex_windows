import os
import gradio as gr
from typing import List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from gradio.components import Markdown
import torch
from langdetect import detect, LangDetectException
import re

from xlm.registry.generator import load_generator
from xlm.registry.retriever import load_enhanced_retriever
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.reranker import QwenReranker
from xlm.utils.visualizer import Visualizer
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig, EMBEDDING_CACHE_DIR, RERANKER_CACHE_DIR

def try_load_qwen_reranker(model_name, cache_dir=None):
    """尝试加载Qwen重排序器，支持GPU 0和CPU回退"""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        
        # 确保cache_dir是有效的字符串
        if cache_dir is None:
            cache_dir = RERANKER_CACHE_DIR
        
        print(f"尝试使用8bit量化加载QwenReranker...")
        print(f"加载重排序器模型: {model_name}")
        
        # 首先尝试GPU 0
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            device = "cuda:0"  # 明确指定GPU 0
            print(f"- 设备: {device}")
            print(f"- 缓存目录: {cache_dir}")
            print(f"- 量化: True (8bit)")
            print(f"- Flash Attention: False")
            
            try:
                # 检查GPU 0的可用内存
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                allocated_memory = torch.cuda.memory_allocated(0)
                free_memory = gpu_memory - allocated_memory
                
                print(f"- GPU 0 总内存: {gpu_memory / 1024**3:.1f}GB")
                print(f"- GPU 0 已用内存: {allocated_memory / 1024**3:.1f}GB")
                print(f"- GPU 0 可用内存: {free_memory / 1024**3:.1f}GB")
                
                # 如果可用内存少于2GB，回退到CPU
                if free_memory < 2 * 1024**3:  # 2GB
                    print("- GPU 0 内存不足，回退到CPU")
                    device = "cpu"
                else:
                    # 尝试在GPU 0上加载
                    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        load_in_8bit=True
                    )
                    print("量化模型已自动设置到设备，跳过手动移动")
                    print("重排序器模型加载完成")
                    print("量化加载成功！")
                    return QwenReranker(model_name, device=device, cache_dir=cache_dir)
                    
            except Exception as e:
                print(f"- GPU 0 加载失败: {e}")
                print("- 回退到CPU")
                device = "cpu"
        
        # CPU回退
        if device == "cpu" or not torch.cuda.is_available():
            device = "cpu"
            print(f"- 设备: {device}")
            print(f"- 缓存目录: {cache_dir}")
            print(f"- 量化: False (CPU模式)")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                torch_dtype=torch.float32
            )
            model = model.to(device)
            print("重排序器模型加载完成")
            print("CPU加载成功！")
            return QwenReranker(model_name, device=device, cache_dir=cache_dir)
            
    except Exception as e:
        print(f"加载重排序器失败: {e}")
        return None

class OptimizedRagUI:
    def __init__(
        self,
        # encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encoder_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        # generator_model_name: str = "facebook/opt-125m",
        # generator_model_name: str = "Qwen/Qwen2-1.5B-Instruct",
        # generator_model_name: str = "SUFE-AIFLM-Lab/Fin-R1",  # 使用金融专用Fin-R1模型
        cache_dir: Optional[str] = None,
        use_faiss: bool = True,
        enable_reranker: bool = True,
        use_existing_embedding_index: Optional[bool] = None,  # 从config读取，None表示使用默认值
        max_alphafin_chunks: Optional[int] = None,  # 从config读取，None表示使用默认值
        window_title: str = "RAG System with FAISS",
        title: str = "RAG System with FAISS",
        examples: Optional[List[List[str]]] = None,
    ):
        # 使用config中的平台感知配置
        config = Config()
        self.cache_dir = EMBEDDING_CACHE_DIR if (not cache_dir or not isinstance(cache_dir, str)) else cache_dir
        self.encoder_model_name = encoder_model_name
        # 从config读取生成器模型名称，而不是硬编码
        self.generator_model_name = config.generator.model_name
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        # 从config读取参数，如果传入None则使用config默认值
        self.use_existing_embedding_index = use_existing_embedding_index if use_existing_embedding_index is not None else config.retriever.use_existing_embedding_index
        self.max_alphafin_chunks = max_alphafin_chunks if max_alphafin_chunks is not None else config.retriever.max_alphafin_chunks
        self.window_title = window_title
        self.title = title
        self.examples = examples or [
            ["什么是股票投资？"],
            ["请解释债券的基本概念"],
            ["基金投资与股票投资有什么区别？"],
            ["What is stock investment?"],
            ["Explain the basic concepts of bonds"],
            ["What are the differences between fund investment and stock investment?"]
        ]
        
        # Set environment variables for model caching
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(self.cache_dir, 'transformers')
        os.environ['HF_HOME'] = self.cache_dir
        os.environ['HF_DATASETS_CACHE'] = os.path.join(self.cache_dir, 'datasets')
        
        # Initialize system components
        self._init_components()
        
        # Create Gradio interface
        self.interface = self._create_interface()
    
    def _init_components(self):
        """Initialize RAG system components"""
        print("\nStep 1. Loading bilingual retriever with dual encoders...")
        
        # 使用config中的平台感知配置
        config = Config()
        
        # 使用优化的数据加载器，实现文档级别chunking
        print("\nStep 1.1. Loading data with optimized chunking...")
        try:
            from xlm.utils.optimized_data_loader import OptimizedDataLoader
            
            # 使用文档级别chunking处理中文数据
            data_loader = OptimizedDataLoader(
                data_dir="data",
                max_samples=config.data.max_samples,
                chinese_document_level=True,  # 中文使用文档级别
                english_chunk_level=True      # 英文保持chunk级别
            )
            
            # 获取处理后的文档
            chinese_chunks = data_loader.chinese_docs
            english_chunks = data_loader.english_docs
            
            # 显示统计信息
            stats = data_loader.get_statistics()
            print(f"✅ 文档级别chunking完成:")
            print(f"   中文文档数: {stats['chinese_docs']}")
            print(f"   英文文档数: {stats['english_docs']}")
            print(f"   中文平均长度: {stats['chinese_avg_length']:.2f}")
            print(f"   英文平均长度: {stats['english_avg_length']:.2f}")
            
        except Exception as e:
            print(f"❌ 优化数据加载器失败: {e}")
            print("回退到传统数据加载方式...")
            
            # 回退到传统方式
            from xlm.utils.dual_language_loader import DualLanguageLoader
            
            data_loader = DualLanguageLoader()
            chinese_docs, english_docs = data_loader.load_dual_language_data(
                chinese_data_path=config.data.chinese_data_path,
                english_data_path=config.data.english_data_path
            )
            
            print(f"Loaded {len(chinese_docs)} Chinese documents")
            print(f"Loaded {len(english_docs)} English documents")
            
            # 使用传统chunking
            print("\nStep 1.1. Applying traditional document chunking...")
            chinese_chunks = self._chunk_documents_advanced(chinese_docs)
            english_chunks = self._chunk_documents_simple(english_docs, chunk_size=512, overlap=50)
            
            # 限制AlphaFin数据chunk数量，避免200k+ chunks影响测试
            if len(chinese_chunks) > self.max_alphafin_chunks:
                print(f"Limiting Chinese chunks from {len(chinese_chunks)} to {self.max_alphafin_chunks} for testing...")
                chinese_chunks = chinese_chunks[:self.max_alphafin_chunks]
        
        print(f"Final chunk count: {len(chinese_chunks)} Chinese chunks, {len(english_chunks)} English chunks")
        
        # 直接创建BilingualRetriever，embedding基于chunk
        from xlm.components.retriever.bilingual_retriever import BilingualRetriever
        from xlm.components.encoder.finbert import FinbertEncoder
        
        print("\nStep 2. Loading Chinese encoder (models/finetuned_alphafin_zh)...")
        encoder_ch = FinbertEncoder(
            model_name="models/finetuned_alphafin_zh",
            cache_dir=config.encoder.cache_dir,  # 使用encoder的缓存目录
        )
        print("Step 3. Loading English encoder (models/finetuned_finbert_tatqa)...")
        encoder_en = FinbertEncoder(
            model_name="models/finetuned_finbert_tatqa",
            cache_dir=config.encoder.cache_dir,  # 使用encoder的缓存目录
        )
        
        # 根据use_existing_embedding_index参数决定是否使用现有索引
        if self.use_existing_embedding_index:
            print("Using existing embedding index (if available)...")
            cache_dir = config.encoder.cache_dir  # 使用encoder的缓存目录
        else:
            print("Forcing to recompute embeddings (ignoring existing cache)...")
            import tempfile
            cache_dir = tempfile.mkdtemp(prefix="rag_temp_cache_")
            print(f"Using temporary cache directory: {cache_dir}")
        
        self.retriever = BilingualRetriever(
            encoder_en=encoder_en,
            encoder_ch=encoder_ch,
            corpus_documents_en=english_chunks,
            corpus_documents_ch=chinese_chunks,
            use_faiss=self.use_faiss,
            use_gpu=False,
            batch_size=8,
            cache_dir=cache_dir
        )
        
        if self.use_faiss:
            print("Step 3.1. Initializing FAISS index...")
            self._init_faiss()
            print("FAISS index initialized successfully")
        
        # Initialize reranker if enabled
        if self.enable_reranker:
            print("\nStep 4. Loading reranker...")
            self.reranker = try_load_qwen_reranker(
                model_name="Qwen/Qwen3-Reranker-0.6B",
                cache_dir=config.reranker.cache_dir  # 使用reranker的缓存目录
            )
        else:
            self.reranker = None
        
        print("\nStep 5. Loading generator...")
        self.generator = load_generator(
            generator_model_name=self.generator_model_name,
            use_local_llm=True,
            use_gpu=True,  # 启用GPU
            gpu_device="cuda:1",  # 使用GPU 1
            cache_dir=config.generator.cache_dir  # 使用generator的缓存目录
        )
        
        print("\nStep 6. Initializing RAG system...")
        
        self.rag_system = RagSystem(
            retriever=self.retriever,
            generator=self.generator,
            retriever_top_k=20  # 增加到20，从config读取retriever_top_k
        )
        
        print("\nStep 7. Loading visualizer...")
        self.visualizer = Visualizer(show_mid_features=True)
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        # 对于双空间检索器，我们需要合并英文和中文嵌入向量
        if hasattr(self.retriever, 'corpus_embeddings_en') and hasattr(self.retriever, 'corpus_embeddings_ch'):
            # 双空间检索器
            embeddings_en = self.retriever.corpus_embeddings_en
            embeddings_ch = self.retriever.corpus_embeddings_ch
            
            if embeddings_en is not None and embeddings_ch is not None:
                # 合并嵌入向量
                all_embeddings = np.vstack([embeddings_en, embeddings_ch])
                self.dimension = all_embeddings.shape[1]
                self.index = faiss.IndexFlatL2(self.dimension)
                # 确保数据类型正确
                embeddings_float32 = all_embeddings.astype('float32')
                self.index.add(embeddings_float32)
                print(f"Step 3.1. Initializing FAISS index with {all_embeddings.shape[0]} embeddings...")
            else:
                print("Warning: No embeddings available for FAISS initialization")
        else:
            # 单空间检索器 - 对于BilingualRetriever，跳过FAISS初始化
            print("Warning: BilingualRetriever detected, skipping FAISS initialization")
            print("FAISS index will not be available for this retriever type")
    
    def _create_interface(self) -> gr.Blocks:
        """Create optimized Gradio interface"""
        with gr.Blocks(
            title=self.window_title
        ) as interface:
            # 标题
            gr.Markdown(f"# {self.title}")
            
            # 输入区域
            with gr.Row():
                with gr.Column(scale=4):
                    datasource = gr.Radio(
                        choices=["TatQA", "AlphaFin", "Both"],
                        value="Both",
                        label="Data Source"
                    )
                    
            with gr.Row():
                with gr.Column(scale=4):
                    question_input = gr.Textbox(
                        show_label=False,
                        placeholder="Enter your question",
                        label="Question",
                        lines=3
                    )
            
            # 控制按钮区域
            with gr.Row():
                with gr.Column(scale=1):
                    reranker_checkbox = gr.Checkbox(
                        label="Enable Reranker",
                        value=True,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Submit")
            
            # 使用标签页分离显示
            with gr.Tabs():
                # 回答标签页
                with gr.TabItem("Answer"):
                    answer_output = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        label="Generated Response",
                        lines=5
                    )
                
                # 解释标签页
                with gr.TabItem("Explanation"):
                    context_output = gr.Dataframe(
                        headers=["Score", "Context"],
                        datatype=["number", "str"],
                        label="Retrieved Contexts",
                        interactive=False
                    )

            # 添加示例问题
            gr.Examples(
                examples=self.examples,
                inputs=[question_input],
                label="Example Questions"
            )

            # 绑定事件
            submit_btn.click(
                self._process_question,
                inputs=[question_input, datasource, reranker_checkbox],
                outputs=[answer_output, context_output]
            )
            
            return interface
    
    def _process_question(
        self,
        question: str,
        datasource: str,
        reranker_checkbox: bool
    ) -> tuple[str, List[List[str]], Optional[gr.Plot]]:
        """Process user question and return results"""
        if not question.strip():
            return "Please enter a question.", [], None
        print(f"\nProcessing question: {question}")
        print(f"Data source: {datasource}")
        
        # Detect language and pass to rag_system
        try:
            from langdetect import detect
            lang = detect(question)
            # 中文字符兜底
            def is_chinese(text):
                return len([c for c in text if '\u4e00' <= c <= '\u9fff']) > max(1, len(text) // 6)
            if lang.startswith('zh') or (lang == 'ko' and is_chinese(question)) or is_chinese(question):
                language = 'zh'
            else:
                language = 'en'
        except Exception as e:
            print(f"Language detection failed: {e}, fallback to English.")
            language = 'en'
        print(f"Detected language: {language}")
        
        # 根据数据源选择决定是否使用重排序器
        use_reranker = reranker_checkbox and self.enable_reranker and self.reranker is not None
        
        try:
            rag_output = self.rag_system.run(user_input=question, language=language)
            
            # 检测问题语言并打印数据源信息
            def is_chinese(text):
                return len(re.findall(r'[\u4e00-\u9fff]', text)) > max(1, len(text) // 6)
            try:
                lang = detect(question)
                # 修正：如果检测为韩文但内容明显是中文，强制设为中文
                if lang == 'ko' and is_chinese(question):
                    lang = 'zh-cn'
                is_chinese_q = lang.startswith('zh')
                detected_data_source = "AlphaFin" if is_chinese_q else "TAT_QA"
                print(f"Detected language: {lang} -> Using data source: {detected_data_source}")
            except LangDetectException:
                print("Language detection failed, defaulting to English -> TAT_QA")
                detected_data_source = "TAT_QA"
            
            # Apply reranker if enabled
            if use_reranker and rag_output.retrieved_documents and self.reranker is not None:
                print("Applying reranker...")
                # Prepare documents for reranking
                docs_for_rerank = [(doc.content, doc.metadata.source) for doc in rag_output.retrieved_documents]
                
                # Rerank documents
                reranked_docs = self.reranker.rerank(
                    query=question,
                    documents=[doc[0] for doc in docs_for_rerank]  # 只传入文档内容，不传入元数据
                )
                
                # Update retrieved documents with reranked results
                if reranked_docs:
                    # Create new document objects with reranked order
                    reranked_documents = []
                    reranked_scores = []
                    
                    for doc_content, score in reranked_docs:
                        # Find original document
                        for orig_doc in rag_output.retrieved_documents:
                            if orig_doc.content == doc_content:
                                reranked_documents.append(orig_doc)
                                reranked_scores.append(score)
                                break
                    
                    rag_output.retrieved_documents = reranked_documents
                    rag_output.retriever_scores = reranked_scores
            
            # 检索结果去重，只显示前20条chunk
            unique_docs = []
            seen_contents = set()
            for doc, score in zip(rag_output.retrieved_documents, rag_output.retriever_scores):
                if doc.content not in seen_contents:
                    unique_docs.append((doc, score))
                    seen_contents.add(doc.content)
                if len(unique_docs) >= 20:
                    break
            
            # Prepare answer
            answer = rag_output.generated_responses[0] if rag_output.generated_responses else "Unable to generate answer"
            
            # 调试信息：打印prompt和语言信息
            print(f"\n=== DEBUG INFO ===")
            print(f"Question language: {lang}")
            print(f"Data source: {detected_data_source}")
            print(f"User selected data source: {datasource}")
            print(f"Prompt template used: {rag_output.metadata.get('prompt_template', 'Unknown')}")
            print(f"Generated response: {answer[:200]}...")  # 只打印前200个字符
            print(f"=== END DEBUG ===\n")
            
            # Add reranker info to answer if used
            if use_reranker:
                answer = f"[Reranker: Enabled] {answer}"
            else:
                answer = f"[Reranker: Disabled] {answer}"
            
            # Prepare context data
            context_data = []
            for doc, score in unique_docs:
                context_data.append([f"{score:.4f}", doc.content])
            
            return answer, context_data, None
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg, [], None
    
    def launch(self, share: bool = False):
        """Launch UI interface"""
        self.interface.launch(share=share)
    
    def _chunk_documents(self, documents: List[DocumentWithMetadata], chunk_size: int = 512, overlap: int = 50) -> List[DocumentWithMetadata]:
        """
        将文档分割成更小的chunks
        
        Args:
            documents: 原始文档列表
            chunk_size: chunk大小（字符数）
            overlap: 重叠字符数
            
        Returns:
            分块后的文档列表
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                # 文档较短，不需要分块
                chunked_docs.append(doc)
            else:
                # 文档较长，需要分块
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # 确保不在单词中间截断
                    if end < len(content):
                        # 尝试在句号、逗号或空格处截断
                        for i in range(end, max(start + chunk_size - 100, start), -1):
                            if content[i] in '.。，, ':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end].strip()
                    
                    if chunk_content:  # 确保chunk不为空
                        # 创建新的文档元数据
                        chunk_metadata = DocumentMetadata(
                            source=f"{doc.metadata.source}_chunk_{chunk_id}",
                            created_at=doc.metadata.created_at,
                            author=doc.metadata.author,
                            language=doc.metadata.language
                        )
                        
                        # 创建新的文档对象
                        chunk_doc = DocumentWithMetadata(
                            content=chunk_content,
                            metadata=chunk_metadata
                        )
                        
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # 移动到下一个chunk，考虑重叠
                    start = end - overlap
                    if start >= len(content):
                        break
        
        return chunked_docs 
    
    def _chunk_documents_advanced(self, documents: List[DocumentWithMetadata]) -> List[DocumentWithMetadata]:
        """
        使用finetune_chinese_encoder.py中的高级chunk逻辑处理中文文档
        并集成finetune_encoder.py中的表格文本化处理
        """
        import re
        import json
        import ast
        
        def extract_unit_from_paragraph(paragraphs):
            """从段落中提取数值单位"""
            for para in paragraphs:
                text = para.get("text", "") if isinstance(para, dict) else para
                match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
                if match:
                    unit = match.group(1) or match.group(2)
                    if unit:
                        return unit.lower().replace('s', '') + " USD"
            return ""

        def table_to_natural_text(table_dict, caption="", unit_info=""):
            """将表格转换为自然语言描述"""
            rows = table_dict.get("table", [])
            lines = []

            if caption:
                lines.append(f"Table Topic: {caption}.")

            if not rows:
                return ""

            headers = rows[0]
            data_rows = rows[1:]

            for i, row in enumerate(data_rows):
                if not row or all(str(v).strip() == "" for v in row):
                    continue

                if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
                    lines.append(f"Table Category: {str(row[0]).strip()}.")
                    continue

                row_name = str(row[0]).strip().replace('.', '')

                data_descriptions = []
                for h_idx, v in enumerate(row):
                    if h_idx == 0:
                        continue
                    
                    header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"
                    value = str(v).strip()

                    if value:
                        if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value): 
                            formatted_value = value.replace('$', '')
                            if unit_info:
                                if formatted_value.startswith('(') and formatted_value.endswith(')'):
                                     formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                                else:
                                     formatted_value = f"${formatted_value} {unit_info}"
                            else:
                                formatted_value = f"${formatted_value}"
                        else:
                            formatted_value = value
                        
                        data_descriptions.append(f"{header} is {formatted_value}")

                if row_name and data_descriptions:
                    lines.append(f"Details for item {row_name}: {'; '.join(data_descriptions)}.")
                elif data_descriptions:
                    lines.append(f"Other data item: {'; '.join(data_descriptions)}.")
                elif row_name:
                    lines.append(f"Data item: {row_name}.")

            return "\n".join(lines)
        
        def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
            chunks = []
            if not json_str_context or not json_str_context.strip():
                return chunks
            processed_str_context = json_str_context.replace("\\n", "\n")
            cleaned_initial = re.sub(re.escape("【问题】:"), "", processed_str_context)
            cleaned_initial = re.sub(re.escape("【答案】:"), "", cleaned_initial).strip()
            cleaned_initial = cleaned_initial.replace('，', ',')
            cleaned_initial = cleaned_initial.replace('：', ':')
            cleaned_initial = cleaned_initial.replace('【', '') 
            cleaned_initial = cleaned_initial.replace('】', '') 
            cleaned_initial = cleaned_initial.replace('\u3000', ' ')
            cleaned_initial = cleaned_initial.replace('\xa0', ' ').strip()
            cleaned_initial = re.sub(r'\s+', ' ', cleaned_initial).strip()
            
            # 处理研报格式
            report_match = re.match(
                r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
                cleaned_initial, 
                re.DOTALL
            )
            if report_match:
                report_title_full = report_match.group(1).strip()
                report_date = report_match.group(2).strip()
                report_raw_content = report_match.group(3).strip() 
                content_after_second_title_match = re.match(r"研报题目是:(.+)", report_raw_content, re.DOTALL)
                if content_after_second_title_match:
                    report_content_preview = content_after_second_title_match.group(1).strip()
                else:
                    report_content_preview = report_raw_content 
                report_content_preview = re.sub(re.escape("【问题】:"), "", report_content_preview)
                report_content_preview = re.sub(re.escape("【答案】:"), "", report_content_preview).strip()
                report_content_preview = re.sub(r'\s+', ' ', report_content_preview).strip() 
                company_stock_match = re.search(r"(.+?)（(\d{6}\.\w{2})）", report_title_full)
                company_info = ""
                if company_stock_match:
                    report_company_name = company_stock_match.group(1).strip()
                    report_stock_code = company_stock_match.group(2).strip()
                    company_info = f"，公司名称：{report_company_name}，股票代码：{report_stock_code}"
                    report_title_main = re.sub(r"（\d{6}\.\w{2}）", "", report_title_full).strip()
                else:
                    report_title_main = report_title_full
                chunk_text = f"一份发布日期为 {report_date} 的研究报告，其标题是：\"{report_title_main}\"{company_info}。报告摘要内容：{report_content_preview.rstrip('...') if report_content_preview.endswith('...') else report_content_preview}。"
                chunks.append(chunk_text)
                return chunks 

            # 处理字典格式
            extracted_dict_str = None
            parsed_data = None 
            temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_initial) 
            all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 
            for potential_dict_str in all_dict_matches:
                cleaned_potential_dict_str = potential_dict_str.strip()
                json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
                try:
                    parsed_data_temp = json.loads(json_compatible_str_temp)
                    if isinstance(parsed_data_temp, dict):
                        extracted_dict_str = cleaned_potential_dict_str
                        parsed_data = parsed_data_temp
                        break 
                except json.JSONDecodeError:
                    pass 
                fixed_for_ast_eval_temp = re.sub(
                    r"(?<!['\"\w.])\b(0[1-9]\d*)\b(?![\d.]|['\"\w.])", 
                    r"'\1'", 
                    cleaned_potential_dict_str
                )
                try:
                    parsed_data_temp = ast.literal_eval(fixed_for_ast_eval_temp)
                    if isinstance(parsed_data_temp, dict):
                        extracted_dict_str = cleaned_potential_dict_str
                        parsed_data = parsed_data_temp
                        break 
                except (ValueError, SyntaxError):
                    pass 

            if extracted_dict_str is not None and isinstance(parsed_data, dict):
                for metric_name, time_series_data in parsed_data.items():
                    if not isinstance(metric_name, str):
                        metric_name = str(metric_name)
                    cleaned_metric_name = re.sub(r'（.*?）', '', metric_name).strip()
                    if not isinstance(time_series_data, dict):
                        if time_series_data is not None and str(time_series_data).strip():
                            chunks.append(f"{company_name}的{cleaned_metric_name}数据为：{time_series_data}。")
                        continue
                    if not time_series_data:
                        continue
                    try:
                        sorted_dates = sorted(time_series_data.keys(), key=str)
                    except TypeError:
                        sorted_dates = [str(k) for k in time_series_data.keys()]
                    description_parts = []
                    for date in sorted_dates:
                        value = time_series_data[date]
                        if isinstance(value, (int, float)):
                            formatted_value = f"{value:.4f}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value)
                        else:
                            formatted_value = str(value)
                        description_parts.append(f"在{date}为{formatted_value}")
                    if description_parts:
                        if len(description_parts) <= 3:
                            full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                        else:
                            first_part = "，".join(description_parts[:3])
                            last_part = "，".join(description_parts[-3:])
                            if len(sorted_dates) > 6:
                                full_description = f"{company_name}的{cleaned_metric_name}数据从{sorted_dates[0]}到{sorted_dates[-1]}，主要变化为：{first_part}，...，{last_part}。"
                            else:
                                full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                        chunks.append(full_description)
                return chunks 

            # 处理纯文本
            pure_text = cleaned_initial
            pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", pure_text, 1).strip()
            pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()
            pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()
            pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()

            if pure_text: 
                chunks.append(pure_text)
            else:
                chunks.append(f"原始格式，解析失败或无有效结构：{json_str_context.strip()[:100]}...")
            return chunks
        
        chunked_docs = []
        for doc in documents:
            # 检查是否包含表格数据
            content = doc.content
            
            # 尝试解析为JSON，检查是否包含表格结构
            try:
                parsed_content = json.loads(content)
                if isinstance(parsed_content, dict) and 'tables' in parsed_content:
                    # 处理包含表格的文档
                    paragraphs = parsed_content.get('paragraphs', [])
                    tables = parsed_content.get('tables', [])
                    
                    # 提取单位信息
                    unit_info = extract_unit_from_paragraph(paragraphs)
                    
                    # 处理段落
                    for p_idx, para in enumerate(paragraphs):
                        para_text = para.get("text", "") if isinstance(para, dict) else para
                        if para_text.strip():
                            chunk_metadata = DocumentMetadata(
                                source=f"{doc.metadata.source}_table_para_{p_idx}",
                                created_at=doc.metadata.created_at,
                                author=doc.metadata.author,
                                language=doc.metadata.language
                            )
                            chunk_doc = DocumentWithMetadata(
                                content=para_text.strip(),
                                metadata=chunk_metadata
                            )
                            chunked_docs.append(chunk_doc)
                    
                    # 处理表格
                    for t_idx, table in enumerate(tables):
                        table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                        if table_text.strip():
                            chunk_metadata = DocumentMetadata(
                                source=f"{doc.metadata.source}_table_text_{t_idx}",
                                created_at=doc.metadata.created_at,
                                author=doc.metadata.author,
                                language=doc.metadata.language
                            )
                            chunk_doc = DocumentWithMetadata(
                                content=table_text.strip(),
                                metadata=chunk_metadata
                            )
                            chunked_docs.append(chunk_doc)
                    
                    continue  # 已处理表格数据，跳过后续处理
                    
            except (json.JSONDecodeError, TypeError):
                pass  # 不是JSON格式，继续使用原有的chunk逻辑
            
            # 使用原有的高级chunk逻辑处理
            chunks = convert_json_context_to_natural_language_chunks(content)
            
            for i, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_metadata = DocumentMetadata(
                        source=f"{doc.metadata.source}_advanced_chunk_{i}",
                        created_at=doc.metadata.created_at,
                        author=doc.metadata.author,
                        language=doc.metadata.language
                    )
                    
                    chunk_doc = DocumentWithMetadata(
                        content=chunk_content,
                        metadata=chunk_metadata
                    )
                    
                    chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def _chunk_documents_simple(self, documents: List[DocumentWithMetadata], chunk_size: int = 512, overlap: int = 50) -> List[DocumentWithMetadata]:
        """
        简单的文档分块方法，用于英文文档
        """
        chunked_docs = []
        
        for doc in documents:
            content = doc.content
            if len(content) <= chunk_size:
                # 文档较短，不需要分块
                chunked_docs.append(doc)
            else:
                # 文档较长，需要分块
                start = 0
                chunk_id = 0
                
                while start < len(content):
                    end = start + chunk_size
                    
                    # 确保不在单词中间截断
                    if end < len(content):
                        # 尝试在句号、逗号或空格处截断
                        for i in range(end, max(start + chunk_size - 100, start), -1):
                            if content[i] in '.。，, ':
                                end = i + 1
                                break
                    
                    chunk_content = content[start:end].strip()
                    
                    if chunk_content:  # 确保chunk不为空
                        # 创建新的文档元数据
                        chunk_metadata = DocumentMetadata(
                            source=f"{doc.metadata.source}_simple_chunk_{chunk_id}",
                            created_at=doc.metadata.created_at,
                            author=doc.metadata.author,
                            language=doc.metadata.language
                        )
                        
                        # 创建新的文档对象
                        chunk_doc = DocumentWithMetadata(
                            content=chunk_content,
                            metadata=chunk_metadata
                        )
                        
                        chunked_docs.append(chunk_doc)
                        chunk_id += 1
                    
                    # 移动到下一个chunk，考虑重叠
                    start = end - overlap
                    if start >= len(content):
                        break
        
        return chunked_docs 