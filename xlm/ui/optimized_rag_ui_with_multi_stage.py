#!/usr/bin/env python3
"""
Optimized RAG UI with Multi-Stage Retrieval System Integration
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Optional, Tuple
import gradio as gr
import numpy as np
import torch
import faiss
from langdetect import detect, LangDetectException

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata, RagOutput
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.generator.generator import Generator
from xlm.components.retriever.reranker import QwenReranker
from xlm.utils.visualizer import Visualizer
from xlm.registry.retriever import load_enhanced_retriever
from xlm.registry.generator import load_generator
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig, EMBEDDING_CACHE_DIR, RERANKER_CACHE_DIR

# 导入多阶段检索系统
try:
    sys.path.append(str(Path(__file__).parent.parent.parent / "alphafin_data_process"))
    from multi_stage_retrieval_final import MultiStageRetrievalSystem
    MULTI_STAGE_AVAILABLE = True
except ImportError as e:
    print(f"多阶段检索系统导入失败: {e}")
    MULTI_STAGE_AVAILABLE = False

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

class OptimizedRagUIWithMultiStage:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_faiss: bool = True,
        enable_reranker: bool = True,
        use_existing_embedding_index: Optional[bool] = None,
        max_alphafin_chunks: Optional[int] = None,
        window_title: str = "Financial Explainable RAG System with Multi-Stage Retrieval",
        title: str = "Financial Explainable RAG System with Multi-Stage Retrieval",
        examples: Optional[List[List[str]]] = None,
    ):
        # 使用config中的平台感知配置
        config = Config()
        self.cache_dir = EMBEDDING_CACHE_DIR if (not cache_dir or not isinstance(cache_dir, str)) else cache_dir
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        self.use_existing_embedding_index = use_existing_embedding_index if use_existing_embedding_index is not None else config.retriever.use_existing_embedding_index
        self.max_alphafin_chunks = max_alphafin_chunks if max_alphafin_chunks is not None else config.retriever.max_alphafin_chunks
        self.window_title = window_title
        self.title = title
        self.examples = examples or [
            ["德赛电池(000049)的下一季度收益预测如何？"],
            ["用友网络2019年的每股经营活动产生的现金流量净额是多少？"],
            ["下月股价能否上涨?"],
            ["How was internally developed software capitalised?"],
            ["Why did the Operating revenues decreased from 2018 to 2019?"],
            ["Why did the Operating costs decreased from 2018 to 2019?"]
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
        """Initialize RAG system components with multi-stage retrieval"""
        print("\nStep 1. Initializing Multi-Stage Retrieval System...")
        
        # 使用config中的平台感知配置
        config = Config()
        
        # 初始化多阶段检索系统
        if MULTI_STAGE_AVAILABLE:
            try:
                # 中文数据路径
                chinese_data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
                
                if chinese_data_path.exists():
                    print("✅ 初始化中文多阶段检索系统...")
                    self.chinese_retrieval_system = MultiStageRetrievalSystem(
                        data_path=chinese_data_path,
                        dataset_type="chinese",
                        use_existing_config=True
                    )
                    print("✅ 中文多阶段检索系统初始化完成")
                else:
                    print(f"❌ 中文数据文件不存在: {chinese_data_path}")
                    self.chinese_retrieval_system = None
                
                # 英文数据路径（如果有的话）
                english_data_path = Path("data/tatqa/processed_data.json")  # 需要预处理
                if english_data_path.exists():
                    print("✅ 初始化英文多阶段检索系统...")
                    self.english_retrieval_system = MultiStageRetrievalSystem(
                        data_path=english_data_path,
                        dataset_type="english",
                        use_existing_config=True
                    )
                    print("✅ 英文多阶段检索系统初始化完成")
                else:
                    print(f"⚠️ 英文数据文件不存在: {english_data_path}")
                    self.english_retrieval_system = None
                
            except Exception as e:
                print(f"❌ 多阶段检索系统初始化失败: {e}")
                self.chinese_retrieval_system = None
                self.english_retrieval_system = None
        else:
            print("❌ 多阶段检索系统不可用，回退到传统检索")
            self.chinese_retrieval_system = None
            self.english_retrieval_system = None
        
        print("\nStep 2. Loading generator...")
        self.generator = load_generator(
            generator_model_name=config.generator.model_name,
            use_local_llm=True,
            use_gpu=True,
            gpu_device="cuda:1",
            cache_dir=config.generator.cache_dir
        )
        
        print("\nStep 3. Loading visualizer...")
        self.visualizer = Visualizer(show_mid_features=True)
        
        print("✅ 所有组件初始化完成")
    
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
    ) -> tuple[str, List[List[str]]]:
        """Process user question using multi-stage retrieval"""
        if not question.strip():
            return "Please enter a question.", []
        
        print(f"\nProcessing question: {question}")
        print(f"Data source: {datasource}")
        
        # Detect language
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
        
        # 使用多阶段检索系统
        if MULTI_STAGE_AVAILABLE:
            try:
                # 根据语言选择检索系统
                if language == 'zh' and self.chinese_retrieval_system:
                    print("🔍 使用中文多阶段检索系统...")
                    retrieval_system = self.chinese_retrieval_system
                    
                    # 尝试提取公司名称和股票代码用于元数据过滤
                    company_name = None
                    stock_code = None
                    
                    # 简单的实体提取（可以改进）
                    import re
                    # 提取股票代码
                    stock_match = re.search(r'\((\d{6})\)', question)
                    if stock_match:
                        stock_code = stock_match.group(1)
                    
                    # 提取公司名称（简单实现）
                    company_patterns = [
                        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险))',
                        r'([^，。？\s]+(?:股份|集团|公司|有限|科技|网络|银行|证券|保险)[^，。？\s]*)'
                    ]
                    
                    for pattern in company_patterns:
                        company_match = re.search(pattern, question)
                        if company_match:
                            company_name = company_match.group(1)
                            break
                    
                    # 执行多阶段检索
                    results = retrieval_system.search(
                        query=question,
                        company_name=company_name,
                        stock_code=stock_code,
                        top_k=20
                    )
                    
                    # 转换为DocumentWithMetadata格式
                    retrieved_documents = []
                    retriever_scores = []
                    
                    for result in results:
                        # 创建DocumentWithMetadata对象
                        doc = DocumentWithMetadata(
                            content=result.get('original_context', result.get('summary', '')),
                            metadata=DocumentMetadata(
                                source=result.get('company_name', 'Unknown'),
                                created_at="",
                                author="",
                                language="chinese"
                            )
                        )
                        retrieved_documents.append(doc)
                        retriever_scores.append(result.get('combined_score', 0.0))
                    
                    print(f"✅ 多阶段检索完成，找到 {len(retrieved_documents)} 条结果")
                    
                elif language == 'en' and self.english_retrieval_system:
                    print("🔍 使用英文多阶段检索系统...")
                    retrieval_system = self.english_retrieval_system
                    
                    # 英文数据不支持元数据过滤
                    results = retrieval_system.search(
                        query=question,
                        top_k=20
                    )
                    
                    # 转换为DocumentWithMetadata格式
                    retrieved_documents = []
                    retriever_scores = []
                    
                    for result in results:
                        doc = DocumentWithMetadata(
                            content=result.get('context', result.get('content', '')),
                            metadata=DocumentMetadata(
                                source=result.get('source', 'Unknown'),
                                created_at="",
                                author="",
                                language="english"
                            )
                        )
                        retrieved_documents.append(doc)
                        retriever_scores.append(result.get('combined_score', 0.0))
                    
                    print(f"✅ 多阶段检索完成，找到 {len(retrieved_documents)} 条结果")
                    
                else:
                    print("⚠️ 多阶段检索系统不可用，使用传统检索")
                    return self._fallback_retrieval(question, language)
                
                # 生成答案
                if retrieved_documents:
                    # 构建上下文
                    context_str = "\n\n".join([doc.content for doc in retrieved_documents[:10]])
                    
                    # 根据语言选择prompt模板
                    if language == 'zh':
                        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH
                        prompt = PROMPT_TEMPLATE_ZH.format(context=context_str, question=question)
                    else:
                        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_EN
                        prompt = PROMPT_TEMPLATE_EN.format(context=context_str, question=question)
                    
                    # 生成答案
                    generated_responses = self.generator.generate(texts=[prompt])
                    answer = generated_responses[0] if generated_responses else "Unable to generate answer"
                    
                    # 准备上下文数据
                    context_data = []
                    for doc, score in zip(retrieved_documents[:20], retriever_scores[:20]):
                        context_data.append([f"{score:.4f}", doc.content[:500] + "..." if len(doc.content) > 500 else doc.content])
                    
                    # 添加检索系统信息
                    answer = f"[Multi-Stage Retrieval: {language.upper()}] {answer}"
                    
                    return answer, context_data
                else:
                    return "No relevant documents found.", []
                    
            except Exception as e:
                print(f"❌ 多阶段检索失败: {e}")
                return self._fallback_retrieval(question, language)
        else:
            print("❌ 多阶段检索系统不可用，使用传统检索")
            return self._fallback_retrieval(question, language)
    
    def _fallback_retrieval(self, question: str, language: str) -> tuple[str, List[List[str]]]:
        """传统检索回退方法"""
        # 这里可以实现传统的检索逻辑
        return f"[Fallback Retrieval: {language.upper()}] Traditional retrieval not implemented yet.", []
    
    def launch(self, share: bool = False):
        """Launch UI interface"""
        self.interface.launch(share=share) 