#!/usr/bin/env python3
"""
使用增强检索器的UI示例
支持双空间双索引和Qwen重排序器
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from typing import List, Optional

from config.parameters import Config
from xlm.registry.retriever import load_enhanced_retriever
from xlm.registry.generator import load_generator
from xlm.registry.rag_system import load_rag_system
from xlm.dto.dto import DocumentWithMetadata

class EnhancedRagUI:
    def __init__(
        self,
        chinese_data_path: str = "data/alphafin/alphafin_rag_ready_generated_cleaned.json",
        english_data_path: str = "data/tatqa_dataset_raw/tatqa_dataset_train.json",
        cache_dir: str = None,
        use_faiss: bool = True,
        enable_reranker: bool = True
    ):
        """
        初始化增强RAG UI
        
        Args:
            chinese_data_path: 中文数据路径
            english_data_path: 英文数据路径
            cache_dir: 缓存目录
            use_faiss: 是否使用FAISS
            enable_reranker: 是否启用重排序器
        """
        self.chinese_data_path = chinese_data_path
        self.english_data_path = english_data_path
        self.cache_dir = cache_dir
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        
        # 设置环境变量
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
            os.environ['HF_HOME'] = cache_dir
            os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
        
        # 初始化系统组件
        self._init_components()
        
        # 创建Gradio界面
        self.interface = self._create_interface()
    
    def _init_components(self):
        """初始化系统组件"""
        print("\n=== 初始化增强RAG系统 ===")
        
        # 创建配置
        self.config = Config()
        self.config.retriever.use_faiss = self.use_faiss
        self.config.reranker.enabled = self.enable_reranker
        
        if self.cache_dir:
            self.config.cache_dir = self.cache_dir
        
        print(f"配置信息:")
        print(f"- FAISS: {self.config.retriever.use_faiss}")
        print(f"- 重排序器: {self.config.reranker.enabled}")
        print(f"- 中文编码器: {self.config.encoder.chinese_model_path}")
        print(f"- 英文编码器: {self.config.encoder.english_model_path}")
        print(f"- 重排序器: {self.config.reranker.model_name}")
        
        # 加载增强检索器
        print("\n1. 加载增强检索器...")
        self.retriever = load_enhanced_retriever(
            config=self.config,
            chinese_data_path=self.chinese_data_path,
            english_data_path=self.english_data_path
        )
        
        # 加载生成器
        print("\n2. 加载生成器...")
        self.generator = load_generator(
            generator_model_name=self.config.generator.model_name,
            use_local_llm=True
        )
        
        # 初始化RAG系统
        print("\n3. 初始化RAG系统...")
        self.prompt_template = "Context: {context}\nQuestion: {question}\nAnswer:"
        self.rag_system = load_rag_system(
            retriever=self.retriever,
            generator=self.generator,
            prompt_template=self.prompt_template
        )
        
        print("=== 系统初始化完成 ===")
    
    def _create_interface(self):
        """创建Gradio界面"""
        # 示例问题
        examples = [
            ["安井食品主要生产什么产品？"],
            ["安井食品2020年营业收入是多少？"],
            ["What does Apple Inc. specialize in?"],
            ["What was Apple's total revenue in 2023?"],
            ["请介绍一下安井食品的业务情况"],
            ["Explain Apple's business model"]
        ]
        
        # 创建界面
        with gr.Blocks(title="Enhanced RAG System - Dual Space Dual Index") as interface:
            gr.Markdown("# 🚀 Enhanced RAG System")
            gr.Markdown("### 双空间双索引 + Qwen重排序器")
            gr.Markdown("支持中英文查询，自动选择对应编码器和索引空间")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    query_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="支持中英文查询，系统会自动检测语言并选择对应的编码器...",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🔍 检索并生成", variant="primary")
                        clear_btn = gr.Button("🗑️ 清空")
                    
                    # 参数设置
                    with gr.Accordion("⚙️ 参数设置", open=False):
                        top_k = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="检索文档数量 (Top-K)"
                        )
                        enable_rerank = gr.Checkbox(
                            value=self.enable_reranker,
                            label="启用重排序器"
                        )
                
                with gr.Column(scale=1):
                    # 系统信息
                    gr.Markdown("### 📊 系统信息")
                    info_text = gr.Textbox(
                        value=f"中文文档: {self.retriever.get_corpus_size()['chinese']}\n英文文档: {self.retriever.get_corpus_size()['english']}\nFAISS: {'启用' if self.use_faiss else '禁用'}\n重排序器: {'启用' if self.enable_reranker else '禁用'}",
                        label="系统状态",
                        lines=5,
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column():
                    # 检索结果
                    gr.Markdown("### 📄 检索到的文档")
                    retrieved_docs = gr.Textbox(
                        label="检索结果",
                        lines=10,
                        interactive=False
                    )
                
                with gr.Column():
                    # 生成答案
                    gr.Markdown("### 💡 生成的答案")
                    generated_answer = gr.Textbox(
                        label="答案",
                        lines=10,
                        interactive=False
                    )
            
            # 示例
            gr.Markdown("### 💡 示例问题")
            gr.Examples(
                examples=examples,
                inputs=query_input
            )
            
            # 事件处理
            def process_query(query, top_k_val, enable_rerank_val):
                if not query.strip():
                    return "", "", ""
                
                try:
                    # 更新配置
                    self.config.retriever.rerank_top_k = top_k_val
                    self.config.reranker.enabled = enable_rerank_val
                    
                    # 执行检索
                    print(f"\n处理查询: {query}")
                    print(f"参数: top_k={top_k_val}, reranker={enable_rerank_val}")
                    
                    # 检索文档
                    retrieved_documents, retriever_scores = self.retriever.retrieve(
                        text=query,
                        top_k=top_k_val,
                        return_scores=True
                    )
                    
                    # 格式化检索结果
                    docs_text = ""
                    for i, (doc, score) in enumerate(zip(retrieved_documents, retriever_scores)):
                        docs_text += f"文档 {i+1} (分数: {score:.4f}):\n"
                        docs_text += f"{doc.content}\n"
                        docs_text += f"元数据: {doc.metadata.doc_id}, {doc.metadata.language}\n"
                        docs_text += "-" * 50 + "\n"
                    
                    # 生成答案
                    if retrieved_documents:
                        rag_output = self.rag_system.run(query)
                        answer = rag_output.generated_responses[0] if rag_output.generated_responses else "无法生成答案"
                    else:
                        answer = "未找到相关文档，无法生成答案"
                    
                    return docs_text, answer, ""
                    
                except Exception as e:
                    error_msg = f"处理查询时发生错误: {str(e)}"
                    print(error_msg)
                    return "", "", error_msg
            
            def clear_outputs():
                return "", "", ""
            
            # 绑定事件
            submit_btn.click(
                fn=process_query,
                inputs=[query_input, top_k, enable_rerank],
                outputs=[retrieved_docs, generated_answer, info_text]
            )
            
            clear_btn.click(
                fn=clear_outputs,
                outputs=[query_input, retrieved_docs, generated_answer]
            )
            
            query_input.submit(
                fn=process_query,
                inputs=[query_input, top_k, enable_rerank],
                outputs=[retrieved_docs, generated_answer, info_text]
            )
        
        return interface
    
    def launch(self, **kwargs):
        """启动UI"""
        return self.interface.launch(**kwargs)

def main():
    """主函数"""
    # 创建UI实例
    ui = EnhancedRagUI(
        chinese_data_path="data/alphafin/alphafin_rag_ready_generated_cleaned.json",
        english_data_path="data/tatqa_dataset_raw/tatqa_dataset_train.json",
        cache_dir="M:/huggingface",
        use_faiss=True,
        enable_reranker=True
    )
    
    # 启动UI
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main() 