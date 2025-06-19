"""
Optimized UI for unified financial data system using Gradio
"""

import os
import warnings
# import gradio as gr # 移除 Gradio 导入
from pathlib import Path
import numpy as np
import sys
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xlm.utils.unified_data_loader import UnifiedDataLoader
from xlm.registry.generator import load_generator
from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.sbert_retriever import SBERTRetriever
from xlm.components.encoder.multimodal_encoder import MultiModalEncoder
from config.parameters import Config, EncoderConfig, RetrieverConfig, ModalityConfig

# Ignore warnings
warnings.filterwarnings("ignore")


def ensure_directories():
    """Ensure required directories exist"""
    dirs = [
        "data",
        "data/processed",
        "data/tatqa_dataset_raw",
        "data/alphafin"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def is_chinese(text):
    """检测文本是否包含中文"""
    pattern = re.compile(r'[\u4e00-\u9fff]')
    return bool(pattern.search(text))


# 移除了 load_model 和 generate_response 函数，因为它们在原始 Gradio UI 逻辑中用于直接的文本生成，
# 而我们的 RAG 系统有自己的生成器加载逻辑。
# 如果您的原始 `load_generator` 函数（来自 xlm.registry.generator）能够处理小型本地模型，
# 那么我们继续使用它。

class OptimizedConsoleQA:  # 将类名更改为更具描述性
    def __init__(self):
        self.data_loader = UnifiedDataLoader(
            batch_size=16,  # 减小批处理大小
            max_samples=500  # 减少每个数据源的样本数
        )
        self.documents = self.data_loader.documents
        self.rag_system = None

    def init_model(self):
        """Initialize the model and system components"""
        if self.rag_system is not None:  # 检查 RAG 系统是否已经初始化
            return

        try:
            # 初始化检索器
            print("Initializing retriever...")
            config = Config(
                encoder=EncoderConfig(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu",  # 在本地测试时，检索器编码器通常可以在 CPU 上运行
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
                use_enhanced_encoders=True  # 使用增强编码器
            )
            retriever = SBERTRetriever(
                encoder=encoder,
                corpus_documents=self.documents
            )

            # 初始化生成器：这里使用原始的 load_generator，它会加载 facebook/opt-125m
            print("加载生成器模型: facebook/opt-125m (本地测试用小型模型)")
            generator = load_generator(
                generator_model_name="facebook/opt-125m",  # 使用小型模型进行本地测试
                use_local_llm=True
            )
            print("生成器模型加载完成")

            # 初始化RAG系统
            print("Initializing RAG system...")
            self.rag_system = RagSystem(
                retriever=retriever,
                generator=generator,
                prompt_template="Context: {context}\nQuestion: {question}\nAnswer:",
                retriever_top_k=1
            )

        except Exception as e:
            print(f"初始化过程中出错: {str(e)}")
            traceback.print_exc()
            raise e

    def process_query(self, query: str):  # 移除了 datasource 参数，因为控制台模式下通常不需要选择
        """Process a query and return results"""
        try:
            if self.rag_system is None:
                self.init_model()

            print(f"处理查询: {query}")
            rag_output = self.rag_system.run(user_input=query)

            # 准备输出
            answer = rag_output.generated_responses[0]
            doc = rag_output.retrieved_documents[0]
            score = rag_output.retriever_scores[0]

            # 构建响应
            response = {
                "answer": answer,
                "context": f"相关度分数: {score:.4f}\n\n{doc.content}"
            }

            return response["answer"], response["context"]

        except Exception as e:
            print(f"处理查询时发生错误: {str(e)}")
            traceback.print_exc()
            return "处理查询时发生错误", str(e)


if __name__ == "__main__":
    # 确保必要的目录存在
    ensure_directories()

    # 启动 QA 系统实例
    print("Starting Financial QA System (Console Mode for Local Test)...")
    qa_system = OptimizedConsoleQA()

    # 预加载模型和RAG系统 (本地测试时也会下载 facebook/opt-125m)
    print("首次初始化模型和RAG系统，这可能需要一段时间...")
    try:
        qa_system.init_model()
        print("模型和RAG系统初始化完成。")
    except Exception as e:
        print(f"系统启动失败: {e}")
        sys.exit(1)

    # Interactive command-line mode
    print("\nEnter your question, type 'exit' to quit.")
    while True:
        user_input = input("\nQuestion: ")
        if user_input.lower() == 'exit':
            break

        answer, context = qa_system.process_query(user_input)
        print("\nAnswer:")
        print(answer)
        print("\nExplanation (Context):")
        print(context)
        print("-" * 80)