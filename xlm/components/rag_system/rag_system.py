from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import RagOutput
import os
import collections
import re


class RagSystem:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        prompt_template: str,
        retriever_top_k: int,
    ):
        self.retriever = retriever
        self.generator = generator
        self.prompt_template = prompt_template
        self.retriever_top_k = retriever_top_k

    def _select_prompt_template(self, doc_sources, task_type="qa"):
        """
        根据文档来源和任务类型选择prompt模板文件路径。
        task_type: "qa"/"table"/"paragraph"/"news"/"time_series"
        """
        # TatQA 英文模板
        tatqa_map = {
            "qa": "tatqa_qa.txt",
            "table": "tatqa_table.txt",
            "paragraph": "tatqa_paragraph.txt"
        }
        # AlphaFin 中文模板
        alphafin_map = {
            "news": "alphafin_financial_news.txt",
            "qa": "alphafin_stock_qa.txt",
            "time_series": "alphafin_time_series.txt"
        }
        prompt_dir = "data/prompt_templates"
        # 简单判断
        if any("tatqa" in src.lower() for src in doc_sources):
            fname = tatqa_map.get(task_type, "tatqa_qa.txt")
        elif any("stock_data" in src or "time_series" in src or "financial_news" in src for src in doc_sources):
            # AlphaFin
            if any("time_series" in src for src in doc_sources):
                fname = alphafin_map["time_series"]
            elif any("financial_news" in src for src in doc_sources):
                fname = alphafin_map["news"]
            else:
                fname = alphafin_map["qa"]
        else:
            fname = "tatqa_qa.txt"  # fallback
        return os.path.join(prompt_dir, fname)

    def _load_prompt_template(self, template_path):
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return self.prompt_template

    def is_chinese(self, text):
        return bool(re.search(r'[\u4e00-\u9fff]', text))

    def run(self, user_input: str) -> RagOutput:
        # 检测问题语言
        question_is_chinese = self.is_chinese(user_input)
        
        retrieved_documents, retriever_scores = self.retriever.retrieve(
            text=user_input, top_k=self.retriever_top_k, return_scores=True
        )
        doc_sources = [doc.metadata.source for doc in retrieved_documents]
        # 简单任务类型推断
        if any("table" in src for src in doc_sources):
            task_type = "table"
        elif any("paragraph" in src for src in doc_sources):
            task_type = "paragraph"
        elif any("news" in src for src in doc_sources):
            task_type = "news"
        elif any("time_series" in src for src in doc_sources):
            task_type = "time_series"
        else:
            task_type = "qa"
        # 动态选择prompt模板：中文问题用AlphaFin模板，英文问题用TatQA模板
        if question_is_chinese:
            # AlphaFin中文模板
            alphafin_map = {
                "news": "alphafin_financial_news.txt",
                "qa": "alphafin_stock_qa.txt",
                "time_series": "alphafin_time_series.txt"
            }
            prompt_dir = "data/prompt_templates"
            fname = alphafin_map.get(task_type, "alphafin_stock_qa.txt")
            template_path = os.path.join(prompt_dir, fname)
        else:
            # TatQA英文模板
            tatqa_map = {
                "qa": "tatqa_qa.txt",
                "table": "tatqa_table.txt",
                "paragraph": "tatqa_paragraph.txt"
            }
            prompt_dir = "data/prompt_templates"
            fname = tatqa_map.get(task_type, "tatqa_qa.txt")
            template_path = os.path.join(prompt_dir, fname)
        prompt_template = self._load_prompt_template(template_path)
        document_contents = [doc.content for doc in retrieved_documents]
        # 收集所有metadata字段
        extra_kwargs = {}
        for doc in retrieved_documents:
            if doc.metadata and isinstance(doc.metadata, dict):
                extra_kwargs.update(doc.metadata)
        # 兜底：用defaultdict防止KeyError
        safe_kwargs = dict(extra_kwargs)
        safe_kwargs["context"] = "\n".join(document_contents)
        safe_kwargs["question"] = user_input
        # 兜底：为所有常见模板变量提供默认值，防止KeyError
        for key in [
            "answer", "answer_type", "scale", "derivation", "table_content", "paragraph", "stock_code", "date", "raw_data", "indicator", "word_count", "sentence_count"
        ]:
            if key not in safe_kwargs:
                safe_kwargs[key] = ""
        prompt = prompt_template.format_map(safe_kwargs)
        # 根据问题语言加前缀指令
        if question_is_chinese:
            prompt = "请用中文回答。\n" + prompt
        else:
            prompt = "Please answer in English.\n" + prompt
        generated_responses = self.generator.generate(texts=[prompt])
        return RagOutput(
            retrieved_documents=retrieved_documents,
            retriever_scores=retriever_scores,
            prompt=prompt,
            generated_responses=generated_responses,
            metadata=dict(
                retriever_model_name=self.retriever.encoder.model_name,
                top_k=self.retriever_top_k,
                generator_model_name=self.generator.model_name,
                prompt_template=prompt_template,
                question_language="zh" if question_is_chinese else "en"
            ),
        )
