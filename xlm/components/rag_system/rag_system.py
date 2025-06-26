from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import RagOutput
import os
import collections
import re
from langdetect import detect, LangDetectException

# Define the robust "Golden Prompts" directly in the code
PROMPT_TEMPLATE_EN = """You are a highly analytical and precise financial expert. Your task is to answer the user's question **strictly based on the provided <context> information**.

Requirements:
1.  **Strictly adhere to the provided <context>. Do not use any external knowledge or make assumptions.**
2.  If the <context> does not contain sufficient information to answer the question, state: "The answer cannot be found in the provided context."
3.  For questions involving financial predictions or future outlook, prioritize information explicitly stated as forecasts or outlooks within the <context>. If the <context> specifies a report date, base your answer on that date's perspective.
4.  Provide a concise and direct answer in complete sentences.
5.  Do not repeat the question or add conversational fillers.

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_ZH = """你是一位严谨且精确的金融分析专家。请你**严格根据以下<context>标签内提供的信息**来回答用户的问题。

要求：
1.  **严格遵循<context>中提供的信息。切勿使用任何外部知识或进行猜测。**
2.  如果<context>中没有足够的信息来回答问题，请直接说："在提供的上下文中找不到答案。"
3.  对于涉及金融预测或未来展望的问题，请优先提取<context>中明确陈述为预测或展望的信息。如果<context>中提及了报告发布日期，请以该日期时的视角进行回答。
4.  用中文进行回答，内容要简洁、直接，形成完整的句子，回答不超过2-3句话。
5.  不要重复问题内容或添加无关的寒暄。

上下文:
{context}

问题: {question}

回答:"""


class RagSystem:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retriever_top_k: int,
        prompt_template: str = None, # No longer used, but kept for compatibility
    ):
        self.retriever = retriever
        self.generator = generator
        # self.prompt_template is now obsolete
        self.retriever_top_k = retriever_top_k

    def run(self, user_input: str, language: str = None) -> RagOutput:
        # 1. Detect language of the user's question
        if language is None:
            try:
                lang = detect(user_input)
            except LangDetectException:
                lang = 'en' # Default to English if detection fails
            is_chinese_q = lang.startswith('zh')
            language = 'zh' if is_chinese_q else 'en'
        else:
            is_chinese_q = (language == 'zh')
        
        # 2. Retrieve relevant documents
        retrieved_documents, retriever_scores = self.retriever.retrieve(
            text=user_input, top_k=self.retriever_top_k, return_scores=True, language=language
        )

        # 3. Select prompt based on question language and format the context
        if is_chinese_q:
            prompt_template = PROMPT_TEMPLATE_ZH
            no_context_message = "未找到合适的语料，请检查数据源。"
        else:
            prompt_template = PROMPT_TEMPLATE_EN
            no_context_message = "No suitable context found for your question. Please check the data sources."

        if not retrieved_documents:
            return RagOutput(
                retrieved_documents=[],
                retriever_scores=[],
                prompt="",
                generated_responses=[no_context_message],
                metadata={}
            )

        context_str = "\n\n".join([doc.content for doc in retrieved_documents])
        
        # 4. Create the final prompt
        prompt = prompt_template.format(context=context_str, question=user_input)
        
        # 5. Generate the response
        generated_responses = self.generator.generate(texts=[prompt])
        
        # 6. Gather metadata
        retriever_model_name = ""
        if hasattr(self.retriever, 'encoder_en') and hasattr(self.retriever, 'encoder_ch'):
            if is_chinese_q:
                retriever_model_name = getattr(self.retriever.encoder_ch, 'model_name', 'unknown')
            else:
                retriever_model_name = getattr(self.retriever.encoder_en, 'model_name', 'unknown')

        return RagOutput(
            retrieved_documents=retrieved_documents,
            retriever_scores=retriever_scores,
            prompt=prompt,
            generated_responses=generated_responses,
            metadata=dict(
                retriever_model_name=retriever_model_name,
                top_k=self.retriever_top_k,
                generator_model_name=self.generator.model_name,
                prompt_template="Golden-" + ("ZH" if is_chinese_q else "EN"),
                question_language="zh" if is_chinese_q else "en"
            ),
        )
