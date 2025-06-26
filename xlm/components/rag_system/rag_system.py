from xlm.components.generator.generator import Generator
from xlm.components.retriever.retriever import Retriever
from xlm.dto.dto import RagOutput
import os
import collections
import re
from langdetect import detect, LangDetectException

# Define the robust "Golden Prompts" directly in the code
PROMPT_TEMPLATE_EN = """You are a professional financial analyst. Answer the question based on the provided context.

Requirements:
1. Use concise, direct answers
2. Only use information from the provided context
3. If the answer is not in the context, say "The answer cannot be found in the provided context"
4. Do not repeat the question
5. Do not add information outside the context
6. Ensure complete sentences

Context:
{context}

Question: {question}

Answer:"""

PROMPT_TEMPLATE_ZH = """基于上下文信息回答问题。

要求：
1. 用中文回答
2. 只基于提供的上下文信息
3. 如果上下文中没有答案，直接说"在提供的上下文中找不到答案"
4. 回答要简洁、直接，不超过2-3句话
5. 不要重复问题内容
6. 不要添加上下文之外的信息

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
