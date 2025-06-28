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

Example 1:
Context: Apple Inc. reported Q3 2023 revenue of $81.8 billion, down 1.4% year-over-year. iPhone sales increased 2.8% to $39.7 billion.
Question: How did Apple perform in Q3 2023?
Answer: Apple's Q3 2023 performance was mixed. Total revenue declined 1.4% to $81.8 billion, but iPhone sales grew 2.8% to $39.7 billion.

Example 2:
Context: Tesla's vehicle deliveries in Q2 2023 reached 466,140 units, up 83% from the previous year. Production capacity utilization improved to 95%.
Question: What were Tesla's delivery numbers in Q2 2023?
Answer: Tesla delivered 466,140 vehicles in Q2 2023, representing an 83% increase from the previous year.

Context:
{context}

Question: {question}

Answer:"""

# 超简洁版本（推荐用于生产环境）
PROMPT_TEMPLATE_ZH_SIMPLE = """基于上下文信息，用一句话回答用户问题。

上下文：{context}
问题：{question}
回答："""

PROMPT_TEMPLATE_ZH = """基于以下上下文信息，直接回答用户问题。只使用提供的信息，不要添加任何外部知识或格式化内容。

示例1：
上下文：中国平安2023年第一季度实现营业收入2,345.67亿元，同比增长8.5%；净利润为156.78亿元，同比增长12.3%。
问题：中国平安的业绩如何？
回答：中国平安2023年第一季度业绩表现良好，营业收入同比增长8.5%至2,345.67亿元，净利润同比增长12.3%至156.78亿元。

示例2：
上下文：腾讯控股2023年上半年游戏业务收入同比下降5.2%，广告业务收入同比增长3.1%。
问题：腾讯的游戏业务表现如何？
回答：腾讯2023年上半年游戏业务收入同比下降5.2%，表现不佳。

上下文：
{context}

问题：{question}

回答："""

# Chain-of-Thought版本（优化版，隐藏思考过程）
PROMPT_TEMPLATE_ZH_COT = """你是一位专业的金融分析师。请基于以下上下文信息，通过内部思考来回答用户问题。

重要要求：
1. 请进行内部思考，但不要输出任何思考步骤或过程
2. 直接给出最终答案，不要包含"思考"、"步骤"等词汇
3. 只使用提供的上下文信息，不要添加外部知识
4. 回答要简洁直接，用自然的中文表达

上下文：
{context}

问题：{question}

回答："""


class RagSystem:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        retriever_top_k: int,
        prompt_template: str = None, # No longer used, but kept for compatibility
        use_cot: bool = False,  # 是否使用Chain-of-Thought
        use_simple: bool = False,  # 是否使用超简洁模式
    ):
        self.retriever = retriever
        self.generator = generator
        # self.prompt_template is now obsolete
        self.retriever_top_k = retriever_top_k
        self.use_cot = use_cot
        self.use_simple = use_simple

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
            if self.use_simple:
                prompt_template = PROMPT_TEMPLATE_ZH_SIMPLE
            elif self.use_cot:
                prompt_template = PROMPT_TEMPLATE_ZH_COT
            else:
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

        # 确定prompt模板类型
        if is_chinese_q:
            if self.use_simple:
                template_type = "ZH-SIMPLE"
            elif self.use_cot:
                template_type = "ZH-COT"
            else:
                template_type = "ZH"
        else:
            template_type = "EN"

        return RagOutput(
            retrieved_documents=retrieved_documents,
            retriever_scores=retriever_scores,
            prompt=prompt,
            generated_responses=generated_responses,
            metadata=dict(
                retriever_model_name=retriever_model_name,
                top_k=self.retriever_top_k,
                generator_model_name=self.generator.model_name,
                prompt_template=f"Golden-{template_type}",
                question_language="zh" if is_chinese_q else "en",
                use_cot=self.use_cot,
                use_simple=self.use_simple
            ),
        )
