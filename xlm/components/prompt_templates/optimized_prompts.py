"""
Optimized prompt templates with few-shot examples for RAG system.
"""

from typing import Dict, List, Optional

class OptimizedPromptTemplates:
    """优化的Prompt模板类，包含few-shot示例"""
    
    def __init__(self):
        self.few_shot_examples_en = [
            {
                "context": "The company reported revenue of $1.2 billion in 2023, an increase of 15% from the previous year. Net income was $180 million.",
                "question": "What was the company's revenue in 2023?",
                "answer": "$1.2 billion"
            },
            {
                "context": "The modified retrospective method was used for the adoption of Topic 606. This resulted in a cumulative adjustment of $0.5 million.",
                "question": "What method was used for Topic 606 adoption?",
                "answer": "the modified retrospective method"
            },
            {
                "context": "Capital expenditures totaled $50 million, including $30 million for equipment and $20 million for facilities.",
                "question": "How much was spent on equipment?",
                "answer": "$30 million"
            }
        ]
        
        self.few_shot_examples_zh = [
            {
                "context": "公司2023年实现营业收入52.66亿元，同比增长23.66%。归母净利润3.73亿元，同比增长38.14%。",
                "question": "公司2023年的营业收入是多少？",
                "answer": "52.66亿元"
            },
            {
                "context": "安井食品于2020年4月14日发布2019年度报告，报告期内公司实现收入52.66亿。",
                "question": "安井食品何时发布年度报告？",
                "answer": "2020年4月14日"
            }
        ]
    
    def format_few_shot_examples(self, examples: List[Dict], language: str = "en") -> str:
        """格式化few-shot示例"""
        formatted_examples = []
        
        for example in examples:
            if language == "en":
                formatted_examples.append(
                    f"Context: {example['context']}\n"
                    f"Question: {example['question']}\n"
                    f"Answer: {example['answer']}\n"
                )
            else:
                formatted_examples.append(
                    f"上下文：{example['context']}\n"
                    f"问题：{example['question']}\n"
                    f"答案：{example['answer']}\n"
                )
        
        return "\n".join(formatted_examples)
    
    def get_optimized_prompt(self, context: str, question: str, language: str = "en") -> str:
        """获取优化的Prompt"""
        
        if language == "en":
            examples = self.few_shot_examples_en[:2]  # 使用前2个示例
            formatted_examples = self.format_few_shot_examples(examples, "en")
            
            prompt = f"""Answer the following question based on the provided context. Give a direct, concise answer in 1-2 sentences maximum.

{formatted_examples}

Context: {context}

Question: {question}

Answer:"""
        else:
            examples = self.few_shot_examples_zh[:2]  # 使用前2个示例
            formatted_examples = self.format_few_shot_examples(examples, "zh")
            
            prompt = f"""根据提供的上下文回答以下问题。请给出直接、简洁的答案，最多1-2句话。

{formatted_examples}

上下文：{context}

问题：{question}

答案："""
        
        return prompt
    
    def get_strict_prompt(self, context: str, question: str, language: str = "en") -> str:
        """获取严格模式的Prompt（无few-shot）"""
        
        if language == "en":
            prompt = f"""Based on the context, provide a direct answer to the question. Keep your response brief and factual.

Context: {context}

Question: {question}

Direct Answer:"""
        else:
            prompt = f"""根据上下文直接回答问题。保持回答简洁且基于事实。

上下文：{context}

问题：{question}

直接答案："""
        
        return prompt
    
    def get_qa_prompt(self, context: str, question: str, language: str = "en") -> str:
        """获取简单的QA格式Prompt"""
        
        if language == "en":
            prompt = f"""Question: {question}

Context: {context}

Answer:"""
        else:
            prompt = f"""问题：{question}

上下文：{context}

答案："""
        
        return prompt

# 全局实例
optimized_prompts = OptimizedPromptTemplates() 