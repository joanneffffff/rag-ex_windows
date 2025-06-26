终端有问题#!/usr/bin/env python3
"""
优化的RAG系统性能评估脚本
使用Qwen3-8B模型和优化的few-shot Prompt模板
"""

import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from xlm.ui.optimized_rag_ui import OptimizedRagUI
from xlm.components.rag_system.rag_system import RagSystem
from xlm.registry.retriever import load_bilingual_retriever
from xlm.registry.generator import load_generator
from xlm.utils.unified_data_loader import UnifiedDataLoader
from config.parameters import config
from xlm.components.prompt_templates.optimized_prompts import optimized_prompts
import torch
from sklearn.metrics.pairwise import cosine_similarity
import re

@dataclass
class EvaluationResult:
    question: str
    context: str
    answer: str
    retrieved_chunks: List[str]
    retrieved_scores: List[float]
    generated_answer: str
    retrieval_relevance: float
    generation_accuracy: float
    is_retrieval_good: bool
    is_generation_good: bool
    prompt_type: str

class OptimizedRAGEvaluator:
    def __init__(self):
        """初始化优化的RAG评估器"""
        print("初始化优化的RAG评估器...")
        print("使用模型: Qwen3-8B-Instruct")
        print("使用优化Prompt模板")
        
        # 修改配置使用Qwen3-8B
        config.generator.model_name = "Qwen/Qwen3-8B-Instruct"
        config.generator.temperature = 0.1
        config.generator.max_new_tokens = 100
        
        # 加载优化的RAG系统
        self.ui = OptimizedRagUI(
            cache_dir="/users/sgjfei3/data/huggingface",
            use_faiss=True,
            enable_reranker=True,
            window_title="优化RAG评估",
            title="优化RAG评估"
        )
        
        # 加载评估数据
        self.eval_data = self._load_evaluation_data()
        
        print(f"加载了 {len(self.eval_data)} 个评估样本")
    
    def _load_evaluation_data(self) -> List[Dict]:
        """加载评估数据"""
        eval_data = []
        
        # 加载TAT-QA评估数据
        try:
            with open('evaluate_mrr/tatqa_eval.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    eval_data.append({
                        'question': data['query'],
                        'context': data.get('context', ''),
                        'answer': data.get('answer', ''),
                        'source': 'tatqa',
                        'language': 'en'
                    })
            print(f"加载了 {len([d for d in eval_data if d['source'] == 'tatqa'])} 个TAT-QA样本")
        except Exception as e:
            print(f"加载TAT-QA数据失败: {e}")
        
        # 加载AlphaFin评估数据
        try:
            with open('evaluate_mrr/alphafin_eval.jsonl', 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    eval_data.append({
                        'question': data['query'],
                        'context': data.get('context', ''),
                        'answer': data.get('answer', ''),
                        'source': 'alphafin',
                        'language': 'zh'
                    })
            print(f"加载了 {len([d for d in eval_data if d['source'] == 'alphafin'])} 个AlphaFin样本")
        except Exception as e:
            print(f"加载AlphaFin数据失败: {e}")
        
        return eval_data
    
    def _calculate_retrieval_relevance(self, question: str, retrieved_chunks: List[str], 
                                     original_context: str) -> float:
        """计算检索相关性分数"""
        if not retrieved_chunks:
            return 0.0
        
        # 方法1: 基于关键词重叠
        question_keywords = set(re.findall(r'\w+', question.lower()))
        context_keywords = set(re.findall(r'\w+', original_context.lower()))
        
        # 计算每个检索chunk与原始context的重叠度
        relevance_scores = []
        for chunk in retrieved_chunks:
            chunk_keywords = set(re.findall(r'\w+', chunk.lower()))
            
            # 计算与原始context的重叠度
            context_overlap = len(chunk_keywords & context_keywords) / max(len(context_keywords), 1)
            
            # 计算与问题的重叠度
            question_overlap = len(chunk_keywords & question_keywords) / max(len(question_keywords), 1)
            
            # 综合分数
            relevance = (context_overlap * 0.7 + question_overlap * 0.3)
            relevance_scores.append(relevance)
        
        return float(np.mean(relevance_scores)) if relevance_scores else 0.0
    
    def _calculate_generation_accuracy(self, generated_answer: str, 
                                     reference_answer: str) -> float:
        """计算生成准确性分数"""
        if not reference_answer or not generated_answer:
            return 0.0
        
        # 清理生成的答案
        cleaned_answer = self._clean_generated_answer(generated_answer)
        
        # 方法1: 基于关键词匹配
        ref_keywords = set(re.findall(r'\w+', reference_answer.lower()))
        gen_keywords = set(re.findall(r'\w+', cleaned_answer.lower()))
        
        if not ref_keywords:
            return 0.0
        
        # 计算关键词重叠率
        overlap = len(ref_keywords & gen_keywords) / len(ref_keywords)
        
        # 方法2: 基于数字匹配（对金融数据特别重要）
        ref_numbers = set(re.findall(r'\d+\.?\d*', reference_answer))
        gen_numbers = set(re.findall(r'\d+\.?\d*', cleaned_answer))
        
        number_accuracy = 0.0
        if ref_numbers:
            number_overlap = len(ref_numbers & gen_numbers) / len(ref_numbers)
            number_accuracy = number_overlap * 0.5  # 数字准确性权重
        
        # 综合分数
        accuracy = overlap * 0.5 + number_accuracy
        return min(accuracy, 1.0)
    
    def _clean_generated_answer(self, answer: str) -> str:
        """清理生成的答案"""
        # 移除前缀
        prefixes_to_remove = [
            "[Reranker: Enabled]",
            "Based on the passage above",
            "Based on the context",
            "According to the passage",
            "The passage shows that"
        ]
        
        cleaned = answer
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # 移除后缀（从第一个换行符开始）
        if "\n" in cleaned:
            cleaned = cleaned.split("\n")[0].strip()
        
        # 移除额外的指令
        if "Based on the context" in cleaned:
            cleaned = cleaned.split("Based on the context")[0].strip()
        
        return cleaned
    
    def evaluate_sample(self, sample: Dict) -> EvaluationResult:
        """评估单个样本"""
        question = sample['question']
        context = sample['context']
        answer = sample['answer']
        language = sample['language']
        
        print(f"\n评估问题: {question[:50]}...")
        
        try:
            # 运行RAG系统
            rag_answer, context_data, _ = self.ui._process_question(
                question=question,
                datasource="Both",
                reranker_checkbox=True
            )
            
            # 提取检索结果
            retrieved_chunks = [ctx for _, ctx in context_data]
            retrieved_scores = [float(score) for score, _ in context_data]
            
            # 计算检索相关性
            retrieval_relevance = self._calculate_retrieval_relevance(
                question, retrieved_chunks, context
            )
            
            # 计算生成准确性
            generation_accuracy = self._calculate_generation_accuracy(
                rag_answer, answer
            )
            
            # 判断好坏
            is_retrieval_good = retrieval_relevance > 0.3
            is_generation_good = generation_accuracy > 0.4
            
            return EvaluationResult(
                question=question,
                context=context,
                answer=answer,
                retrieved_chunks=retrieved_chunks,
                retrieved_scores=retrieved_scores,
                generated_answer=rag_answer,
                retrieval_relevance=retrieval_relevance,
                generation_accuracy=generation_accuracy,
                is_retrieval_good=is_retrieval_good,
                is_generation_good=is_generation_good,
                prompt_type="optimized"
            )
            
        except Exception as e:
            print(f"评估失败: {e}")
            return EvaluationResult(
                question=question,
                context=context,
                answer=answer,
                retrieved_chunks=[],
                retrieved_scores=[],
                generated_answer="",
                retrieval_relevance=0.0,
                generation_accuracy=0.0,
                is_retrieval_good=False,
                is_generation_good=False,
                prompt_type="optimized"
            )
    
    def evaluate_all(self, max_samples: int = 10) -> Dict:
        """评估所有样本"""
        print(f"开始评估 {min(len(self.eval_data), max_samples)} 个样本...")
        
        results = []
        retrieval_good_count = 0
        generation_good_count = 0
        both_good_count = 0
        
        for i, sample in enumerate(self.eval_data[:max_samples]):
            print(f"进度: {i+1}/{min(len(self.eval_data), max_samples)}")
            
            result = self.evaluate_sample(sample)
            results.append(result)
            
            if result.is_retrieval_good:
                retrieval_good_count += 1
            if result.is_generation_good:
                generation_good_count += 1
            if result.is_retrieval_good and result.is_generation_good:
                both_good_count += 1
        
        # 计算统计信息
        total = len(results)
        avg_retrieval_relevance = np.mean([r.retrieval_relevance for r in results])
        avg_generation_accuracy = np.mean([r.generation_accuracy for r in results])
        
        # 分析问题分布
        retrieval_only_good = retrieval_good_count - both_good_count
        generation_only_good = generation_good_count - both_good_count
        both_bad = total - retrieval_good_count - generation_good_count + both_good_count
        
        return {
            'total_samples': total,
            'avg_retrieval_relevance': avg_retrieval_relevance,
            'avg_generation_accuracy': avg_generation_accuracy,
            'retrieval_good_rate': retrieval_good_count / total,
            'generation_good_rate': generation_good_count / total,
            'both_good_rate': both_good_count / total,
            'problem_distribution': {
                'retrieval_only_good': retrieval_only_good,
                'generation_only_good': generation_only_good,
                'both_good': both_good_count,
                'both_bad': both_bad
            },
            'detailed_results': results
        }
    
    def print_analysis(self, results: Dict):
        """打印分析结果"""
        print("\n" + "="*60)
        print("优化RAG系统性能评估结果 (Qwen3-8B)")
        print("="*60)
        
        print(f"总样本数: {results['total_samples']}")
        print(f"平均检索相关性: {results['avg_retrieval_relevance']:.3f}")
        print(f"平均生成准确性: {results['avg_generation_accuracy']:.3f}")
        print(f"检索良好率: {results['retrieval_good_rate']:.3f}")
        print(f"生成良好率: {results['generation_good_rate']:.3f}")
        print(f"整体良好率: {results['both_good_rate']:.3f}")
        
        print("\n问题分布分析:")
        dist = results['problem_distribution']
        print(f"  - 仅检索良好: {dist['retrieval_only_good']} ({dist['retrieval_only_good']/results['total_samples']:.1%})")
        print(f"  - 仅生成良好: {dist['generation_only_good']} ({dist['generation_only_good']/results['total_samples']:.1%})")
        print(f"  - 整体良好: {dist['both_good']} ({dist['both_good']/results['total_samples']:.1%})")
        print(f"  - 整体不良: {dist['both_bad']} ({dist['both_bad']/results['total_samples']:.1%})")
        
        # 给出建议
        print("\n诊断建议:")
        if dist['retrieval_only_good'] > dist['generation_only_good']:
            print("  → 主要问题在生成器，建议进一步优化prompt或参数")
        elif dist['generation_only_good'] > dist['retrieval_only_good']:
            print("  → 主要问题在检索器，建议优化embedding模型或检索策略")
        else:
            print("  → 检索器和生成器都有问题，需要全面优化")
        
        if dist['both_bad'] > results['total_samples'] * 0.5:
            print("  → 系统整体性能较差，建议重新设计架构")
        else:
            print("  → 系统性能有所改善，继续优化")
    
    def save_detailed_results(self, results: Dict, filename: str = "optimized_rag_evaluation_results.json"):
        """保存详细结果"""
        # 转换为可序列化的格式
        serializable_results = {
            'summary': {
                'total_samples': results['total_samples'],
                'avg_retrieval_relevance': float(results['avg_retrieval_relevance']),
                'avg_generation_accuracy': float(results['avg_generation_accuracy']),
                'retrieval_good_rate': float(results['retrieval_good_rate']),
                'generation_good_rate': float(results['generation_good_rate']),
                'both_good_rate': float(results['both_good_rate']),
                'problem_distribution': results['problem_distribution']
            },
            'detailed_results': [
                {
                    'question': r.question,
                    'context': r.context,
                    'answer': r.answer,
                    'retrieved_chunks': r.retrieved_chunks,
                    'retrieved_scores': r.retrieved_scores,
                    'generated_answer': r.generated_answer,
                    'retrieval_relevance': float(r.retrieval_relevance),
                    'generation_accuracy': float(r.generation_accuracy),
                    'is_retrieval_good': r.is_retrieval_good,
                    'is_generation_good': r.is_generation_good,
                    'prompt_type': r.prompt_type
                }
                for r in results['detailed_results']
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细结果已保存到: {filename}")

def main():
    """主函数"""
    print("开始优化RAG系统性能评估...")
    print("模型: Qwen3-8B-Instruct")
    print("优化: 优化参数 (temperature=0.1, max_tokens=100)")
    
    # 创建评估器
    evaluator = OptimizedRAGEvaluator()
    
    # 运行评估
    results = evaluator.evaluate_all(max_samples=10)  # 评估10个样本
    
    # 打印分析结果
    evaluator.print_analysis(results)
    
    # 保存详细结果
    evaluator.save_detailed_results(results)
    
    print("\n评估完成！")

if __name__ == "__main__":
    main() 