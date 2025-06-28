#!/usr/bin/env python3
"""
RAG系统扰动策略实验
集成现有的扰动系统，对RAG的检索和生成阶段进行可解释性分析
不使用LLM-based扰动器，专注于其他扰动策略
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.enhanced_retriever import EnhancedRetriever
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.explainer.generic_retriever_explainer import GenericRetrieverExplainer
from xlm.explainer.generic_generator_explainer import GenericGeneratorExplainer
from xlm.modules.perturber.leave_one_out_perturber import LeaveOneOutPerturber
from xlm.modules.perturber.reorder_perturber import ReorderPerturber
from xlm.modules.perturber.trend_perturber import TrendPerturber
from xlm.modules.perturber.year_perturber import YearPerturber
from xlm.modules.perturber.term_perturber import TermPerturber
from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer
from xlm.dto.dto import ExplanationGranularity
from config.parameters import Config

class RAGPerturbationExperiment:
    """RAG系统扰动实验类"""
    
    def __init__(self):
        """初始化实验环境"""
        print("🔬 初始化RAG扰动实验环境...")
        
        # 加载配置
        self.config = Config()
        
        # 初始化RAG系统组件
        self.generator = LocalLLMGenerator()
        self.retriever = EnhancedRetriever(config=self.config)
        
        # 初始化扰动系统组件
        self.tokenizer = CustomTokenizer()
        
        # 为比较器提供encoder
        from xlm.components.encoder.finbert import FinbertEncoder
        encoder = FinbertEncoder(model_name="ProsusAI/finbert")
        self.comparator = EmbeddingComparator(encoder=encoder)
        
        # 初始化多种扰动器
        self.perturbers = {
            'leave_one_out': LeaveOneOutPerturber(),
            'reorder': ReorderPerturber(),
            'trend': TrendPerturber(),
            'year': YearPerturber(),
            'term': TermPerturber()
        }
        
        print("✅ 实验环境初始化完成")
        print(f"📊 可用的扰动器: {list(self.perturbers.keys())}")
    
    def run_perturbation_experiment(self, question: str, perturber_name: str, stage: str = 'retrieval'):
        """运行特定扰动器的实验"""
        print(f"\n🔬 {stage.upper()} 阶段 - {perturber_name} 扰动实验")
        print(f"问题: {question}")
        print("=" * 60)
        
        try:
            perturber = self.perturbers[perturber_name]
            
            if stage == 'retrieval':
                # 检索阶段扰动实验
                retriever_explainer = GenericRetrieverExplainer(
                    perturber=perturber,
                    comparator=self.comparator,
                    retriever=self.retriever,
                    tokenizer=self.tokenizer
                )
                
                # 获取参考检索结果
                reference_doc, reference_score = retriever_explainer.get_reference(question)
                print(f"参考文档: {reference_doc[:100]}...")
                print(f"参考分数: {reference_score:.4f}")
                
                # 进行扰动分析
                explanation = retriever_explainer.explain(
                    user_input=question,
                    granularity=ExplanationGranularity.WORD_LEVEL,
                    reference_text=reference_doc,
                    reference_score=str(reference_score)
                )
                
            else:  # generation stage
                # 生成阶段扰动实验
                generator_explainer = GenericGeneratorExplainer(
                    perturber=perturber,
                    comparator=self.comparator,
                    generator=self.generator,
                    tokenizer=self.tokenizer
                )
                
                # 先获取上下文
                retrieved_docs, _ = self.retriever.retrieve(text=question, top_k=3, return_scores=True)
                context = "\n\n".join([doc.content for doc in retrieved_docs])
                
                # 构建完整prompt
                from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH_CLEAN
                full_prompt = PROMPT_TEMPLATE_ZH_CLEAN.format(context=context, question=question)
                
                # 获取参考生成结果
                reference_response = generator_explainer.get_reference(full_prompt)
                print(f"参考回答: {reference_response[:100]}...")
                
                # 进行扰动分析
                explanation = generator_explainer.explain(
                    user_input=full_prompt,
                    granularity=ExplanationGranularity.WORD_LEVEL,
                    reference_text=reference_response,
                    reference_score="1.0"
                )
            
            # 显示结果
            print(f"\n📊 {perturber_name} 扰动分析结果:")
            print(f"分析的特征数量: {len(explanation.explanations)}")
            
            # 显示最重要的特征
            top_features = explanation.explanations[:5]
            print(f"\n🏆 Top 5 重要特征:")
            for i, feature in enumerate(top_features, 1):
                print(f"{i}. '{feature.feature}' - 重要性: {feature.score:.4f}")
            
            return explanation
            
        except Exception as e:
            print(f"❌ {perturber_name} 扰动实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_comprehensive_perturbation_experiment(self, question: str):
        """运行全面的扰动实验，测试所有扰动器"""
        print(f"\n🚀 全面扰动实验")
        print(f"问题: {question}")
        print("=" * 60)
        
        results = {}
        
        # 1. 运行标准RAG作为基准
        try:
            rag_system = RagSystem(
                retriever=self.retriever,
                generator=self.generator,
                retriever_top_k=5
            )
            
            standard_result = rag_system.run(question)
            print(f"✅ 标准RAG运行成功")
            print(f"检索文档数: {len(standard_result.retrieved_documents)}")
            print(f"生成回答: {standard_result.generated_responses[0][:100]}...")
            results['standard_rag'] = standard_result
            
        except Exception as e:
            print(f"❌ 标准RAG运行失败: {str(e)}")
            results['standard_rag'] = None
        
        # 2. 测试所有扰动器在检索阶段的效果
        print(f"\n🔍 检索阶段扰动实验...")
        retrieval_results = {}
        for perturber_name in self.perturbers.keys():
            print(f"\n--- 测试 {perturber_name} ---")
            result = self.run_perturbation_experiment(question, perturber_name, 'retrieval')
            retrieval_results[perturber_name] = result
        
        results['retrieval_perturbations'] = retrieval_results
        
        # 3. 测试所有扰动器在生成阶段的效果
        print(f"\n🤖 生成阶段扰动实验...")
        generation_results = {}
        for perturber_name in self.perturbers.keys():
            print(f"\n--- 测试 {perturber_name} ---")
            result = self.run_perturbation_experiment(question, perturber_name, 'generation')
            generation_results[perturber_name] = result
        
        results['generation_perturbations'] = generation_results
        
        # 4. 总结实验结果
        print(f"\n📋 实验总结:")
        print(f"✅ 标准RAG: {'成功' if results['standard_rag'] else '失败'}")
        
        successful_retrieval = sum(1 for r in retrieval_results.values() if r is not None)
        successful_generation = sum(1 for r in generation_results.values() if r is not None)
        
        print(f"✅ 检索扰动: {successful_retrieval}/{len(self.perturbers)} 成功")
        print(f"✅ 生成扰动: {successful_generation}/{len(self.perturbers)} 成功")
        
        return results

def main():
    """主函数"""
    print("🧪 RAG系统扰动策略实验")
    print("=" * 60)
    
    # 创建实验实例
    experiment = RAGPerturbationExperiment()
    
    # 测试问题
    test_questions = [
        "首钢股份在2023年上半年的业绩表现如何？",
        "中国平安的财务状况怎么样？",
        "腾讯控股的游戏业务发展如何？"
    ]
    
    # 运行实验
    all_results = {}
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} 实验 {i} {'='*20}")
        results = experiment.run_comprehensive_perturbation_experiment(question)
        all_results[f'experiment_{i}'] = results
        
        if results:
            print(f"✅ 实验 {i} 完成")
        else:
            print(f"❌ 实验 {i} 失败")
    
    print(f"\n🎉 所有实验完成！")
    print("📊 扰动系统已成功集成到RAG系统中")
    print("🔬 可以进行可解释性分析和扰动策略研究")
    
    # 显示可用扰动器
    print(f"\n📋 使用的扰动策略:")
    for perturber_name in experiment.perturbers.keys():
        print(f"  - {perturber_name}")
    
    print(f"\n💡 实验建议:")
    print(f"  - 可以分析不同扰动策略的效果差异")
    print(f"  - 可以识别RAG系统的关键特征")
    print(f"  - 可以评估系统的鲁棒性")

if __name__ == "__main__":
    main() 