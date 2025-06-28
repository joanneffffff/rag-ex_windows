#!/usr/bin/env python3
"""
Prompt扰动 vs Context扰动实验
展示扰动在RAG系统中的不同位置和效果
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

class PromptVsContextPerturbation:
    """Prompt扰动 vs Context扰动对比实验"""
    
    def __init__(self):
        """初始化实验环境"""
        print("🔬 初始化Prompt vs Context扰动实验...")
        
        # 加载配置
        self.config = Config()
        
        # 初始化RAG系统组件
        self.generator = LocalLLMGenerator()
        self.retriever = EnhancedRetriever(config=self.config)
        
        # 初始化扰动系统组件
        self.tokenizer = CustomTokenizer()
        from xlm.components.encoder.finbert import FinbertEncoder
        encoder = FinbertEncoder(model_name="ProsusAI/finbert")
        self.comparator = EmbeddingComparator(encoder=encoder)
        
        # 初始化扰动器
        self.perturber = LeaveOneOutPerturber()
        
        print("✅ 实验环境初始化完成")
    
    def run_context_perturbation_experiment(self, question: str):
        """运行Context扰动实验（检索阶段）"""
        print(f"\n🔍 Context扰动实验（检索阶段）")
        print(f"问题: {question}")
        print("=" * 60)
        
        try:
            # 创建检索器解释器
            retriever_explainer = GenericRetrieverExplainer(
                perturber=self.perturber,
                comparator=self.comparator,
                retriever=self.retriever,
                tokenizer=self.tokenizer
            )
            
            # 获取参考检索结果
            reference_doc, reference_score = retriever_explainer.get_reference(question)
            print(f"📄 原始文档: {reference_doc[:100]}...")
            print(f"📊 原始分数: {reference_score:.4f}")
            
            # 对文档内容进行分词
            features = self.tokenizer.tokenize(text=reference_doc, granularity=ExplanationGranularity.WORD_LEVEL)
            print(f"🔤 文档特征: {features[:10]}...")  # 显示前10个特征
            
            # 生成扰动文档
            perturbations = self.perturber.perturb(text=reference_doc, features=features)
            print(f"🔄 生成扰动文档数量: {len(perturbations)}")
            
            # 显示几个扰动示例
            for i, perturbed in enumerate(perturbations[:3], 1):
                print(f"扰动{i}: {perturbed[:80]}...")
            
            # 进行扰动分析
            explanation = retriever_explainer.explain(
                user_input=question,
                granularity=ExplanationGranularity.WORD_LEVEL,
                reference_text=reference_doc,
                reference_score=str(reference_score)
            )
            
            # 显示结果
            print(f"\n📊 Context扰动分析结果:")
            print(f"分析的特征数量: {len(explanation.explanations)}")
            
            if explanation.explanations:
                print(f"\n🏆 Top 5 重要文档特征:")
                for i, feature in enumerate(explanation.explanations[:5], 1):
                    print(f"{i}. '{feature.feature}' - 重要性: {feature.score:.4f}")
            
            return explanation
            
        except Exception as e:
            print(f"❌ Context扰动实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_prompt_perturbation_experiment(self, question: str):
        """运行Prompt扰动实验（生成阶段）"""
        print(f"\n🤖 Prompt扰动实验（生成阶段）")
        print(f"问题: {question}")
        print("=" * 60)
        
        try:
            # 先获取上下文
            retrieved_docs, _ = self.retriever.retrieve(text=question, top_k=3, return_scores=True)
            context = "\n\n".join([doc.content for doc in retrieved_docs])
            
            # 构建完整prompt
            from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH_CLEAN
            full_prompt = PROMPT_TEMPLATE_ZH_CLEAN.format(context=context, question=question)
            
            print(f"📝 完整Prompt: {full_prompt[:200]}...")
            
            # 创建生成器解释器
            generator_explainer = GenericGeneratorExplainer(
                perturber=self.perturber,
                comparator=self.comparator,
                generator=self.generator,
                tokenizer=self.tokenizer
            )
            
            # 获取参考生成结果
            reference_response = generator_explainer.get_reference(full_prompt)
            print(f"📄 原始回答: {reference_response[:100]}...")
            
            # 对prompt进行分词
            features = self.tokenizer.tokenize(text=full_prompt, granularity=ExplanationGranularity.WORD_LEVEL)
            print(f"🔤 Prompt特征: {features[:10]}...")  # 显示前10个特征
            
            # 生成扰动prompt
            perturbations = self.perturber.perturb(text=full_prompt, features=features)
            print(f"🔄 生成扰动Prompt数量: {len(perturbations)}")
            
            # 显示几个扰动示例
            for i, perturbed in enumerate(perturbations[:3], 1):
                print(f"扰动{i}: {perturbed[:80]}...")
            
            # 进行扰动分析
            explanation = generator_explainer.explain(
                user_input=full_prompt,
                granularity=ExplanationGranularity.WORD_LEVEL,
                reference_text=reference_response,
                reference_score="1.0"
            )
            
            # 显示结果
            print(f"\n📊 Prompt扰动分析结果:")
            print(f"分析的特征数量: {len(explanation.explanations)}")
            
            if explanation.explanations:
                print(f"\n🏆 Top 5 重要Prompt特征:")
                for i, feature in enumerate(explanation.explanations[:5], 1):
                    print(f"{i}. '{feature.feature}' - 重要性: {feature.score:.4f}")
            
            return explanation
            
        except Exception as e:
            print(f"❌ Prompt扰动实验失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_comparison_experiment(self, question: str):
        """运行对比实验"""
        print(f"\n🚀 Prompt vs Context扰动对比实验")
        print(f"问题: {question}")
        print("=" * 60)
        
        # 1. 运行标准RAG
        try:
            rag_system = RagSystem(
                retriever=self.retriever,
                generator=self.generator,
                retriever_top_k=5
            )
            
            result = rag_system.run(question)
            print(f"✅ 标准RAG运行成功")
            print(f"检索文档数: {len(result.retrieved_documents)}")
            print(f"生成回答: {result.generated_responses[0][:100]}...")
            
        except Exception as e:
            print(f"❌ 标准RAG运行失败: {str(e)}")
        
        # 2. Context扰动实验
        context_explanation = self.run_context_perturbation_experiment(question)
        
        # 3. Prompt扰动实验
        prompt_explanation = self.run_prompt_perturbation_experiment(question)
        
        # 4. 对比分析
        print(f"\n📋 对比分析:")
        print(f"✅ Context扰动: {'成功' if context_explanation else '失败'}")
        print(f"✅ Prompt扰动: {'成功' if prompt_explanation else '失败'}")
        
        if context_explanation and prompt_explanation:
            print(f"\n🔍 扰动位置对比:")
            print(f"  - Context扰动: 分析文档内容的重要性")
            print(f"  - Prompt扰动: 分析完整prompt的重要性")
            print(f"  - 两者结合: 全面理解RAG系统的关键特征")
        
        return {
            'context_explanation': context_explanation,
            'prompt_explanation': prompt_explanation
        }

def main():
    """主函数"""
    print("🧪 Prompt vs Context扰动实验")
    print("=" * 60)
    
    # 创建实验实例
    experiment = PromptVsContextPerturbation()
    
    # 测试问题
    test_questions = [
        "首钢股份在2023年上半年的业绩表现如何？",
        "中国平安的财务状况怎么样？"
    ]
    
    # 运行实验
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*20} 实验 {i} {'='*20}")
        results = experiment.run_comparison_experiment(question)
        
        if results:
            print(f"✅ 实验 {i} 完成")
        else:
            print(f"❌ 实验 {i} 失败")
    
    print(f"\n🎉 所有实验完成！")
    print("📊 扰动位置分析:")
    print("  - Context扰动: 分析检索阶段文档内容的重要性")
    print("  - Prompt扰动: 分析生成阶段prompt内容的重要性")
    print("🔬 可以全面理解RAG系统的可解释性")

if __name__ == "__main__":
    main() 