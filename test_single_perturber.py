#!/usr/bin/env python3
"""
测试单个扰动器的效果
用于快速验证扰动系统是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.enhanced_retriever import EnhancedRetriever
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from xlm.explainer.generic_retriever_explainer import GenericRetrieverExplainer
from xlm.modules.perturber.leave_one_out_perturber import LeaveOneOutPerturber
from xlm.modules.comparator.embedding_comparator import EmbeddingComparator
from xlm.modules.tokenizer.custom_tokenizer import CustomTokenizer
from xlm.dto.dto import ExplanationGranularity
from config.parameters import Config

def test_single_perturber():
    """测试单个扰动器（Leave-One-Out）"""
    print("🧪 测试单个扰动器效果")
    print("=" * 50)
    
    try:
        # 1. 初始化组件
        print("🔧 初始化组件...")
        config = Config()
        
        # RAG组件
        generator = LocalLLMGenerator()
        retriever = EnhancedRetriever(config=config)
        
        # 扰动系统组件
        tokenizer = CustomTokenizer()
        from xlm.components.encoder.finbert import FinbertEncoder
        encoder = FinbertEncoder(model_name="ProsusAI/finbert")
        comparator = EmbeddingComparator(encoder=encoder)
        
        # 扰动器
        perturber = LeaveOneOutPerturber()
        
        print("✅ 组件初始化完成")
        
        # 2. 测试问题
        test_question = "首钢股份在2023年上半年的业绩表现如何？"
        print(f"\n📝 测试问题: {test_question}")
        
        # 3. 运行标准RAG
        print(f"\n🚀 运行标准RAG...")
        rag_system = RagSystem(
            retriever=retriever,
            generator=generator,
            retriever_top_k=5
        )
        
        result = rag_system.run(test_question)
        print(f"✅ 标准RAG运行成功")
        print(f"检索文档数: {len(result.retrieved_documents)}")
        print(f"生成回答: {result.generated_responses[0][:100]}...")
        
        # 4. 运行扰动实验
        print(f"\n🔬 运行Leave-One-Out扰动实验...")
        
        # 创建解释器
        explainer = GenericRetrieverExplainer(
            perturber=perturber,
            comparator=comparator,
            retriever=retriever,
            tokenizer=tokenizer
        )
        
        # 获取参考结果
        reference_doc, reference_score = explainer.get_reference(test_question)
        print(f"参考文档: {reference_doc[:100]}...")
        print(f"参考分数: {reference_score:.4f}")
        
        # 进行扰动分析
        explanation = explainer.explain(
            user_input=test_question,
            granularity=ExplanationGranularity.WORD_LEVEL,
            reference_text=reference_doc,
            reference_score=str(reference_score)
        )
        
        # 5. 显示结果
        print(f"\n📊 扰动分析结果:")
        print(f"分析的特征数量: {len(explanation.explanations)}")
        
        if explanation.explanations:
            print(f"\n🏆 Top 5 重要特征:")
            for i, feature in enumerate(explanation.explanations[:5], 1):
                print(f"{i}. '{feature.feature}' - 重要性: {feature.score:.4f}")
        else:
            print("⚠️ 没有找到重要特征")
        
        print(f"\n🎉 扰动实验成功完成！")
        return True
        
    except Exception as e:
        print(f"❌ 扰动实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_perturber_types():
    """测试不同类型的扰动器"""
    print("\n🔬 测试不同类型的扰动器")
    print("=" * 50)
    
    try:
        # 初始化基础组件
        config = Config()
        tokenizer = CustomTokenizer()
        from xlm.components.encoder.finbert import FinbertEncoder
        encoder = FinbertEncoder(model_name="ProsusAI/finbert")
        comparator = EmbeddingComparator(encoder=encoder)
        
        # 测试文本
        test_text = "首钢股份在2023年上半年的业绩表现不佳"
        test_features = ["首钢股份", "2023年", "上半年", "业绩", "表现"]
        
        # 测试不同扰动器
        perturbers = {
            'leave_one_out': LeaveOneOutPerturber(),
            'random_word': None,  # 需要特殊处理
            'reorder': None,      # 需要特殊处理
        }
        
        print(f"📝 测试文本: {test_text}")
        print(f"📝 测试特征: {test_features}")
        
        # 测试Leave-One-Out扰动器
        print(f"\n🔍 测试Leave-One-Out扰动器...")
        loo_perturber = perturbers['leave_one_out']
        perturbations = loo_perturber.perturb(text=test_text, features=test_features)
        
        print(f"✅ 生成扰动文本数量: {len(perturbations)}")
        for i, perturbed in enumerate(perturbations, 1):
            print(f"{i}. {perturbed}")
        
        print(f"\n🎉 扰动器测试成功完成！")
        return True
        
    except Exception as e:
        print(f"❌ 扰动器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🧪 扰动器测试")
    print("=" * 60)
    
    # 测试单个扰动器
    success1 = test_single_perturber()
    
    # 测试扰动器类型
    success2 = test_perturber_types()
    
    print(f"\n📋 测试总结:")
    print(f"✅ 单个扰动器测试: {'成功' if success1 else '失败'}")
    print(f"✅ 扰动器类型测试: {'成功' if success2 else '失败'}")
    
    if success1 and success2:
        print(f"\n🎉 所有测试通过！扰动系统可以正常使用")
        print(f"💡 可以开始进行完整的扰动策略实验")
    else:
        print(f"\n⚠️ 部分测试失败，需要进一步调试")

if __name__ == "__main__":
    main() 