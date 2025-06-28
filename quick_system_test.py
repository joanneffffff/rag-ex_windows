#!/usr/bin/env python3
"""
快速验证RAG系统基本功能
用于确认系统可以正常运转，为后续扰动策略实验做准备
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.rag_system.rag_system import RagSystem
from xlm.components.retriever.enhanced_retriever import EnhancedRetriever
from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from config.parameters import Config

def quick_system_test():
    """快速测试RAG系统基本功能"""
    print("🚀 快速验证RAG系统基本功能...")
    
    try:
        # 1. 加载配置
        config = Config()
        print(f"✅ 配置加载成功")
        print(f"   - 生成器模型: {config.generator.model_name}")
        print(f"   - 量化类型: {config.generator.quantization_type}")
        print(f"   - max_new_tokens: {config.generator.max_new_tokens}")
        
        # 2. 加载组件
        print("\n🔧 加载系统组件...")
        generator = LocalLLMGenerator()
        print("✅ 生成器加载成功")
        
        retriever = EnhancedRetriever(config=config)
        print("✅ 检索器加载成功")
        
        # 3. 创建RAG系统
        rag_system = RagSystem(
            retriever=retriever,
            generator=generator,
            retriever_top_k=5
        )
        print("✅ RAG系统创建成功")
        
        # 4. 测试基本功能
        print("\n🧪 测试基本功能...")
        
        # 测试中文问题
        test_question_zh = "首钢股份的业绩如何？"
        print(f"测试问题: {test_question_zh}")
        
        result = rag_system.run(test_question_zh)
        
        print("✅ 系统运行成功！")
        print(f"检索到文档数: {len(result.retrieved_documents)}")
        print(f"生成回答长度: {len(result.generated_responses[0])}")
        print(f"Prompt模板: {result.metadata['prompt_template']}")
        
        # 5. 显示部分结果
        print("\n📝 生成回答预览:")
        print("-" * 40)
        answer = result.generated_responses[0]
        if len(answer) > 200:
            print(answer[:200] + "...")
        else:
            print(answer)
        print("-" * 40)
        
        print("\n🎉 系统验证完成！")
        print("✅ RAG系统可以正常运转")
        print("✅ 适合进行扰动策略实验")
        print("✅ 具备可解释性分析基础")
        
        return True
        
    except Exception as e:
        print(f"❌ 系统测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_system_test()
    if success:
        print("\n🚀 系统已准备就绪，可以开始扰动策略实验！")
    else:
        print("\n⚠️ 系统需要进一步调试") 