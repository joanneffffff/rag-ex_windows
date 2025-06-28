#!/usr/bin/env python3
"""
测试RAG系统修复
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

def test_rag_system():
    """测试RAG系统是否正常工作"""
    print("🔍 测试RAG系统修复")
    print("=" * 50)
    
    try:
        from xlm.components.rag_system.rag_system import RagSystem
        from xlm.components.retriever.retriever import Retriever
        from xlm.components.generator.generator import Generator
        
        print("✅ RAG系统导入成功")
        
        # 测试prompt模板格式化
        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_EN
        
        context = "Apple Inc. reported Q3 2023 revenue of $81.8 billion."
        question = "How did Apple perform in Q3 2023?"
        
        try:
            prompt = PROMPT_TEMPLATE_EN.format(context=context, question=question)
            print("✅ Prompt模板格式化成功")
            print(f"Prompt长度: {len(prompt)} 字符")
        except Exception as e:
            print(f"❌ Prompt模板格式化失败: {e}")
            print(f"Context: {context}")
            print(f"Question: {question}")
            print(f"Template: {PROMPT_TEMPLATE_EN[:100]}...")
            return
        
        # 测试中文prompt模板
        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH_CLEAN
        
        context_zh = "中国平安2023年第一季度实现营业收入2,345.67亿元。"
        question_zh = "中国平安的业绩如何？"
        
        try:
            prompt_zh = PROMPT_TEMPLATE_ZH_CLEAN.format(context=context_zh, question=question_zh)
            print("✅ 中文Prompt模板格式化成功")
            print(f"中文Prompt长度: {len(prompt_zh)} 字符")
        except Exception as e:
            print(f"❌ 中文Prompt模板格式化失败: {e}")
            return
        
        print("\n🎉 RAG系统修复测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_system() 