#!/usr/bin/env python3
"""
测试问题处理功能的调试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.ui.optimized_rag_ui import OptimizedRagUI

def test_question_processing():
    """测试问题处理功能"""
    print("开始初始化UI...")
    
    try:
        # 初始化UI
        ui = OptimizedRagUI()
        print("✅ UI初始化成功")
        
        # 测试问题
        test_questions = [
            "How was internally developed software capitalised?",
            "2019年第四季度利润是多少？",
            "What is the revenue growth rate?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*50}")
            print(f"测试问题 {i}: {question}")
            print(f"{'='*50}")
            
            try:
                # 调用问题处理方法
                answer, context_data, plot = ui._process_question(
                    question=question,
                    datasource="Both",
                    reranker_checkbox=False
                )
                
                print(f"✅ 问题处理成功")
                print(f"答案长度: {len(answer) if answer else 0}")
                print(f"上下文数据条数: {len(context_data) if context_data else 0}")
                print(f"答案前100字符: {answer[:100] if answer else 'None'}...")
                
            except Exception as e:
                print(f"❌ 问题处理失败: {e}")
                print(f"错误类型: {type(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n✅ 所有测试完成")
        
    except Exception as e:
        print(f"❌ UI初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_question_processing() 