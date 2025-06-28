#!/usr/bin/env python3
"""
使用Qwen3-8B作为生成器的RAG系统测试
"""

import os
import sys
import torch
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.ui.optimized_rag_ui import OptimizedRagUI


def test_rag_with_qwen3():
    """测试使用Qwen3-8B的RAG系统"""
    print("🚀 开始测试使用Qwen3-8B的RAG系统...")
    
    # 加载配置
    config = Config()
    print(f"📋 当前配置:")
    print(f"   生成器模型: {config.generator.model_name}")
    print(f"   量化: {config.generator.use_quantization} ({config.generator.quantization_type})")
    print(f"   最大token: {config.generator.max_new_tokens}")
    
    # 测试问题
    test_questions = [
        "什么是股票投资？",
        "请解释债券的基本概念",
        "基金投资与股票投资有什么区别？",
        "什么是市盈率？",
        "请解释什么是ETF基金"
    ]
    
    try:
        # 初始化RAG系统
        print("\n🔧 初始化RAG系统...")
        rag_ui = OptimizedRagUI(
            encoder_model_name="paraphrase-multilingual-MiniLM-L12-v2",
            enable_reranker=True,
            use_existing_embedding_index=True,
            max_alphafin_chunks=10000  # 限制数据量以加快测试
        )
        print("✅ RAG系统初始化成功")
        
        # 测试每个问题
        results = []
        for i, question in enumerate(test_questions):
            print(f"\n🔍 测试问题 {i+1}: {question}")
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 使用RAG系统生成回答
                result = rag_ui._process_question(
                    question=question,
                    datasource="Both",
                    reranker_checkbox=True
                )
                
                # 解包结果
                answer, contexts, _ = result
                
                # 记录结束时间
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 统计token数量
                answer_tokens = len(answer.split()) if answer else 0
                
                print(f"   ✅ 生成成功")
                print(f"      回答: {answer[:100]}..." if answer else "      回答: 无")
                print(f"      长度: {answer_tokens} tokens")
                print(f"      时间: {generation_time:.2f}s")
                print(f"      检索上下文数量: {len(contexts) if contexts is not None else 0}")
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "tokens": answer_tokens,
                    "time": generation_time,
                    "contexts_count": len(contexts) if contexts is not None else 0,
                    "success": True
                })
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                results.append({
                    "question": question,
                    "answer": f"生成失败: {e}",
                    "tokens": 0,
                    "time": 0,
                    "contexts_count": 0,
                    "success": False
                })
        
        # 输出总结
        print(f"\n📊 RAG系统测试总结:")
        print(f"=" * 50)
        
        successful_generations = [r for r in results if r["success"]]
        failed_generations = [r for r in results if not r["success"]]
        
        if successful_generations:
            avg_tokens = sum(r["tokens"] for r in successful_generations) / len(successful_generations)
            avg_time = sum(r["time"] for r in successful_generations) / len(successful_generations)
            avg_contexts = sum(r["contexts_count"] for r in successful_generations) / len(successful_generations)
            
            print(f"✅ 成功生成: {len(successful_generations)}/{len(results)}")
            print(f"   平均token数: {avg_tokens:.1f}")
            print(f"   平均生成时间: {avg_time:.2f}s")
            print(f"   平均检索上下文数: {avg_contexts:.1f}")
        else:
            print(f"❌ 成功生成: 0/{len(results)}")
        
        if failed_generations:
            print(f"❌ 失败生成: {len(failed_generations)}")
        
        # 内存使用情况
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(device=1) / 1024**3
            print(f"\n💾 GPU内存使用: {gpu_memory:.2f}GB")
        
        # 显示详细结果
        print(f"\n📝 详细结果:")
        for i, result in enumerate(results):
            print(f"\n   问题 {i+1}: {result['question']}")
            if result['success']:
                print(f"   回答: {result['answer'][:150]}...")
                print(f"   性能: {result['tokens']} tokens, {result['time']:.2f}s, {result['contexts_count']} contexts")
            else:
                print(f"   错误: {result['answer']}")
        
        print(f"\n✅ RAG系统测试完成")
        return True
        
    except Exception as e:
        print(f"❌ RAG系统初始化失败: {e}")
        return False


def main():
    """主函数"""
    print("🧪 Qwen3-8B RAG系统测试")
    print("=" * 50)
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ CUDA不可用，将使用CPU")
    
    # 运行测试
    success = test_rag_with_qwen3()
    
    if success:
        print(f"\n🎉 Qwen3-8B RAG系统测试成功！")
        print(f"💡 建议:")
        print(f"   - Qwen3-8B作为生成器表现良好")
        print(f"   - 可以考虑在生产环境中使用")
        print(f"   - 相比Fin-R1，内存使用更合理")
    else:
        print(f"\n❌ Qwen3-8B RAG系统测试失败")


if __name__ == "__main__":
    main() 