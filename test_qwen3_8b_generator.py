#!/usr/bin/env python3
"""
测试Qwen3-8B作为生成器的效果
比较与Fin-R1模型的差异
"""

import os
import sys
import torch
from typing import List, Dict, Any
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator


def test_qwen3_8b_generator():
    """测试Qwen3-8B生成器"""
    print("🚀 开始测试Qwen3-8B生成器...")
    
    # 加载配置
    config = Config()
    print(f"📋 当前生成器配置:")
    print(f"   模型: {config.generator.model_name}")
    print(f"   量化: {config.generator.use_quantization} ({config.generator.quantization_type})")
    print(f"   最大token: {config.generator.max_new_tokens}")
    print(f"   温度: {config.generator.temperature}")
    print(f"   Top-p: {config.generator.top_p}")
    
    # 初始化生成器
    print("\n🔧 初始化Qwen3-8B生成器...")
    try:
        generator = LocalLLMGenerator(
            model_name=config.generator.model_name,
            cache_dir=config.generator.cache_dir,
            device="cuda:1",  # 使用GPU 1
            use_quantization=config.generator.use_quantization,
            quantization_type=config.generator.quantization_type
        )
        print("✅ Qwen3-8B生成器初始化成功")
    except Exception as e:
        print(f"❌ Qwen3-8B生成器初始化失败: {e}")
        return False
    
    # 测试问题列表
    test_questions = [
        "什么是股票投资？",
        "请解释债券的基本概念",
        "基金投资与股票投资有什么区别？",
        "什么是市盈率？",
        "请解释什么是ETF基金"
    ]
    
    # 测试prompt模板
    prompt_templates = {
        "simple": "请回答以下问题：{question}",
        "clean": "问题：{question}\n回答：",
        "detailed": "基于金融知识，请详细回答以下问题：{question}\n请提供准确、清晰的解释。"
    }
    
    print(f"\n🧪 开始生成测试...")
    print(f"   测试问题数量: {len(test_questions)}")
    print(f"   Prompt模板数量: {len(prompt_templates)}")
    
    results = {}
    
    for template_name, template in prompt_templates.items():
        print(f"\n📝 测试模板: {template_name}")
        print(f"   模板: {template}")
        
        template_results = []
        
        for i, question in enumerate(test_questions):
            print(f"\n   🔍 问题 {i+1}: {question}")
            
            # 构建prompt
            prompt = template.format(question=question)
            
            try:
                # 记录开始时间
                start_time = time.time()
                
                # 生成回答
                response = generator.generate(
                    texts=[prompt]
                )[0]  # generate方法返回列表，取第一个元素
                
                # 记录结束时间
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 统计token数量
                response_tokens = len(response.split())
                
                print(f"   ✅ 生成成功")
                print(f"      回答: {response[:100]}...")
                print(f"      长度: {response_tokens} tokens")
                print(f"      时间: {generation_time:.2f}s")
                
                template_results.append({
                    "question": question,
                    "response": response,
                    "tokens": response_tokens,
                    "time": generation_time
                })
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                template_results.append({
                    "question": question,
                    "response": f"生成失败: {e}",
                    "tokens": 0,
                    "time": 0
                })
        
        results[template_name] = template_results
    
    # 输出总结
    print(f"\n📊 测试总结:")
    print(f"=" * 50)
    
    for template_name, template_results in results.items():
        print(f"\n📝 模板: {template_name}")
        
        # 计算统计信息
        successful_generations = [r for r in template_results if "生成失败" not in r["response"]]
        failed_generations = [r for r in template_results if "生成失败" in r["response"]]
        
        if successful_generations:
            avg_tokens = sum(r["tokens"] for r in successful_generations) / len(successful_generations)
            avg_time = sum(r["time"] for r in successful_generations) / len(successful_generations)
            
            print(f"   成功生成: {len(successful_generations)}/{len(template_results)}")
            print(f"   平均token数: {avg_tokens:.1f}")
            print(f"   平均生成时间: {avg_time:.2f}s")
        else:
            print(f"   成功生成: 0/{len(template_results)}")
        
        if failed_generations:
            print(f"   失败生成: {len(failed_generations)}")
    
    # 内存使用情况
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated(device=1) / 1024**3
        print(f"\n💾 GPU内存使用: {gpu_memory:.2f}GB")
    
    print(f"\n✅ Qwen3-8B生成器测试完成")
    return True


def compare_with_fin_r1():
    """比较Qwen3-8B与Fin-R1的效果"""
    print("\n🔄 比较Qwen3-8B与Fin-R1的效果...")
    
    # 保存当前配置
    config = Config()
    original_model = config.generator.model_name
    
    # 测试问题
    test_question = "什么是股票投资？"
    test_prompt = f"请回答以下问题：{test_question}"
    
    results = {}
    
    # 测试Qwen3-8B
    print(f"\n📝 测试Qwen3-8B...")
    try:
        config.generator.model_name = "Qwen/Qwen3-8B"
        generator_qwen = LocalLLMGenerator(
            model_name=config.generator.model_name,
            device="cuda:1"
        )
        
        start_time = time.time()
        response_qwen = generator_qwen.generate(texts=[test_prompt])[0]
        time_qwen = time.time() - start_time
        
        results["Qwen3-8B"] = {
            "response": response_qwen,
            "time": time_qwen,
            "tokens": len(response_qwen.split())
        }
        
        print(f"   ✅ Qwen3-8B生成成功")
        print(f"      回答: {response_qwen[:100]}...")
        print(f"      时间: {time_qwen:.2f}s")
        print(f"      Token数: {results['Qwen3-8B']['tokens']}")
        
    except Exception as e:
        print(f"   ❌ Qwen3-8B测试失败: {e}")
        results["Qwen3-8B"] = {"error": str(e)}
    
    # 测试Fin-R1
    print(f"\n📝 测试Fin-R1...")
    try:
        config.generator.model_name = "SUFE-AIFLM-Lab/Fin-R1"
        generator_fin = LocalLLMGenerator(
            model_name=config.generator.model_name,
            device="cuda:1"
        )
        
        start_time = time.time()
        response_fin = generator_fin.generate(texts=[test_prompt])[0]
        time_fin = time.time() - start_time
        
        results["Fin-R1"] = {
            "response": response_fin,
            "time": time_fin,
            "tokens": len(response_fin.split())
        }
        
        print(f"   ✅ Fin-R1生成成功")
        print(f"      回答: {response_fin[:100]}...")
        print(f"      时间: {time_fin:.2f}s")
        print(f"      Token数: {results['Fin-R1']['tokens']}")
        
    except Exception as e:
        print(f"   ❌ Fin-R1测试失败: {e}")
        results["Fin-R1"] = {"error": str(e)}
    
    # 输出比较结果
    print(f"\n📊 模型比较结果:")
    print(f"=" * 50)
    
    if "Qwen3-8B" in results and "Fin-R1" in results:
        if "error" not in results["Qwen3-8B"] and "error" not in results["Fin-R1"]:
            print(f"\n🔍 回答长度比较:")
            print(f"   Qwen3-8B: {results['Qwen3-8B']['tokens']} tokens")
            print(f"   Fin-R1: {results['Fin-R1']['tokens']} tokens")
            
            print(f"\n⏱️ 生成速度比较:")
            print(f"   Qwen3-8B: {results['Qwen3-8B']['time']:.2f}s")
            print(f"   Fin-R1: {results['Fin-R1']['time']:.2f}s")
            
            print(f"\n📝 回答风格比较:")
            print(f"   Qwen3-8B: {results['Qwen3-8B']['response'][:200]}...")
            print(f"   Fin-R1: {results['Fin-R1']['response'][:200]}...")
        else:
            print("❌ 无法比较：至少有一个模型生成失败")
    else:
        print("❌ 无法比较：模型初始化失败")
    
    # 恢复原始配置
    config.generator.model_name = original_model
    print(f"\n✅ 模型比较完成")


if __name__ == "__main__":
    print("🧪 Qwen3-8B生成器测试脚本")
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
    success = test_qwen3_8b_generator()
    
    if success:
        # 询问是否进行比较测试
        try:
            choice = input("\n🤔 是否进行与Fin-R1的比较测试？(y/n): ").lower().strip()
            if choice in ['y', 'yes', '是']:
                compare_with_fin_r1()
        except KeyboardInterrupt:
            print("\n👋 用户中断测试")
    
    print("\n🎉 测试完成！") 