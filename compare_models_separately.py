#!/usr/bin/env python3
"""
分离的模型比较脚本
分别测试Qwen3-8B和Fin-R1，避免内存冲突
"""

import os
import sys
import torch
import time
import json
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU内存已清理")


def test_model_separately(model_name: str, device: str = "cuda:1") -> Dict[str, Any]:
    """单独测试一个模型"""
    print(f"\n🚀 开始测试模型: {model_name}")
    print(f"   设备: {device}")
    
    # 清理内存
    clear_gpu_memory()
    
    # 测试问题
    test_questions = [
        "什么是股票投资？",
        "请解释债券的基本概念", 
        "基金投资与股票投资有什么区别？",
        "什么是市盈率？",
        "请解释什么是ETF基金"
    ]
    
    results = {
        "model_name": model_name,
        "device": device,
        "questions": [],
        "success_count": 0,
        "total_time": 0,
        "avg_tokens": 0,
        "memory_usage": 0
    }
    
    try:
        # 初始化生成器
        print(f"🔧 初始化 {model_name}...")
        generator = LocalLLMGenerator(
            model_name=model_name,
            device=device,
            use_quantization=True,
            quantization_type="4bit"
        )
        print(f"✅ {model_name} 初始化成功")
        
        # 记录内存使用
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated(device=int(device.split(':')[1])) / 1024**3
            results["memory_usage"] = gpu_memory
            print(f"💾 GPU内存使用: {gpu_memory:.2f}GB")
        
        # 测试每个问题
        for i, question in enumerate(test_questions):
            print(f"\n   🔍 问题 {i+1}: {question}")
            
            try:
                # 构建prompt
                prompt = f"请回答以下问题：{question}"
                
                # 记录开始时间
                start_time = time.time()
                
                # 生成回答
                response = generator.generate(texts=[prompt])[0]
                
                # 记录结束时间
                end_time = time.time()
                generation_time = end_time - start_time
                
                # 统计token数量
                response_tokens = len(response.split())
                
                print(f"   ✅ 生成成功")
                print(f"      回答: {response[:100]}...")
                print(f"      长度: {response_tokens} tokens")
                print(f"      时间: {generation_time:.2f}s")
                
                results["questions"].append({
                    "question": question,
                    "response": response,
                    "tokens": response_tokens,
                    "time": generation_time,
                    "success": True
                })
                
                results["success_count"] += 1
                results["total_time"] += generation_time
                
            except Exception as e:
                print(f"   ❌ 生成失败: {e}")
                results["questions"].append({
                    "question": question,
                    "response": f"生成失败: {e}",
                    "tokens": 0,
                    "time": 0,
                    "success": False
                })
        
        # 计算平均token数
        successful_responses = [q for q in results["questions"] if q["success"]]
        if successful_responses:
            results["avg_tokens"] = sum(q["tokens"] for q in successful_responses) / len(successful_responses)
        
        # 清理内存
        del generator
        clear_gpu_memory()
        
    except Exception as e:
        print(f"❌ {model_name} 初始化失败: {e}")
        results["error"] = str(e)
    
    return results


def save_results(results: Dict[str, Any], filename: str):
    """保存测试结果到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存到: {filename}")


def compare_results(qwen_results: Dict[str, Any], fin_results: Dict[str, Any]):
    """比较两个模型的结果"""
    print(f"\n📊 模型比较结果:")
    print(f"=" * 60)
    
    # 基本信息
    print(f"\n📋 模型信息:")
    print(f"   Qwen3-8B: {qwen_results.get('model_name', 'N/A')}")
    print(f"   Fin-R1: {fin_results.get('model_name', 'N/A')}")
    
    # 成功率
    qwen_success = qwen_results.get('success_count', 0)
    fin_success = fin_results.get('success_count', 0)
    total_questions = len(qwen_results.get('questions', []))
    
    print(f"\n✅ 成功率:")
    print(f"   Qwen3-8B: {qwen_success}/{total_questions} ({qwen_success/total_questions*100:.1f}%)")
    print(f"   Fin-R1: {fin_success}/{total_questions} ({fin_success/total_questions*100:.1f}%)")
    
    # 性能指标
    if 'error' not in qwen_results:
        print(f"\n⏱️ 性能指标 (Qwen3-8B):")
        print(f"   平均生成时间: {qwen_results.get('total_time', 0)/qwen_success:.2f}s")
        print(f"   平均token数: {qwen_results.get('avg_tokens', 0):.1f}")
        print(f"   GPU内存使用: {qwen_results.get('memory_usage', 0):.2f}GB")
    
    if 'error' not in fin_results:
        print(f"\n⏱️ 性能指标 (Fin-R1):")
        print(f"   平均生成时间: {fin_results.get('total_time', 0)/fin_success:.2f}s")
        print(f"   平均token数: {fin_results.get('avg_tokens', 0):.1f}")
        print(f"   GPU内存使用: {fin_results.get('memory_usage', 0):.2f}GB")
    
    # 回答质量比较
    if 'error' not in qwen_results and 'error' not in fin_results:
        print(f"\n📝 回答质量比较:")
        for i, (qwen_q, fin_q) in enumerate(zip(qwen_results['questions'], fin_results['questions'])):
            if qwen_q['success'] and fin_q['success']:
                print(f"\n   问题 {i+1}: {qwen_q['question']}")
                print(f"   Qwen3-8B: {qwen_q['response'][:100]}...")
                print(f"   Fin-R1: {fin_q['response'][:100]}...")
                print(f"   长度对比: {qwen_q['tokens']} vs {fin_q['tokens']} tokens")
    
    # 总结
    print(f"\n🎯 总结:")
    if qwen_success > fin_success:
        print(f"   Qwen3-8B 表现更好，成功率更高")
    elif fin_success > qwen_success:
        print(f"   Fin-R1 表现更好，成功率更高")
    else:
        print(f"   两个模型成功率相同")
    
    if 'error' in qwen_results:
        print(f"   Qwen3-8B 存在问题: {qwen_results['error']}")
    if 'error' in fin_results:
        print(f"   Fin-R1 存在问题: {fin_results['error']}")


def main():
    """主函数"""
    print("🧪 分离模型比较测试")
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
        return
    
    # 测试Qwen3-8B
    print(f"\n{'='*20} 测试Qwen3-8B {'='*20}")
    qwen_results = test_model_separately("Qwen/Qwen3-8B", "cuda:1")
    save_results(qwen_results, "qwen3_8b_test_results.json")
    
    # 等待用户确认
    try:
        choice = input(f"\n🤔 是否继续测试Fin-R1？(y/n): ").lower().strip()
        if choice not in ['y', 'yes', '是']:
            print("👋 测试结束")
            return
    except KeyboardInterrupt:
        print("\n👋 用户中断测试")
        return
    
    # 测试Fin-R1
    print(f"\n{'='*20} 测试Fin-R1 {'='*20}")
    fin_results = test_model_separately("SUFE-AIFLM-Lab/Fin-R1", "cuda:1")
    save_results(fin_results, "fin_r1_test_results.json")
    
    # 比较结果
    compare_results(qwen_results, fin_results)
    
    print(f"\n🎉 分离模型比较测试完成！")
    print(f"📁 结果文件:")
    print(f"   - qwen3_8b_test_results.json")
    print(f"   - fin_r1_test_results.json")


if __name__ == "__main__":
    main() 