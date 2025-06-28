#!/usr/bin/env python3
"""
使用AlphaFin数据集中的问题比较不同模型
支持通过--model_name参数指定不同的模型
"""

import os
import sys
import json
import torch
import time
import argparse
import random
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.components.generator.local_llm_generator import LocalLLMGenerator


def load_alphafin_questions(data_path: str, max_questions: int = 10) -> List[str]:
    """从AlphaFin数据集加载问题，支持jsonl和json数组格式，优先使用generated_question字段"""
    questions = []
    try:
        # 判断是否为json数组
        with open(data_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # JSON数组格式
                data_list = json.load(f)
                # 优先使用generated_question字段
                all_questions = [item.get('generated_question') for item in data_list if item.get('generated_question')]
                # 如果不足max_questions，尝试用original_question补充
                if len(all_questions) < max_questions:
                    all_questions += [item.get('original_question') for item in data_list if item.get('original_question')]
                # 随机抽取max_questions个
                questions = random.sample(all_questions, min(max_questions, len(all_questions)))
            else:
                # JSONL格式
                for i, line in enumerate(f):
                    if i >= max_questions:
                        break
                    try:
                        data = json.loads(line.strip())
                        if 'generated_question' in data:
                            questions.append(data['generated_question'])
                        elif 'question' in data:
                            questions.append(data['question'])
                        elif 'query' in data:
                            questions.append(data['query'])
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"❌ 加载AlphaFin数据失败: {e}")
        # 使用默认问题作为备选
        questions = [
            "什么是股票投资？",
            "请解释债券的基本概念",
            "基金投资与股票投资有什么区别？",
            "什么是市盈率？",
            "请解释什么是ETF基金"
        ]
    print(f"✅ 加载了 {len(questions)} 个问题")
    return questions


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU内存已清理")


def test_model_with_alphafin_questions(
    model_name: str, 
    questions: List[str], 
    device: str = "cuda:1",
    max_new_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.8
) -> Dict[str, Any]:
    """使用AlphaFin问题测试指定模型"""
    print(f"\n🚀 开始测试模型: {model_name}")
    print(f"   设备: {device}")
    print(f"   问题数量: {len(questions)}")
    
    # 清理内存
    clear_gpu_memory()
    
    results = {
        "model_name": model_name,
        "device": device,
        "questions": [],
        "success_count": 0,
        "total_time": 0,
        "avg_tokens": 0,
        "memory_usage": 0,
        "config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
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
        for i, question in enumerate(questions):
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


def compare_multiple_models(
    model_names: List[str], 
    questions: List[str],
    device: str = "cuda:1"
) -> Dict[str, Dict[str, Any]]:
    """比较多个模型的结果"""
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*20} 测试 {model_name} {'='*20}")
        results = test_model_with_alphafin_questions(
            model_name=model_name,
            questions=questions,
            device=device
        )
        all_results[model_name] = results
        
        # 保存单个模型结果
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        save_results(results, f"{safe_model_name}_alphafin_results.json")
        
        # 等待用户确认是否继续
        if model_name != model_names[-1]:  # 不是最后一个模型
            try:
                choice = input(f"\n🤔 是否继续测试下一个模型？(y/n): ").lower().strip()
                if choice not in ['y', 'yes', '是']:
                    print("👋 用户中断测试")
                    break
            except KeyboardInterrupt:
                print("\n👋 用户中断测试")
                break
    
    return all_results


def generate_comparison_report(all_results: Dict[str, Dict[str, Any]], output_file: str = "model_comparison_report.md"):
    """生成比较报告"""
    print(f"\n📊 生成比较报告...")
    
    report = f"""# 模型比较报告 - AlphaFin数据集

## 📋 测试概述

- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 测试问题数量: {len(all_results[list(all_results.keys())[0]]['questions']) if all_results else 0}
- 测试模型数量: {len(all_results)}

## 📈 性能对比

| 模型 | 成功率 | 平均时间(s) | 平均Token数 | GPU内存(GB) | 状态 |
|------|--------|-------------|-------------|-------------|------|
"""
    
    for model_name, results in all_results.items():
        success_count = results.get('success_count', 0)
        total_questions = len(results.get('questions', []))
        success_rate = f"{success_count}/{total_questions} ({success_count/total_questions*100:.1f}%)" if total_questions > 0 else "0/0 (0%)"
        
        avg_time = results.get('total_time', 0) / success_count if success_count > 0 else 0
        avg_tokens = results.get('avg_tokens', 0)
        memory_usage = results.get('memory_usage', 0)
        
        status = "✅ 成功" if 'error' not in results else "❌ 失败"
        
        report += f"| {model_name} | {success_rate} | {avg_time:.2f} | {avg_tokens:.1f} | {memory_usage:.2f} | {status} |\n"
    
    report += f"""
## 📝 详细结果

"""
    
    for model_name, results in all_results.items():
        report += f"### {model_name}\n\n"
        
        if 'error' in results:
            report += f"**错误**: {results['error']}\n\n"
        else:
            report += f"- **成功率**: {results['success_count']}/{len(results['questions'])}\n"
            report += f"- **平均时间**: {results['total_time']/results['success_count']:.2f}s\n"
            report += f"- **平均Token数**: {results['avg_tokens']:.1f}\n"
            report += f"- **GPU内存**: {results['memory_usage']:.2f}GB\n\n"
            
            report += "**示例回答**:\n\n"
            for i, q_result in enumerate(results['questions'][:3]):  # 只显示前3个
                if q_result['success']:
                    report += f"{i+1}. **问题**: {q_result['question']}\n"
                    report += f"   **回答**: {q_result['response'][:200]}...\n\n"
    
    report += f"""
## 🎯 总结

"""
    
    # 找出最佳模型
    best_model = None
    best_score = 0
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            score = results['success_count'] / len(results['questions'])
            if score > best_score:
                best_score = score
                best_model = model_name
    
    if best_model:
        report += f"- **最佳模型**: {best_model} (成功率: {best_score*100:.1f}%)\n"
    
    report += f"- **测试完成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 比较报告已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用AlphaFin数据集比较不同模型")
    parser.add_argument("--model_names", nargs="+", 
                       default=["Qwen/Qwen3-8B", "SUFE-AIFLM-Lab/Fin-R1"],
                       help="要测试的模型名称列表")
    parser.add_argument("--data_path", type=str, 
                       default="evaluate_mrr/alphafin_train_qc.jsonl",
                       help="AlphaFin数据文件路径")
    parser.add_argument("--max_questions", type=int, default=5,
                       help="最大测试问题数量")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="GPU设备")
    parser.add_argument("--output_dir", type=str, default="model_comparison_results",
                       help="输出目录")
    
    args = parser.parse_args()
    
    print("🧪 AlphaFin数据集模型比较测试")
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
        args.device = "cpu"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载AlphaFin问题
    print(f"\n📚 加载AlphaFin问题...")
    questions = load_alphafin_questions(args.data_path, args.max_questions)
    
    # 显示问题
    print(f"\n📝 测试问题:")
    for i, question in enumerate(questions):
        print(f"   {i+1}. {question}")
    
    # 比较模型
    print(f"\n🔍 开始比较模型: {args.model_names}")
    all_results = compare_multiple_models(
        model_names=args.model_names,
        questions=questions,
        device=args.device
    )
    
    # 生成比较报告
    report_path = os.path.join(args.output_dir, "model_comparison_report.md")
    generate_comparison_report(all_results, report_path)
    
    print(f"\n🎉 模型比较测试完成！")
    print(f"📁 结果文件保存在: {args.output_dir}")


if __name__ == "__main__":
    main() 