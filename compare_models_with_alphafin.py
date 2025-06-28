#!/usr/bin/env python3
"""
使用AlphaFin数据集中的问题，测试RAG场景下的Generator LLM效果
模型将接收问题和上下文，并生成回答
"""

import os
import sys
import json
import torch
import time
import argparse
import random
import textwrap
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator


def load_alphafin_qa_pairs(data_path: str, max_questions: int = 10) -> List[Dict[str, str]]:
    """
    从AlphaFin数据集加载问答对，包含问题、上下文和答案。
    支持jsonl和json数组格式，优先使用generated_question字段。
    """
    qa_pairs = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                data_list = json.load(f)
            else: # Assume JSONL format
                data_list = [json.loads(line.strip()) for line in f if line.strip()]
        
        candidates = []
        for item in data_list:
            question_text = item.get('generated_question') # LLM生成的问题
            context_text = item.get('original_context') # 原始上下文
            answer_text = item.get('original_answer') # 原始答案

            # Fallback to original_question/query if generated_question is missing or null
            if not question_text:
                question_text = item.get('original_question') or item.get('query')

            if question_text and context_text and answer_text:
                qa_pairs.append({
                    "question": question_text,
                    "context": context_text,
                    "answer": answer_text # Include original answer for reference/analysis
                })
        
        # Randomly sample max_questions pairs if the list is too long
        if len(qa_pairs) > max_questions:
            qa_pairs = random.sample(qa_pairs, max_questions)
        
        print(f"✅ 加载了 {len(qa_pairs)} 个问答对")
        return qa_pairs

    except Exception as e:
        print(f"❌ 加载AlphaFin数据失败: {e}")
        # Fallback to default questions if data loading fails
        return [
            {"question": "什么是股票投资？", "context": "股票是股份公司所有权的一部分，也是发行的所有权凭证，是股份公司为筹集资金而发行给各个股东作为持股凭证并借以取得股息和红利的一种有价证券。", "answer": "股票是股份公司为筹集资金而发行给股东作为持股凭证的一种有价证券。"},
            {"question": "请解释债券的基本概念", "context": "债券是一种有价证券，是社会各类经济主体为筹集资金而向投资者发行，承诺按一定利率支付利息并按约定条件偿还本金的债权债务凭证。", "answer": "债券是发行者承诺按一定利率支付利息并按约定条件偿还本金的债权债务凭证。"}
        ]


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 GPU内存已清理")


# --- 新增的后期处理函数 (需要根据你的模型实际输出调整) ---
def post_process_generator_response(raw_response: str, prompt_template: str) -> str:
    """
    对Generator LLM的原始输出进行后期处理，移除不必要的元评论和格式标记，
    只保留核心答案。
    """
    cleaned_text = raw_response.strip()

    # 1. 移除模型对Prompt的复述或前导语
    # 比如模型可能重复 "请基于以下提供的上下文信息，直接回答用户的问题。"
    # 找到Prompt中最后的用户问题和答案指示，然后在此之后开始截取
    # 寻找 Answer: 这样的明确指示，并在其后开始
    answer_marker_in_prompt = re.search(r"答案：\s*$", prompt_template, re.MULTILINE)
    if answer_marker_in_prompt:
        # 尝试从 Prompt 结束的地方开始截取
        start_pos_in_raw = raw_response.find(prompt_template[answer_marker_in_prompt.start():])
        if start_pos_in_raw != -1:
            cleaned_text = raw_response[start_pos_in_raw:].strip()
            # 移除 Answer: 自身
            cleaned_text = re.sub(r"^\s*答案：\s*", "", cleaned_text, flags=re.MULTILINE).strip()


    # 2. 定义常见的元评论/不需要内容开头模式（注意顺序，从长到短或从具体到一般）
    unwanted_patterns = [
        r"^\s*\[\s*删除了\"[^\]]+\"后的句子及所有非必要文字\s*\]\s*", # "[删除了...]" 这种自我评价
        r"^\s*注意：严格遵循指令要求，去除所有不必要的内容，仅保留核心信息。\s*", # 模型给自己下指令
        r"^\s*总结\s*", # 总结标题
        r"^\s*请注意，上述分析是基于给定的信息进行的解读，并未引入额外的数据或假设。\s*", # 免责声明
        r"^\s*[\-]{3,}\s*", # --- 分隔线
        r"^\s*\\boxed{", # 匹配 \boxed{
        r"^\s*根据要求，", # 常见的前缀
        r"^\s*但对照示例\d的结构：", # 比较示例
        r"^\s*最终版", # 最终版标记
        r"^\s*进一步压缩", # 压缩说明
        r"^\s*然而，", # 这种转折词，如果不是核心答案一部分
        r"^\s*因此，",
        r"^\s*所以，",
        r"^\s*综上所述，",
        r"^\s*请根据上述规则和示例给出正确答案。", # Clean Prompt 的开头
        r"^\s*内容略去", # Clean Prompt 的开头
        r"^\s*按照规定，答案不得超过\d+字。", # 长度提示
        r"^\s*\*\*重要要求：.*", # 匹配Prompt中的重要要求
        r"^\s*基于上述信息的回答：\s*", # Simple Prompt的开始
        r"^\s*你是一位专业的金融分析师。", # 系统的自我介绍
        r"^\s*请回答以下问题：", # 如果模型复述了问题
        r"^\s*AI助手需要根据提供的上下文信息分析", # CoT的开头
        r"^\s*###.*", # Markdown 标题
        r"^\s*[\-\*]\s+.*", # Markdown 列表项
        r"^\s*\(\s*注：.*?\)" # 开头带括号的注释
    ]
    
    # 循环移除这些模式，直到不再匹配
    # 使用 re.DOTALL 和 re.MULTILINE 确保能匹配多行和行开头
    # 使用 re.IGNORECASE 忽略大小写
    for pattern_str in unwanted_patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        old_text = cleaned_text
        cleaned_text = pattern.sub("", cleaned_text).strip()
        if cleaned_text == old_text and pattern_str.startswith(r"^\s*"): # 优化：如果没变化且是开头模式，则停止循环
            break
            
    # 如果回答以 "总结：" 开头，且不应如此，尝试删除
    cleaned_text = re.sub(r"^\s*总结：\s*", "", cleaned_text, flags=re.MULTILINE).strip()

    # 移除所有可能的尾部不完整句或格式标记
    # 从末尾向前匹配并删除常见的收尾元评论
    unwanted_tail_patterns = [
        r"[\s\S]*?(?:---|\*+\s*总结|\s*\\boxed)", # 匹配 ---, **总结**, \boxed 以及之前所有内容
        r"\s*根据要求，.*", # 匹配 "根据要求，" 及之后所有内容
        r"\s*请注意，上述分析是基于给定的信息进行的解读.*", # 匹配 "请注意，..." 及之后所有内容
        r"\s*综上所述，.*", # 匹配 "综上所述，" 及之后所有内容
        r"\s*但对照示例\d的结构：.*", # 匹配 "但对照示例..." 及之后所有内容
        r"\s*最终版.*", # 匹配 "最终版" 及之后所有内容
        r"\s*进一步压缩.*", # 匹配 "进一步压缩" 及之后所有内容
        r"\s*请根据上述规则和示例给出正确答案.*",
        r"\s*如果你能提供更多上下文信息.*",
        r"\s*我无法直接访问最新的市场研究报告或新闻动态.*",
        r"\s*以下是详细分析.*", # 匹配"以下是详细分析"及之后所有内容
        r"\s*最终答案：.*", # 匹配"最终答案："及之后所有内容
        r"\s*用户问题：.*", # 匹配"用户问题："及之后所有内容
    ]
    for pattern_str in unwanted_tail_patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE | re.DOTALL) # DOTALL for multiline match
        match = pattern.search(cleaned_text)
        if match:
            # 找到匹配项，从匹配的开头处截断
            cleaned_text = cleaned_text[:match.start()].strip()
            # 如果截断后为空，则可能是整个回答都是垃圾，尝试保留原始回答的第一句
            if not cleaned_text:
                first_sentence_match = re.search(r'^(.+?[。？！])', raw_response.strip(), re.DOTALL)
                if first_sentence_match:
                    cleaned_text = first_sentence_match.group(1).strip()
    
    # 移除所有可能的尾部不完整句或格式标记的剩余部分
    cleaned_text = re.sub(r'[\s\S]*?\s*(?:请给出详细分析|关键信息点包括|现在，请你以更简练的方式|好的，我需要处理用户的问题|以下是我的分析|总结|\*\*总结\*\*|^\s*\d+\.\s+\*\*.*\*\*|\*\*总结\*\*|^\s*综上所述|^\s*请注意)', '', cleaned_text, flags=re.DOTALL | re.MULTILINE).strip()
    
    # 移除所有 Markdown 格式标记（#，**，*，-，数字列表开头）
    cleaned_text = re.sub(r'^\s*#+\s*', '', cleaned_text, flags=re.MULTILINE).strip() # 移除标题
    cleaned_text = re.sub(r'\*\*', '', cleaned_text).strip() # 移除粗体
    cleaned_text = re.sub(r'\*\s*', '', cleaned_text).strip() # 移除列表项星号
    cleaned_text = re.sub(r'^\s*\d+\.\s*', '', cleaned_text, flags=re.MULTILINE).strip() # 移除数字列表开头
    cleaned_text = re.sub(r'^\s*-\s*', '', cleaned_text, flags=re.MULTILINE).strip() # 移除列表项横线

    # 移除空行和多余的空白
    cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 确保回答以完整句子结束，如果没有，尝试截取到最近的完整句
    if cleaned_text and not re.search(r'[。？！]$', cleaned_text):
        last_sentence_match = re.search(r'(.+?[。？！])', cleaned_text[::-1], re.DOTALL) # 倒序找第一个句号等
        if last_sentence_match:
            cleaned_text = last_sentence_match.group(1)[::-1].strip() # 倒序反转回来取回完整句子
        else:
            # 如果找不到完整的句子，尝试取到最后一个完整词语的末尾，避免被截断的乱码
            cleaned_text = re.sub(r'[^。？！\s]*$', '', cleaned_text).strip()

    # 如果清理后变为空，尝试保留原始回答的第一句或第一段
    if not cleaned_text and raw_response.strip():
        first_paragraph_match = re.search(r'^([^\n]+\n)*[^\n]*?[。？！]', raw_response.strip(), re.DOTALL)
        if first_paragraph_match:
            cleaned_text = first_paragraph_match.group(0).strip()
        else: # 实在不行，就取原始回答的前100字
            cleaned_text = raw_response.strip()[:150] + "..." if len(raw_response.strip()) > 150 else raw_response.strip()


    return cleaned_text

# --- 主要函数 (用于测试Generator LLM) ---
def test_generator_llm(
    model_name: str, 
    qa_pairs: List[Dict[str, str]], 
    device: str = "cuda:1", 
    max_new_tokens: int = 500, 
    temperature: float = 0.2,
    top_p: float = 0.8,
    use_quantization: bool = True,
    quantization_type: str = "4bit"
) -> Dict[str, Any]:
    print(f"\n🚀 开始测试Generator LLM模型: {model_name}")
    print(f"   设备: {device}")
    print(f"   测试问答对数量: {len(qa_pairs)}")
    
    clear_gpu_memory()
    
    results = {
        "model_name": model_name,
        "device": device,
        "qa_results": [], 
        "success_count": 0,
        "total_time": 0,
        "avg_tokens": 0,
        "memory_usage": 0,
        "config": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "use_quantization": use_quantization,
            "quantization_type": quantization_type
        }
    }
    
    try:
        print(f"🔧 初始化 {model_name}...")
        generator = LocalLLMGenerator(
            model_name=model_name,
            device=device,
            use_quantization=use_quantization,
            quantization_type=quantization_type
        )
        print(f"✅ {model_name} 初始化成功")
        
        if torch.cuda.is_available():
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            results["memory_usage"] = torch.cuda.memory_allocated(device=gpu_id) / 1024**3
            print(f"💾 GPU内存使用: {results['memory_usage']:.2f}GB")
        
        for i, qa_pair in enumerate(qa_pairs):
            question = qa_pair["question"]
            context = qa_pair["context"]
            original_answer = qa_pair["answer"] 

            print(f"\n   🔍 问题 {i+1}: {question}")
            print(f"     上下文 (前100字): {context[:100]}...")
            print(f"     原始答案: {original_answer}")
            
            prompt_template = textwrap.dedent(f"""
            你是一名专业的金融分析师。请基于以下提供的【上下文信息】，直接、准确地回答用户的问题。
            
            重要要求：
            1. **只使用【上下文信息】回答问题，不要添加任何外部知识或猜测。**
            2. **如果【上下文信息】中没有足够的相关信息来回答问题，请明确说明"根据提供的信息无法回答此问题"。**
            3. **回答必须简洁、直接、完整，用自然的中文表达。**
            4. **极度重要：你的输出必须是纯粹、直接的回答，不包含任何自我反思、思考过程、对Prompt的分析、与回答无关的额外注释、任何格式标记（如 \\boxed{{}}、数字列表、加粗）、或任何形式的元评论。请勿引用或复述Prompt内容。你的回答必须直接、简洁地结束，不带任何引导语或后续说明。**
            5. **回答中务必提及【上下文信息】中涉及到的公司名称和股票代码（如果有），以提高可解释性。**
            
            【上下文信息】：
            {context}
            
            用户问题：{question}
            
            答案：
            """).strip()

            start_time = time.time()
            # Pass max_new_tokens directly to generate, as it's a generator setting
            response = generator.generate(
                texts=[prompt_template]
            )[0] 
            end_time = time.time()
            generation_time = end_time - start_time
            
            clean_response = post_process_generator_response(response, prompt_template) # 调用后期处理函数
            
            response_tokens = len(generator.tokenizer.encode(clean_response)) # 使用模型分词器计算tokens
            
            print(f"   ✅ 生成成功")
            print(f"     原始回答 (前200字): {response[:200]}...")
            print(f"     清理后回答 (前200字): {clean_response[:200]}...")
            print(f"     长度 (清理后): {response_tokens} tokens") # 指明是清理后的tokens
            print(f"     时间: {generation_time:.2f}s")
            
            results["qa_results"].append({
                "question": question,
                "context": context, 
                "original_answer": original_answer, 
                "raw_response": response, 
                "clean_response": clean_response, 
                "tokens": response_tokens,
                "time": generation_time,
                "success": True 
            })
            results["success_count"] += 1
            results["total_time"] += generation_time
            
        successful_responses_tokens = [q["tokens"] for q in results["qa_results"] if q["success"]]
        if successful_responses_tokens:
            results["avg_tokens"] = sum(successful_responses_tokens) / len(successful_responses_tokens)
        
        del generator
        clear_gpu_memory()
        
    except Exception as e:
        print(f"❌ {model_name} 初始化或生成失败: {e}")
        results["error"] = str(e)
    
    return results

def save_results(results: Dict[str, Any], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 结果已保存到: {filename}")


def compare_multiple_models(
    model_names: List[str], 
    qa_pairs: List[Dict[str, str]],
    device: str = "cuda:1" # Default device set for testing
) -> Dict[str, Dict[str, Any]]:
    all_results = {}
    
    for model_name in model_names:
        print(f"\n{'='*20} 开始测试 {model_name} {'='*20}")
        results = test_generator_llm(
            model_name=model_name,
            qa_pairs=qa_pairs,
            device=device
        )
        all_results[model_name] = results
        
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        save_results(results, f"{safe_model_name}_generator_llm_results.json")
        
        if model_name != model_names[-1]: 
            try:
                choice = input(f"\n🤔 是否继续测试下一个模型？(y/n): ").lower().strip()
                if choice not in ['y', 'yes', '是']:
                    print("👋 用户中断测试")
                    break
            except KeyboardInterrupt:
                print("\n👋 用户中断测试")
                break
    
    return all_results


def generate_comparison_report(all_results: Dict[str, Dict[str, Any]], output_file: str = "generator_llm_comparison_report.md"):
    print(f"\n📊 生成比较报告...")
    
    report = f"""# Generator LLM 模型比较报告 - AlphaFin数据集

## 📋 测试概述

- 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
- 测试问答对数量: {len(all_results[list(all_results.keys())[0]]['qa_results']) if all_results and 'qa_results' in all_results[list(all_results.keys())[0]] else 0}
- 测试模型数量: {len(all_results)}

## 📈 性能对比

| 模型 | 成功率 | 平均时间(s) | 平均Token数 | GPU内存(GB) | 状态 |
|------|--------|-------------|-------------|-------------|------|
"""
    
    for model_name, results in all_results.items():
        success_count = results.get('success_count', 0)
        total_questions = len(results.get('qa_results', [])) # Changed "questions" to "qa_results"
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
            report += f"- **成功率**: {results['success_count']}/{len(results['qa_results'])}\n" # Changed "questions" to "qa_results"
            report += f"- **平均时间**: {results['total_time']/results['success_count']:.2f}s\n"
            report += f"- **平均Token数**: {results['avg_tokens']:.1f}\n"
            report += f"- **GPU内存**: {results['memory_usage']:.2f}GB\n\n"
            
            report += "**示例回答**:\n\n"
            for i, q_result in enumerate(results['qa_results'][:3]): # Changed "questions" to "qa_results"
                if q_result['success']:
                    report += f"{i+1}. **问题**: {q_result['question']}\n"
                    report += f"   **原始上下文 (前100字)**: {q_result['context'][:100]}...\n" # Added context for reference
                    report += f"   **原始答案**: {q_result['original_answer']}\n" # Added original answer
                    report += f"   **原始模型回答 (前200字)**: {q_result['raw_response'][:200]}...\n" # Raw response
                    report += f"   **清理后回答 (前200字)**: {q_result['clean_response'][:200]}...\n\n" # Cleaned response
    
    report += f"""
## 🎯 总结

"""
    
    best_model = None
    best_success_rate = -1
    
    for model_name, results in all_results.items():
        if 'error' not in results:
            success_rate = results['success_count'] / len(results['qa_results']) if results['qa_results'] else 0
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_model = model_name
    
    if best_model:
        report += f"- **最佳模型**: {best_model} (成功率: {best_success_rate*100:.1f}%)\n"
    
    report += f"- **测试完成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    save_dir = Path(output_file).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 比较报告已保存到: {output_file}")


def main():
    print("🧪 AlphaFin数据集Generator LLM比较测试")
    print("=" * 50)
    
    parser = argparse.ArgumentParser(description="使用AlphaFin数据集比较不同模型")
    parser.add_argument("--model_names", nargs="+", 
                         default=["SUFE-AIFLM-Lab/Fin-R1", "Qwen/Qwen3-8B"], # Added Fin-R1 and Qwen3-8B for comparison
                         help="要测试的模型名称列表")
    parser.add_argument("--data_path", type=str, 
                         default="data/alphafin/alphafin_summarized_and_structured_qa_0627_colab_backward.json", # Updated default data path
                         help="AlphaFin数据文件路径，包含问题、上下文和原始答案")
    parser.add_argument("--max_questions", type=int, default=5, # Reduced default questions to avoid high compute units on testing
                         help="最大测试问题数量")
    parser.add_argument("--device", type=str, default="cuda:1",
                         help="GPU设备")
    parser.add_argument("--output_dir", type=str, default="model_comparison_results",
                         help="输出目录")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ CUDA不可用，将使用CPU")
        args.device = "cpu"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load AlphaFin questions
    print(f"\n📚 加载AlphaFin问答对...")
    qa_pairs = load_alphafin_qa_pairs(args.data_path, args.max_questions)
    
    # Display questions (now qa_pairs contain context and answer)
    print(f"\n📝 测试问答对:")
    for i, qa_pair in enumerate(qa_pairs):
        print(f"   {i+1}. 问题: {qa_pair['question']}")
        print(f"      上下文 (前50字): {qa_pair['context'][:50]}...")
        print(f"      原始答案: {qa_pair['answer']}")
    
    # Compare models
    print(f"\n🔍 开始比较模型: {args.model_names}")
    all_results = compare_multiple_models(
        model_names=args.model_names,
        qa_pairs=qa_pairs,
        device=args.device
    )
    
    # Generate comparison report
    report_path = os.path.join(args.output_dir, "generator_llm_comparison_report.md")
    generate_comparison_report(all_results, report_path)
    
    print(f"\n🎉 模型比较测试完成！")
    print(f"📁 结果文件保存在: {args.output_dir}")


if __name__ == "__main__":
    main()