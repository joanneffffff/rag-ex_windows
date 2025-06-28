#!/usr/bin/env python3
"""
测试不同简洁程度prompt的效果
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_clean_prompts():
    """测试不同简洁程度prompt的效果"""
    try:
        from config.parameters import Config
        from xlm.registry.generator import load_generator
        from xlm.components.rag_system.rag_system import (
            PROMPT_TEMPLATE_ZH, 
            PROMPT_TEMPLATE_ZH_CLEAN,
            PROMPT_TEMPLATE_ZH_SIMPLE
        )
        
        print("🧪 测试不同简洁程度Prompt效果...")
        
        # 设置环境变量以优化内存使用
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 加载配置
        config = Config()
        print(f"使用生成器模型: {config.generator.model_name}")
        print(f"量化类型: {config.generator.quantization_type}")
        print(f"max_new_tokens: {config.generator.max_new_tokens}")
        
        # 加载生成器
        print("\n🔧 加载生成器...")
        generator = load_generator(
            generator_model_name=config.generator.model_name,
            use_local_llm=True,
            use_gpu=True,
            gpu_device="cuda:1",
            cache_dir=config.generator.cache_dir
        )
        
        # 测试问题
        test_question = "首钢股份的业绩表现如何？"
        
        # 模拟检索到的上下文
        test_context = """
        首钢股份2023年上半年业绩报告显示，公司实现营业收入1,234.56亿元，同比下降21.7%；
        净利润为-3.17亿元，同比下降77.14%，每股亏损0.06元。
        公司表示将通过注入优质资产来提升长期盈利能力，回馈股东。
        """
        
        # 测试三种prompt
        prompts = [
            ("Few-Shot", PROMPT_TEMPLATE_ZH),
            ("Clean", PROMPT_TEMPLATE_ZH_CLEAN),
            ("Simple", PROMPT_TEMPLATE_ZH_SIMPLE)
        ]
        
        results = {}
        
        for prompt_name, prompt_template in prompts:
            print(f"\n" + "="*60)
            print(f"🔧 测试 {prompt_name} Prompt")
            print("="*60)
            
            # 构建prompt
            prompt = prompt_template.format(context=test_context, question=test_question)
            print(f"Prompt长度: {len(prompt)} 字符")
            
            # 生成回答
            print(f"\n🤖 生成回答中...")
            generated_responses = generator.generate(texts=[prompt])
            answer = generated_responses[0] if generated_responses else "生成失败"
            
            print(f"\n✅ {prompt_name} 回答:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
            # 分析回答
            answer_length = len(answer)
            word_count = len(answer.split())
            
            print(f"\n📊 回答分析:")
            print(f"字符数: {answer_length}")
            print(f"词数: {word_count}")
            
            # 质量评估
            quality_score = 0
            issues = []
            
            # 检查是否包含关键信息
            if "首钢" in answer and ("业绩" in answer or "利润" in answer or "收入" in answer):
                quality_score += 2
            else:
                issues.append("缺少关键信息")
            
            # 检查是否包含具体数据
            if any(char.isdigit() for char in answer):
                quality_score += 1
            else:
                issues.append("缺少具体数据")
            
            # 检查是否完整
            if answer.endswith(('.', '。', '！', '？')):
                quality_score += 1
            else:
                issues.append("回答不完整")
            
            # 检查长度是否合适
            if prompt_name == "Simple":
                # Simple模式应该很短
                if 20 <= answer_length <= 100:
                    quality_score += 1
                elif answer_length < 20:
                    issues.append("回答过短")
                else:
                    issues.append("回答过长")
            elif prompt_name == "Clean":
                # Clean模式应该适中
                if 50 <= answer_length <= 150:
                    quality_score += 1
                elif answer_length < 50:
                    issues.append("回答过短")
                else:
                    issues.append("回答过长")
            else:
                # Few-Shot模式可以稍长
                if 50 <= answer_length <= 300:
                    quality_score += 1
                elif answer_length < 50:
                    issues.append("回答过短")
                else:
                    issues.append("回答过长")
            
            # 检查是否包含格式标记
            format_indicators = ["---", "注意", "\\boxed", "**", "1.", "2.", "3."]
            has_format = any(indicator in answer for indicator in format_indicators)
            if not has_format:
                quality_score += 1
            else:
                issues.append("包含格式标记")
            
            print(f"质量评分: {quality_score}/6")
            if issues:
                print(f"问题: {', '.join(issues)}")
            
            # 保存结果
            results[prompt_name] = {
                "answer": answer,
                "length": answer_length,
                "word_count": word_count,
                "quality_score": quality_score,
                "issues": issues,
                "prompt_length": len(prompt)
            }
        
        # 比较结果
        print(f"\n" + "="*60)
        print("📊 综合比较结果")
        print("="*60)
        
        print(f"{'Prompt':<12} {'字符数':<8} {'词数':<6} {'质量评分':<8} {'Prompt长度':<12} {'状态'}")
        print("-" * 65)
        
        best_score = 0
        best_prompt = None
        
        for prompt_name in ["Few-Shot", "Clean", "Simple"]:
            result = results[prompt_name]
            status = "✅ 推荐" if result["quality_score"] >= 5 else "⚠️ 一般" if result["quality_score"] >= 4 else "❌ 较差"
            
            if result["quality_score"] > best_score:
                best_score = result["quality_score"]
                best_prompt = prompt_name
            
            print(f"{prompt_name:<12} {result['length']:<8} {result['word_count']:<6} {result['quality_score']:<8} {result['prompt_length']:<12} {status}")
        
        print(f"\n🏆 最佳Prompt: {best_prompt} (质量评分: {best_score}/6)")
        
        # 推荐
        print(f"\n💡 推荐:")
        if best_prompt == "Clean":
            print("✅ 推荐使用 Clean Prompt")
            print("   理由: 平衡了简洁性和信息完整性")
        elif best_prompt == "Simple":
            print("✅ 推荐使用 Simple Prompt")
            print("   理由: 最简洁，适合快速问答")
        else:
            print("✅ 推荐使用 Few-Shot Prompt")
            print("   理由: 通过示例学习，质量稳定")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_clean_prompts() 