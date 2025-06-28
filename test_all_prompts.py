#!/usr/bin/env python3
"""
测试所有prompt模式的效果：Few-Shot vs CoT vs Simple
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_all_prompts():
    """测试所有prompt模式的效果"""
    try:
        from config.parameters import Config
        from xlm.registry.generator import load_generator
        from xlm.components.rag_system.rag_system import (
            PROMPT_TEMPLATE_ZH, 
            PROMPT_TEMPLATE_ZH_COT,
            PROMPT_TEMPLATE_ZH_SIMPLE
        )
        
        print("🧪 测试所有Prompt模式效果...")
        
        # 临时修改配置 - 使用更大的max_new_tokens
        config = Config()
        original_max_tokens = config.generator.max_new_tokens
        config.generator.max_new_tokens = 300  # 大幅增加到300
        
        print(f"使用生成器模型: {config.generator.model_name}")
        print(f"生成参数: max_tokens=300, temperature={config.generator.temperature}")
        
        # 加载生成器
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
        
        # 测试三种模式
        modes = [
            ("Few-Shot", PROMPT_TEMPLATE_ZH),
            ("Chain-of-Thought", PROMPT_TEMPLATE_ZH_COT),
            ("Simple", PROMPT_TEMPLATE_ZH_SIMPLE)
        ]
        
        results = {}
        
        for mode_name, prompt_template in modes:
            print(f"\n" + "="*60)
            print(f"🔧 测试 {mode_name} 模式")
            print("="*60)
            
            # 构建prompt
            prompt = prompt_template.format(context=test_context, question=test_question)
            print(f"Prompt长度: {len(prompt)} 字符")
            
            # 生成回答
            print(f"\n🤖 生成回答中...")
            generated_responses = generator.generate(texts=[prompt])
            answer = generated_responses[0] if generated_responses else "生成失败"
            
            print(f"\n✅ {mode_name} 回答:")
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
            
            # 检查长度是否合适（不同模式有不同的长度标准）
            if mode_name == "Simple":
                # Simple模式应该更短
                if 20 <= answer_length <= 150:
                    quality_score += 1
                elif answer_length < 20:
                    issues.append("回答过短")
                else:
                    issues.append("回答过长")
            else:
                # 其他模式的标准
                if 50 <= answer_length <= 300:
                    quality_score += 1
                elif answer_length < 50:
                    issues.append("回答过短")
                else:
                    issues.append("回答过长")
            
            # 检查是否简洁（Simple模式额外加分）
            if mode_name == "Simple" and answer_length <= 100:
                quality_score += 1
            
            print(f"质量评分: {quality_score}/6" if mode_name == "Simple" else f"质量评分: {quality_score}/5")
            if issues:
                print(f"问题: {', '.join(issues)}")
            
            # 保存结果
            results[mode_name] = {
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
        
        print(f"{'模式':<15} {'字符数':<8} {'词数':<6} {'质量评分':<8} {'Prompt长度':<12} {'状态'}")
        print("-" * 70)
        
        best_score = 0
        best_mode = None
        
        for mode_name in ["Few-Shot", "Chain-of-Thought", "Simple"]:
            result = results[mode_name]
            max_score = 6 if mode_name == "Simple" else 5
            status = "✅ 推荐" if result["quality_score"] >= max_score * 0.8 else "⚠️ 一般" if result["quality_score"] >= max_score * 0.6 else "❌ 较差"
            
            if result["quality_score"] > best_score:
                best_score = result["quality_score"]
                best_mode = mode_name
            
            print(f"{mode_name:<15} {result['length']:<8} {result['word_count']:<6} {result['quality_score']:<8} {result['prompt_length']:<12} {status}")
        
        print(f"\n🏆 最佳模式: {best_mode} (质量评分: {best_score})")
        
        # 详细推荐
        print(f"\n💡 推荐配置:")
        if best_mode == "Simple":
            print("✅ 推荐使用 Simple 模式")
            print("   理由: 简洁明了，适合快速问答")
        elif best_mode == "Few-Shot":
            print("✅ 推荐使用 Few-Shot 模式")
            print("   理由: 通过示例学习，回答质量稳定")
        else:
            print("✅ 推荐使用 Chain-of-Thought 模式")
            print("   理由: 复杂推理能力强")
        
        # 使用建议
        print(f"\n🎯 使用建议:")
        print("- Simple模式: 适合简单直接的问题，追求简洁")
        print("- Few-Shot模式: 适合一般问答，平衡质量和长度")
        print("- CoT模式: 适合复杂推理问题，需要详细分析")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_all_prompts() 