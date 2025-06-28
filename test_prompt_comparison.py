#!/usr/bin/env python3
"""
测试和比较few-shot与Chain-of-Thought (CoT) 的效果
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_prompt_comparison():
    """比较不同prompt模板的效果"""
    try:
        from config.parameters import Config
        from xlm.registry.generator import load_generator
        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH, PROMPT_TEMPLATE_ZH_COT
        
        print("🧪 测试Few-Shot vs Chain-of-Thought效果...")
        
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
        
        # 测试Few-Shot版本
        print("\n" + "="*60)
        print("📝 测试Few-Shot版本")
        print("="*60)
        
        prompt_few_shot = PROMPT_TEMPLATE_ZH.format(context=test_context, question=test_question)
        print(f"Prompt长度: {len(prompt_few_shot)} 字符")
        
        generated_responses = generator.generate(texts=[prompt_few_shot])
        answer_few_shot = generated_responses[0] if generated_responses else "生成失败"
        
        print(f"\n✅ Few-Shot回答:")
        print("-" * 40)
        print(answer_few_shot)
        print("-" * 40)
        
        # 测试CoT版本
        print("\n" + "="*60)
        print("🧠 测试Chain-of-Thought版本")
        print("="*60)
        
        prompt_cot = PROMPT_TEMPLATE_ZH_COT.format(context=test_context, question=test_question)
        print(f"Prompt长度: {len(prompt_cot)} 字符")
        
        generated_responses = generator.generate(texts=[prompt_cot])
        answer_cot = generated_responses[0] if generated_responses else "生成失败"
        
        print(f"\n✅ CoT回答:")
        print("-" * 40)
        print(answer_cot)
        print("-" * 40)
        
        # 比较分析
        print("\n" + "="*60)
        print("📊 效果比较分析")
        print("="*60)
        
        # 长度比较
        few_shot_length = len(answer_few_shot)
        cot_length = len(answer_cot)
        
        print(f"Few-Shot回答长度: {few_shot_length} 字符")
        print(f"CoT回答长度: {cot_length} 字符")
        
        if few_shot_length < cot_length:
            print("✅ Few-Shot回答更简洁")
        else:
            print("✅ CoT回答更简洁")
        
        # 质量分析
        def analyze_quality(answer):
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
            
            # 检查是否逻辑清晰
            if "但是" in answer or "然而" in answer or "尽管" in answer:
                quality_score += 1
            else:
                issues.append("逻辑关系不够清晰")
            
            # 检查是否简洁
            if len(answer) < 200:
                quality_score += 1
            else:
                issues.append("回答过长")
            
            return quality_score, issues
        
        few_shot_score, few_shot_issues = analyze_quality(answer_few_shot)
        cot_score, cot_issues = analyze_quality(answer_cot)
        
        print(f"\nFew-Shot质量评分: {few_shot_score}/5")
        if few_shot_issues:
            print(f"Few-Shot问题: {', '.join(few_shot_issues)}")
        
        print(f"CoT质量评分: {cot_score}/5")
        if cot_issues:
            print(f"CoT问题: {', '.join(cot_issues)}")
        
        # 推荐
        if few_shot_score > cot_score:
            print("\n🏆 推荐使用Few-Shot版本")
        elif cot_score > few_shot_score:
            print("\n🏆 推荐使用Chain-of-Thought版本")
        else:
            print("\n🤝 两种方法效果相当")
        
        return {
            "few_shot": answer_few_shot,
            "cot": answer_cot,
            "few_shot_score": few_shot_score,
            "cot_score": cot_score
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_prompt_comparison() 