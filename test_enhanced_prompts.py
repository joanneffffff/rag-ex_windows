#!/usr/bin/env python3
"""
测试增强的Prompt模板效果
- 大幅增加max_new_tokens到600
- 添加更强硬的负面约束
- 验证Few-Shot Prompt的效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xlm.components.generator.local_llm_generator import LocalLLMGenerator
from config.parameters import Config

def test_enhanced_prompts():
    """测试增强的Prompt模板效果"""
    print("🧪 测试增强的Prompt模板效果...")
    
    # 加载配置
    config = Config()
    print(f"使用生成器模型: {config.generator.model_name}")
    print(f"量化类型: {config.generator.quantization_type}")
    print(f"max_new_tokens: {config.generator.max_new_tokens}")
    print()
    
    # 测试问题
    test_question = "首钢股份在2023年上半年的业绩表现如何？"
    test_context = """首钢股份2023年上半年实现营业收入1,234.56亿元，同比下降21.7%；净利润为-3.17亿元，同比下降77.14%，每股亏损0.06元。公司表示将通过注入优质资产来改善长期盈利能力，并承诺回馈股东。"""
    
    print("🔧 加载生成器...")
    generator = LocalLLMGenerator()
    
    print("=" * 60)
    print("🔧 测试增强的Few-Shot Prompt")
    print("=" * 60)
    
    # 手动构建prompt来测试
    from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH
    
    prompt = PROMPT_TEMPLATE_ZH.format(context=test_context, question=test_question)
    print(f"Prompt长度: {len(prompt)} 字符")
    print()
    print("📝 Prompt内容:")
    print("-" * 40)
    print(prompt)
    print("-" * 40)
    print()
    
    print("🤖 生成回答中...")
    response = generator.generate([prompt])[0]
    
    print("✅ 增强的Few-Shot 回答:")
    print("-" * 40)
    print(response)
    print("-" * 40)
    
    # 分析回答质量
    print()
    print("📊 回答分析:")
    print(f"字符数: {len(response)}")
    print(f"词数: {len(response.split())}")
    
    # 检查是否包含元评论
    meta_indicators = [
        "根据上述", "根据以上", "综上所述", "总结", "注意", "评分", 
        "修改后", "版本", "规定", "答案应", "不得超过", "请根据",
        "上述内容", "上述规则", "上述示例", "上述分析"
    ]
    
    has_meta = any(indicator in response for indicator in meta_indicators)
    print(f"包含元评论: {'是' if has_meta else '否'}")
    
    # 检查是否完整
    is_complete = not response.endswith("...") and len(response) > 50
    print(f"回答完整: {'是' if is_complete else '否'}")
    
    # 质量评分
    quality_score = 0
    if is_complete:
        quality_score += 2
    if not has_meta:
        quality_score += 2
    if len(response) < 200:  # 简洁
        quality_score += 1
    if "首钢股份" in response and "2023年" in response:
        quality_score += 1
    
    print(f"质量评分: {quality_score}/6")
    
    if quality_score >= 4:
        print("🎉 回答质量良好！")
    elif quality_score >= 2:
        print("⚠️ 回答质量一般，需要进一步优化")
    else:
        print("❌ 回答质量较差，需要大幅改进")
    
    print()
    print("=" * 60)
    print("🏆 测试总结")
    print("=" * 60)
    print(f"✅ max_new_tokens已增加到: {config.generator.max_new_tokens}")
    print(f"✅ 已添加强硬负面约束")
    print(f"✅ 使用Few-Shot Prompt模板")
    print(f"✅ 回答字符数: {len(response)}")
    print(f"✅ 质量评分: {quality_score}/6")
    
    if quality_score >= 4:
        print("🎉 增强的Prompt模板效果良好！")
    else:
        print("💡 建议进一步调整Prompt模板或生成参数")

if __name__ == "__main__":
    test_enhanced_prompts() 