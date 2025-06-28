#!/usr/bin/env python3
"""
测试优化后的生成器，验证是否还会生成模板化回答
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_generation():
    """测试生成器是否还会生成模板化回答"""
    try:
        from config.parameters import Config
        from xlm.registry.generator import load_generator
        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH
        
        print("🧪 测试优化后的生成器...")
        
        # 加载配置
        config = Config()
        print(f"使用生成器模型: {config.generator.model_name}")
        print(f"生成参数: temperature={config.generator.temperature}, top_p={config.generator.top_p}")
        
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
        
        # 模拟检索到的上下文（实际数据）
        test_context = """
        首钢股份2023年上半年业绩报告显示，公司实现营业收入1,234.56亿元，同比下降21.7%；
        净利润为-3.17亿元，同比下降77.14%，每股亏损0.06元。
        公司表示将通过注入优质资产来提升长期盈利能力，回馈股东。
        """
        
        # 构建prompt
        prompt = PROMPT_TEMPLATE_ZH.format(context=test_context, question=test_question)
        
        print(f"\n📝 测试问题: {test_question}")
        print(f"📄 上下文: {test_context.strip()}")
        print(f"\n🔧 生成的Prompt:")
        print("-" * 50)
        print(prompt)
        print("-" * 50)
        
        # 生成回答
        print("\n🤖 生成回答中...")
        generated_responses = generator.generate(texts=[prompt])
        answer = generated_responses[0] if generated_responses else "生成失败"
        
        print(f"\n✅ 生成的回答:")
        print("=" * 50)
        print(answer)
        print("=" * 50)
        
        # 分析回答质量
        print(f"\n📊 回答质量分析:")
        
        # 检查是否包含模板化内容
        template_indicators = [
            "请按照上述格式",
            "对以下问题进行回答",
            "问题:",
            "回答:",
            "---",
            "首钢股份的股价走势如何？",
            "首钢股份的主要业务是什么？"
        ]
        
        has_template = any(indicator in answer for indicator in template_indicators)
        
        if has_template:
            print("❌ 检测到模板化回答")
            print("包含的模板化内容:")
            for indicator in template_indicators:
                if indicator in answer:
                    print(f"  - {indicator}")
        else:
            print("✅ 未检测到模板化回答")
        
        # 检查回答长度
        answer_length = len(answer)
        print(f"回答长度: {answer_length} 字符")
        
        if answer_length > 500:
            print("⚠️ 回答可能过长")
        elif answer_length < 50:
            print("⚠️ 回答可能过短")
        else:
            print("✅ 回答长度适中")
        
        # 检查是否回答了问题
        if "首钢" in answer and ("业绩" in answer or "利润" in answer or "收入" in answer):
            print("✅ 回答内容相关")
        else:
            print("❌ 回答内容可能不相关")
        
        return answer
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_generation() 