#!/usr/bin/env python3
"""
内存优化版本的测试脚本，使用4bit量化
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_memory_optimized():
    """使用4bit量化的内存优化测试"""
    try:
        from config.parameters import Config
        from xlm.registry.generator import load_generator
        from xlm.components.rag_system.rag_system import PROMPT_TEMPLATE_ZH
        
        print("🧪 内存优化测试（4bit量化）...")
        
        # 设置环境变量以优化内存使用
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # 加载配置
        config = Config()
        print(f"使用生成器模型: {config.generator.model_name}")
        print(f"量化类型: {config.generator.quantization_type}")
        print(f"max_new_tokens: {config.generator.max_new_tokens}")
        
        # 加载生成器（使用4bit量化）
        print("\n🔧 加载生成器（4bit量化）...")
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
        
        # 构建prompt
        prompt = PROMPT_TEMPLATE_ZH.format(context=test_context, question=test_question)
        
        print(f"\n📝 测试问题: {test_question}")
        print(f"📄 上下文长度: {len(test_context)} 字符")
        print(f"🔧 Prompt长度: {len(prompt)} 字符")
        
        # 生成回答
        print(f"\n🤖 生成回答中（4bit量化）...")
        generated_responses = generator.generate(texts=[prompt])
        answer = generated_responses[0] if generated_responses else "生成失败"
        
        print(f"\n✅ 生成的回答:")
        print("=" * 50)
        print(answer)
        print("=" * 50)
        
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
            print("✅ 包含关键信息")
        else:
            issues.append("缺少关键信息")
            print("❌ 缺少关键信息")
        
        # 检查是否包含具体数据
        if any(char.isdigit() for char in answer):
            quality_score += 1
            print("✅ 包含具体数据")
        else:
            issues.append("缺少具体数据")
            print("❌ 缺少具体数据")
        
        # 检查是否完整
        if answer.endswith(('.', '。', '！', '？')):
            quality_score += 1
            print("✅ 回答完整")
        else:
            issues.append("回答不完整")
            print("❌ 回答不完整")
        
        # 检查长度是否合适
        if 50 <= answer_length <= 500:  # 放宽长度限制，因为增加了token数
            quality_score += 1
            print("✅ 长度适中")
        elif answer_length < 50:
            issues.append("回答过短")
            print("❌ 回答过短")
        else:
            issues.append("回答过长")
            print("❌ 回答过长")
        
        print(f"\n🏆 质量评分: {quality_score}/5")
        if issues:
            print(f"问题: {', '.join(issues)}")
        
        # 内存使用情况
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(1) / 1024**3
                gpu_memory_reserved = torch.cuda.memory_reserved(1) / 1024**3
                print(f"\n💾 GPU内存使用情况:")
                print(f"已分配: {gpu_memory:.2f} GB")
                print(f"已保留: {gpu_memory_reserved:.2f} GB")
        except Exception as e:
            print(f"无法获取GPU内存信息: {e}")
        
        return {
            "answer": answer,
            "length": answer_length,
            "word_count": word_count,
            "quality_score": quality_score,
            "issues": issues
        }
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_memory_optimized() 