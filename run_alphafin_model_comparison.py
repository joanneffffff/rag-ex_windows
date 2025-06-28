#!/usr/bin/env python3
"""
运行AlphaFin模型比较的示例脚本
使用真实的AlphaFin问题来比较不同模型
"""

import subprocess
import sys
import os

def run_model_comparison():
    """运行模型比较"""
    print("🚀 开始AlphaFin模型比较测试")
    print("=" * 50)
    
    # 示例1：快速比较Qwen3-8B和Fin-R1
    print("\n📊 示例1：快速比较两个模型")
    print("使用3个AlphaFin问题比较Qwen3-8B和Fin-R1")
    
    cmd1 = [
        "python", "compare_models_with_alphafin.py",
        "--model_names", "Qwen/Qwen3-8B", "SUFE-AIFLM-Lab/Fin-R1",
        "--max_questions", "3",
        "--output_dir", "quick_comparison_results"
    ]
    
    print(f"执行命令: {' '.join(cmd1)}")
    
    # 询问用户是否执行
    try:
        choice = input("\n🤔 是否执行示例1？(y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            print("\n🔧 执行示例1...")
            result = subprocess.run(cmd1, capture_output=True, text=True)
            print("输出:", result.stdout)
            if result.stderr:
                print("错误:", result.stderr)
        else:
            print("跳过示例1")
    except KeyboardInterrupt:
        print("\n👋 用户中断")
        return
    
    # 示例2：详细比较多个模型
    print("\n📊 示例2：详细比较多个模型")
    print("使用5个AlphaFin问题比较多个模型")
    
    cmd2 = [
        "python", "compare_models_with_alphafin.py",
        "--model_names", "Qwen/Qwen3-8B", "Qwen/Qwen2-7B",
        "--max_questions", "5",
        "--output_dir", "detailed_comparison_results"
    ]
    
    print(f"执行命令: {' '.join(cmd2)}")
    
    try:
        choice = input("\n🤔 是否执行示例2？(y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            print("\n🔧 执行示例2...")
            result = subprocess.run(cmd2, capture_output=True, text=True)
            print("输出:", result.stdout)
            if result.stderr:
                print("错误:", result.stderr)
        else:
            print("跳过示例2")
    except KeyboardInterrupt:
        print("\n👋 用户中断")
        return
    
    # 示例3：使用评估数据集
    print("\n📊 示例3：使用评估数据集")
    print("使用alphafin_eval.jsonl中的问题")
    
    cmd3 = [
        "python", "compare_models_with_alphafin.py",
        "--model_names", "Qwen/Qwen3-8B",
        "--data_path", "evaluate_mrr/alphafin_eval.jsonl",
        "--max_questions", "3",
        "--output_dir", "eval_comparison_results"
    ]
    
    print(f"执行命令: {' '.join(cmd3)}")
    
    try:
        choice = input("\n🤔 是否执行示例3？(y/n): ").lower().strip()
        if choice in ['y', 'yes', '是']:
            print("\n🔧 执行示例3...")
            result = subprocess.run(cmd3, capture_output=True, text=True)
            print("输出:", result.stdout)
            if result.stderr:
                print("错误:", result.stderr)
        else:
            print("跳过示例3")
    except KeyboardInterrupt:
        print("\n👋 用户中断")
        return
    
    print("\n🎉 所有示例执行完成！")
    print("\n📁 生成的结果文件:")
    print("   - quick_comparison_results/")
    print("   - detailed_comparison_results/")
    print("   - eval_comparison_results/")
    
    print("\n💡 查看结果:")
    print("   cat quick_comparison_results/model_comparison_report.md")
    print("   cat detailed_comparison_results/model_comparison_report.md")
    print("   cat eval_comparison_results/model_comparison_report.md")

def show_available_models():
    """显示可用的模型"""
    print("\n📋 可用的模型列表:")
    print("=" * 30)
    
    models = [
        ("Qwen/Qwen3-8B", "Qwen3-8B基础版本，推荐使用"),
        ("Qwen/Qwen2-7B", "Qwen2-7B版本，较小但快速"),
        ("Qwen/Qwen2-1.5B", "Qwen2-1.5B版本，最小但最快"),
        ("SUFE-AIFLM-Lab/Fin-R1", "金融专用模型，但内存需求高"),
        ("Llama2-7B-chat-hf", "Llama2-7B聊天版本"),
        ("microsoft/DialoGPT-medium", "微软对话模型，较小"),
    ]
    
    for model_name, description in models:
        print(f"   {model_name}")
        print(f"      {description}")
        print()

def show_usage_examples():
    """显示使用示例"""
    print("\n📖 使用示例:")
    print("=" * 20)
    
    examples = [
        ("基本比较", "python compare_models_with_alphafin.py"),
        ("指定模型", "python compare_models_with_alphafin.py --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B"),
        ("使用评估数据", "python compare_models_with_alphafin.py --data_path evaluate_mrr/alphafin_eval.jsonl"),
        ("限制问题数量", "python compare_models_with_alphafin.py --max_questions 3"),
        ("使用不同GPU", "python compare_models_with_alphafin.py --device cuda:0"),
        ("自定义输出目录", "python compare_models_with_alphafin.py --output_dir my_results"),
    ]
    
    for title, cmd in examples:
        print(f"   {title}:")
        print(f"      {cmd}")
        print()

def main():
    """主函数"""
    print("🧪 AlphaFin模型比较工具")
    print("=" * 30)
    
    while True:
        print("\n请选择操作:")
        print("1. 运行模型比较示例")
        print("2. 查看可用模型")
        print("3. 查看使用示例")
        print("4. 退出")
        
        try:
            choice = input("\n请输入选择 (1-4): ").strip()
            
            if choice == "1":
                run_model_comparison()
            elif choice == "2":
                show_available_models()
            elif choice == "3":
                show_usage_examples()
            elif choice == "4":
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请重新输入")
                
        except KeyboardInterrupt:
            print("\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main() 