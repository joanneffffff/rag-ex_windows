import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import sys

def test_qwen2_7b_quantization(model_name: str = "Qwen/Qwen2-7B-Instruct"):
    print(f"尝试加载和量化模型: {model_name}...")

    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到设备: {device}")

    bnb_config = None
    if device == "cuda":
        print("CUDA设备可用。尝试应用 4-bit 量化...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
            )
            print("BitsAndBytesConfig 已成功创建。")
        except Exception as e:
            print(f"创建 BitsAndBytesConfig 失败: {e}")
            print("请确保您已正确安装 bitsandbytes 库，并且您的CUDA版本与PyTorch和bitsandbytes兼容。")
            print("如果仍然遇到问题，尝试跳过量化（这需要更多显存）。")
            bnb_config = None # 如果创建失败，则不使用量化
            
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("分词器加载成功。")

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, # 如果bnb_config为None，则不进行量化
            device_map="auto", # 使用auto，让transformers自动分配到所有可用GPU或CPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # GPU使用bfloat16，CPU使用float32
        )
        model.eval() # 设置为评估模式
        print(f"模型 {model_name} 加载成功。模型位于: {model.device}")
        
        # 检查模型是否已量化（如果预期是量化）
        if bnb_config is not None and getattr(model, "is_loaded_in_4bit", False):
            print("模型已成功以 4-bit 量化加载。")
        elif bnb_config is not None:
            print("警告: 4-bit 量化配置已指定，但模型可能未按 4-bit 量化加载。请检查日志。")
        else:
            print("模型以全精度或默认精度加载 (未进行 4-bit 量化)。")

        # 简单的文本生成测试
        prompt = "你好，请用中文简单介绍一下大型语言模型。"
        print(f"\n进行简单的文本生成测试，Prompt: {prompt}")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 检查多 GPU 设置
        if isinstance(model, torch.nn.DataParallel):
            # 如果使用了DataParallel，输入需要放到第一个GPU上，或者处理一下
            print("模型正在使用 DataParallel。")
            # DataParallel通常会自动处理设备移动，但为了明确，我们保持inputs在model.device
        elif hasattr(model, 'hf_device_map') and len(model.hf_device_map) > 1:
             print(f"模型已分布到多个设备: {model.hf_device_map}")
        
        
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        
        generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print("\n生成结果:")
        print(generated_text)
        print("\n测试完成。")

    except Exception as e:
        print(f"\n加载或生成过程中发生错误: {e}")
        print("请根据错误信息检查您的PyTorch, CUDA, bitsandbytes, transformers 版本兼容性，以及模型名称是否正确。")
        print("如果您的 GPU 显存不足，可以尝试更小的模型 (如 'Qwen/Qwen2-1.5B-Instruct' 或 'microsoft/Phi-3-mini-4k-instruct')。")
    
    # 释放显存
    del model
    del tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
        print("已清理CUDA显存。")

if __name__ == "__main__":
    # 您可以根据需要修改模型名称进行测试
    test_qwen2_7b_quantization(model_name="Qwen/Qwen2-7B-Instruct")
    # 如果 Qwen2-7B-Instruct 仍然遇到问题，可以尝试更小的模型来确认环境
    # test_qwen2_7b_quantization(model_name="Qwen/Qwen2-1.5B-Instruct")
    # test_qwen2_7b_quantization(model_name="microsoft/Phi-3-mini-4k-instruct")