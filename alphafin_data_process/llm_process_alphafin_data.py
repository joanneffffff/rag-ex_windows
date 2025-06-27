import json
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import textwrap
import re 
import sys

def load_model_and_tokenizer(model_name: str, device: str):
    """
    加载指定的模型和分词器，如果是在 GPU 上则使用 4 位量化。
    """
    print(f"正在加载模型：{model_name}...")
    
    bnb_config = None
    if device == "cuda":
        print("检测到 CUDA 设备。正在应用 4-bit 量化...")
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
            print("这可能是因为您的CUDA版本与PyTorch和bitsandbytes不兼容。")
            print("尝试不使用量化加载（这将需要更多显存），或解决bitsandbytes安装问题。")
            bnb_config = None # 如果创建失败，则不使用量化
            
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("分词器加载成功。")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, # 如果bnb_config为None，则不进行量化
            device_map="auto", # 自动分配到所有可用GPU或CPU
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, # GPU使用bfloat16，CPU使用float32
        )
        model.eval() # 设置为评估模式
        print(f"模型 {model_name} 加载成功。模型位于: {model.device}")
        
        # 检查模型是否已量化（如果预期是量化）
        if bnb_config is not None and getattr(model, "is_loaded_in_4bit", False):
            print("模型已成功以 4-bit 量化加载。")
        elif bnb_config is not None and getattr(model, "is_quantized", False):
             print("模型已成功以 8-bit 或其他量化方式加载。")
        elif bnb_config is not None:
            print("警告: 量化配置已指定，但模型可能未按预期量化加载。请检查日志。")
        else:
            print("模型以全精度或默认精度加载 (未进行量化)。")

    except Exception as e:
        print(f"致命错误：无法加载模型。错误：{e}")
        sys.exit(1) # 加载模型失败则直接退出

    return model, tokenizer

def process_alphafin_data_with_forced_question_metadata(
    input_path: Path,
    output_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    limit: int,
    save_interval: int,
):
    """
    处理 AlphaFin 数据集，进行上下文总结、生成包含强制元数据的问题，并结构化输出。
    """
    # --- 1. 加载模型 ---
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # --- 2. 加载数据 ---
    print(f"正在从 {input_path} 加载已清理的数据...")
    if not input_path.exists():
        print(f"致命错误：输入文件未找到。请创建 '{input_path.name}'。")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    if limit:
        print(f"限制处理 {limit} 条记录。")
        cleaned_data = cleaned_data[:limit]

    # --- 3. 系统提示定义 ---
    system_prompt = textwrap.dedent("""
        你是一名专业的金融分析师。你的任务是根据提供的中文金融新闻文章（Context）和人工编写的答案（Answer），完成以下三项任务：
        
        **任务 1：上下文总结 (summary)**
        对“Context”和“Answer”的核心语义信息进行简洁、精炼的总结。这个总结应该捕捉文章的关键点和“Answer”所表达的核心内容。
        
        **任务 2：生成新问题 (generated_question)**
        根据“Context”和“Answer”生成一个单一的、具体的问题。
        生成的问题必须能被给定的“Answer”直接且完全地回答，且只能使用“Context”中包含的信息。
        **最重要的是：如果“Context”中包含公司名称和股票代码，你生成的问题中必须明确提及这些公司名称和股票代码。例如，如果Context提到了“腾讯控股”和“0700.HK”，你的问题可以是“关于腾讯控股(0700.HK)的最新财报显示了哪些信息？”。**
        
        **任务 3：提取关键元数据**
        从“Context”中提取以下关键元数据：
        - **公司名称 (company_name)**: 从文章中识别出的**公司全名**，例如“腾讯控股”。
        - **股票代码 (stock_code)**: 公司的**股票代码**，例如“0700.HK”或“600519.SH”。
        - **报告日期 (report_date)**: 如果文章中**明确提到了**一个具体的日期（如财报日期、事件发生日期），则提取该日期，格式为“YYYY年MM月DD日”；如果**无法明确识别**或文章中没有提及，则**留空**（即生成一个空字符串""）。
        
        **输出格式要求：**
        请严格以 **JSON 格式** 输出你的所有结果。你的输出必须是一个**完整的、可解析的 JSON 字符串**，不包含任何额外的文字、解释或代码块标记（如 ```json）。JSON 对象必须包含以下字段：
        ```json
        {
          "summary": "金融文章的简洁总结",
          "generated_question": "包含公司名称和股票代码的生成问题",
          "company_name": "提取出的公司全名",
          "stock_code": "提取出的股票代码",
          "report_date": "提取出的报告日期，若无则为空字符串"
        }
        ```
        请确保所有字段都按照示例中的键名和类型严格输出。
    """).strip()

    processed_data = []
    
    # --- 4. 批量处理数据 ---
    print(f"正在批量处理 {len(cleaned_data)} 篇文章，批大小为 {batch_size}...")
    
    for batch_idx, i in enumerate(tqdm(range(0, len(cleaned_data), batch_size), desc="处理数据")):
        batch_records = cleaned_data[i:i+batch_size]
        
        messages_batch = []
        original_records_in_batch = []
        for record in batch_records:
            context = record.get('context', '').strip()
            answer = record.get('answer', '').strip()
            original_question = record.get('query', '').strip() # 从 'query' 键获取原始问题
            
            # --- 核心修改：处理 context 和 answer 中的多余换行符 ---
            # 使用正则表达式将所有连续的空白字符（包括\n, \t, \r, 空格）替换为单个空格
            context = re.sub(r'\s+', ' ', context).strip()
            answer = re.sub(r'\s+', ' ', answer).strip()
            # ----------------------------------------------------

            if context and answer:
                user_content = f"Context:\n{context}\n\nAnswer:\n{answer}"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                messages_batch.append(messages)
                original_records_in_batch.append(record)
        
        if not messages_batch:
            continue

        is_qwen_model = "qwen" in model_name.lower()
        template_args = {"tokenize": False, "add_generation_prompt": True}
        if is_qwen_model:
            template_args["enable_thinking"] = False # Qwen特定的优化

        prompts = [
            tokenizer.apply_chat_template(
                conversation=m, **template_args
            ) for m in messages_batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": 400, # 增加 token 数量以同时覆盖总结、问题和元数据，并确保格式完整
            "do_sample": True, 
            "pad_token_id": tokenizer.pad_token_id,
            "temperature": 0.7, # 稍微降低温度以提高格式稳定性
            "top_p": 0.9 # 限制采样范围
        }
        if is_qwen_model:
            gen_kwargs["top_p"] = 0.95; gen_kwargs["temperature"] = 0.6 # QWen 模型的推荐参数
        
        outputs = model.generate(**inputs, **gen_kwargs)

        input_ids_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_ids_len:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        for idx, generated_text in enumerate(decoded_outputs):
            # 清理生成的文本，移除特殊 token 和可能的代码块标记
            clean_text = generated_text.split('<|im_end|>')[0].replace(tokenizer.eos_token, '').strip()
            # 移除 JSON 代码块标记，如果模型错误地生成了
            if clean_text.startswith("```json"):
                clean_text = clean_text[len("```json"):].strip()
            if clean_text.endswith("```"):
                clean_text = clean_text[:-len("```")].strip()
            
            try:
                # 尝试解析 JSON
                generated_json = json.loads(clean_text)
                
                # 定义期望的字段，用于验证
                expected_fields = ["summary", "generated_question", "company_name", "stock_code", "report_date"]
                # 检查所有期望字段是否存在且关键字段值非空
                if all(field in generated_json for field in expected_fields) and \
                   all(generated_json.get(field) is not None and generated_json.get(field).strip() != '' for field in ["summary", "generated_question", "company_name", "stock_code"]):
                    
                    # 确保 report_date 字段存在，如果为空字符串则存储为 None
                    report_date_value = generated_json['report_date'].strip() if generated_json['report_date'] else None
                    
                    original_record = original_records_in_batch[idx]
                    processed_data.append({
                        'original_context': original_record.get('context', '').strip(), 
                        'original_answer': original_record.get('answer', '').strip(), 
                        'original_question': original_record.get('query', '').strip(), # 从 'query' 字段获取原始问题
                        'summary': generated_json['summary'].strip(),
                        'generated_question': generated_json['generated_question'].strip(),
                        'company_name': generated_json['company_name'].strip(),
                        'stock_code': generated_json['stock_code'].strip(),
                        'report_date': report_date_value 
                    })
                else:
                    print(f"警告：跳过无效生成输出 (JSON 缺少关键字段或关键字段值为空)。\n原始输出：{clean_text}")

            except json.JSONDecodeError:
                print(f"警告：无法解析 JSON 输出，可能格式不正确。\n原始输出：{clean_text}")
            except Exception as e:
                print(f"处理生成输出时发生错误：{e}\n原始输出：{clean_text}")

        # --- 定期保存 ---
        if (batch_idx + 1) % save_interval == 0 and processed_data:
            print(f"\n--- 定期保存：在批次 {batch_idx+1} 保存 {len(processed_data)} 条记录 ---")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)

    # --- 5. 最终保存 ---
    print(f"\n最终保存：正在将所有 {len(processed_data)} 条新处理的记录保存到 {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
        
    print("处理完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="使用 LLM 对 AlphaFin 文章进行上下文总结，并生成包含强制公司名称/股票代码的问题，输出结构化JSON。",
        epilog="在 CPU 上测试时请使用 Phi-3-mini 等小型模型。"
    )
    
    parser.add_argument("--input_file", type=str, default="data/alphafin/alphafin_rag_ready.json", help="输入已清理数据文件的路径。")
    parser.add_argument("--output_file", type=str, default="data/alphafin/alphafin_summarized_and_structured_qa.json", help="输出包含总结、强制元数据问题和结构化JSON的文件的路径。")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct", help="要使用的 Hugging Face 模型名称。默认使用 Qwen2-7B-Instruct。")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备（'cuda' 或 'cpu'）。")
    parser.add_argument("--batch_size", type=int, default=1, help="生成批大小（GPU 上通常可设为更大值，CPU 上建议为1）。")
    parser.add_argument("--limit", type=int, default=None, help="要处理的记录数量（用于测试，None表示处理所有）。")
    parser.add_argument("--save_interval", type=int, default=50, help="每 N 个批次保存一次进度。")
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告：指定了 CUDA 但不可用。正在回退到 CPU。")
        args.device = "cpu"

    if args.device == "cpu":
        print("正在 CPU 上运行。请考虑使用较小的批大小（例如 --batch_size 1）和限制（例如 --limit 10）。")

    process_alphafin_data_with_forced_question_metadata(
        input_path=Path(args.input_file),
        output_path=Path(args.output_file),
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        save_interval=args.save_interval
    )