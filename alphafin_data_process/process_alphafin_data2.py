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
    Loads the specified model and tokenizer with 4-bit quantization if on GPU.
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
            bnb_config = None 
            
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("分词器加载成功。")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, 
        )
        model.eval() 
        print(f"模型 {model_name} 加载成功。模型位于: {model.device}")
        
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
        sys.exit(1) 

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
    Processes AlphaFin dataset for context summarization, question generation with forced metadata, and structured output.
    Supports checkpointing/resume functionality by counting processed records in the output file.
    """
    # --- 1. Load Model ---
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # --- 2. Load Data ---
    print(f"正在从 {input_path} 加载已清理的数据...")
    if not input_path.exists():
        print(f"致命错误：输入文件未找到。请创建 '{input_path.name}'。")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    if limit:
        print(f"Limiting processing to {limit} records.")
        cleaned_data = cleaned_data[:limit]
    
    total_records_in_input = len(cleaned_data)

    # --- 3. Checkpoint/Resume Logic (counting processed records) ---
    start_index = 0 # 从这个索引开始处理
    if output_path.exists():
        print(f"检测到输出文件 '{output_path}'。尝试加载已处理记录以支持断点续跑...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                # 尝试加载整个文件，如果文件是有效JSON数组
                existing_data = json.load(f)
                start_index = len(existing_data)
            print(f"已从 '{output_path}' 加载 {start_index} 条已处理记录。")
            print(f"将从原始输入数据的第 {start_index} 条记录开始处理。")
        except json.JSONDecodeError:
            print(f"警告: 现有输出文件 '{output_path}' 格式不正确。将从头开始处理。")
            start_index = 0 # 文件损坏，从头开始
        except Exception as e:
            print(f"读取现有输出文件时发生错误: {e}。将从头开始处理。")
            start_index = 0
    
    if start_index >= total_records_in_input:
        print("所有记录已处理完成。")
        return # 所有记录都已处理
    
    print(f"总共有 {total_records_in_input} 条记录在输入文件中。")
    print(f"需要处理的记录数: {total_records_in_input - start_index} 条。")


    # --- 4. System Prompt Definition ---
    system_prompt = textwrap.dedent("""
        你是一名专业的金融分析师。你的任务是根据提供的中文金融新闻文章（Context）和人工编写的答案（Answer），严格、清晰地完成以下三项任务：
        
        **任务 1：生成简洁、完整的上下文总结 (summary)**
        对“Context”和“Answer”的核心语义信息进行简洁、精炼的总结。这个总结必须是**一个或多个完整、通顺的中文句子**，精准捕捉文章的关键点和“Answer”所表达的核心内容。请确保总结流畅且信息完整。
        
        **任务 2：生成一个具体、完整的问题 (generated_question)**
        根据“Context”和“Answer”生成一个单一的、具体的问题。这个问题必须是**一个完整、清晰的中文问句**，能够被给定的“Answer”直接且完全地回答，并且只能使用“Context”中包含的信息。
        **如果“Context”中明确提及了公司名称和/或股票代码，你生成的问题中必须明确提及这些信息，以提高问题特异性。例如，如果Context提到了“腾讯控股”和“0700.HK”，你的问题可以是“关于腾讯控股(0700.HK)的最新财报显示了哪些信息？”。如果Context中没有明确的公司名称或股票代码，则问题中无需包含，但仍需确保问题是一个完整且有意义的句子。**
        
        **任务 3：准确提取关键元数据**
        从“Context”中准确提取以下关键元数据：
        - **公司名称 (company_name)**: 从文章中识别出的**公司全名**。如果Context中未明确提及公司，则**留空**（即生成一个空字符串""）。
        - **股票代码 (stock_code)**: 公司的**股票代码**。如果Context中未明确提及股票代码，则**留空**（即生成一个空字符串""）。
        - **报告日期 (report_date)**: 如果文章中**明确提到了**一个具体的日期（如财报日期、事件发生日期），则提取该日期，格式为“YYYY年MM月DD日”；如果**无法明确识别**或文章中没有提及，则**留空**（即生成一个空字符串""）。
        
        **输出格式要求：**
        请严格以 **JSON 格式** 输出你的所有结果。你的输出必须是一个**完整的、可解析的 JSON 字符串**，不包含任何额外的文字、解释或代码块标记（如 ```json）。JSON 对象必须包含以下字段，并确保其内容符合上述任务要求，尤其是**summary 和 generated_question 必须是完整句子**：
        ```json
        {
          "summary": "这里是金融文章的简洁总结，它必须是一个完整的中文句子。",
          "generated_question": "这里是生成的具体问题，它必须是一个完整的中文问句，并且在相关时包含公司名称和股票代码。",
          "company_name": "提取出的公司全名，若无则为空字符串",
          "stock_code": "提取出的股票代码，若无则为空字符串",
          "report_date": "提取出的报告日期，若无则为空字符串"
        }
        ```
        请确保所有字段都按照示例中的键名和类型严格输出，并且回答的质量达到专业金融分析师水平。
    """).strip()

    final_results_buffer = [] # 用于临时存储处理结果，达到save_interval时写入
    
    # --- 5. Generate in Batches ---
    # tqdm 显示的总进度是从所有记录数开始，但实际迭代是从 start_index 开始
    for current_batch_start_idx in tqdm(
        range(start_index, total_records_in_input, batch_size), 
        initial=start_index, 
        total=total_records_in_input,
        desc="处理数据"
    ):
        batch_records = cleaned_data[current_batch_start_idx : current_batch_start_idx + batch_size]
        
        messages_batch = []
        original_records_in_batch = []
        for record in batch_records:
            context = record.get('context', '').strip()
            answer = record.get('answer', '').strip()
            original_question = record.get('query', '').strip() 
            
            # --- Process newlines in context and answer ---
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
            continue # Skip empty batches

        is_qwen_model = "qwen" in model_name.lower()
        template_args = {"tokenize": False, "add_generation_prompt": True}
        if is_qwen_model:
            template_args["enable_thinking"] = False 

        prompts = [
            tokenizer.apply_chat_template(
                conversation=m, **template_args
            ) for m in messages_batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": 250, 
            "do_sample": True, 
            "pad_token_id": tokenizer.pad_token_id,
            "temperature": 0.7, 
            "top_p": 0.9 
        }
        if is_qwen_model:
            gen_kwargs["top_p"] = 0.95; gen_kwargs["temperature"] = 0.6 
        
        outputs = model.generate(**inputs, **gen_kwargs)

        input_ids_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_ids_len:]
        decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # Collect results for the current batch
        current_batch_results = []
        for idx, generated_text in enumerate(decoded_outputs):
            clean_text = generated_text.split('<|im_end|>')[0].replace(tokenizer.eos_token, '').strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[len("```json"):].strip()
            if clean_text.endswith("```"):
                clean_text = clean_text[:-len("```")].strip()
            
            try:
                generated_json = json.loads(clean_text)
                expected_fields = ["summary", "generated_question", "company_name", "stock_code", "report_date"]
                if all(field in generated_json for field in expected_fields) and \
                   all(generated_json.get(field) is not None and generated_json.get(field).strip() != '' for field in ["summary", "generated_question"]):
                    
                    summary_value = generated_json.get('summary', '').strip()
                    generated_question_value = generated_json.get('generated_question', '').strip()
                    company_name_value = generated_json.get('company_name', '').strip()
                    stock_code_value = generated_json.get('stock_code', '').strip()
                    report_date_value = generated_json.get('report_date', '').strip()

                    record_to_append = {
                        'original_context': original_records_in_batch[idx].get('context', '').strip(), 
                        'original_answer': original_records_in_batch[idx].get('answer', '').strip(), 
                        'original_question': original_records_in_batch[idx].get('query', '').strip(), 
                        # Use original_doc_id if present, otherwise generate one based on its index
                        'doc_id': original_records_in_batch[idx].get('doc_id', f"generated_doc_{current_batch_start_idx + idx}"), 
                        'original_split': original_records_in_batch[idx].get('original_split', None),
                        'original_instruction': original_records_in_batch[idx].get('original_instruction', None), 
                        'summary': summary_value,
                        'generated_question': generated_question_value,
                        'company_name': company_name_value if company_name_value else None,
                        'stock_code': stock_code_value if stock_code_value else None,
                        'report_date': report_date_value if report_date_value else None 
                    }
                    current_batch_results.append(record_to_append)
                else:
                    print(f"警告：跳过无效生成输出 (JSON 缺少期望字段或 summary/generated_question 为空)。原始输入索引: {current_batch_start_idx + idx}\n原始输出：{clean_text}")

            except json.JSONDecodeError:
                print(f"警告：无法解析 JSON 输出，可能格式不正确。原始输入索引: {current_batch_start_idx + idx}\n原始输出：{clean_text}")
            except Exception as e:
                print(f"处理生成输出时发生错误：{e}。原始输入索引: {current_batch_start_idx + idx}\n原始输出：{clean_text}")

        # --- Periodic Save (Append Mode based on total processed count) ---
        # 写入文件时，使用追加模式。每当总处理记录数达到 save_interval * batch_size 的倍数时进行保存
        # 或者在处理到最后一部分数据时 (current_batch_start_idx + batch_size >= total_records_in_input)
        total_processed_so_far = current_batch_start_idx + len(current_batch_results)
        if (total_processed_so_far > 0 and (total_processed_so_far % (save_interval * batch_size) == 0 or total_processed_so_far == total_records_in_input)) and current_batch_results:
            print(f"\n--- 周期性保存: 正在将 {len(current_batch_results)} 条新处理的记录追加到 {output_path} (当前总处理: {total_processed_so_far}) ---")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用临时文件进行原子性写入，防止文件损坏
            temp_output_path = output_path.with_suffix('.tmp')
            
            # 读取旧数据，并追加新数据
            existing_data_for_save = []
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_data_for_save = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: 现有输出文件 '{output_path}' 格式不正确，将尝试追加到新文件。请检查源文件。")
                    existing_data_for_save = [] # 放弃旧数据
                except Exception as e:
                    print(f"读取现有输出文件时发生错误: {e}。将尝试追加到新文件。")
                    existing_data_for_save = []
            
            # 将新处理的数据追加到现有数据
            existing_data_for_save.extend(current_batch_results) 
            
            try:
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data_for_save, f, ensure_ascii=False, indent=4)
                
                # 原子性替换
                Path(temp_output_path).rename(output_path)
                print(f"已成功保存 {len(existing_data_for_save)} 条记录到 {output_path}")
                current_batch_results.clear() # 清空缓冲区
            except Exception as e:
                print(f"保存数据时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                
    # --- 6. Final Save ---
    # 如果循环结束时缓冲区中还有数据，进行最终保存
    # 确保在整个处理完成后，所有 remaining records in current_batch_results are saved
    if current_batch_results: # This handles the very last batch if it doesn't align with save_interval
        # Note: the main periodic save check now also includes `total_processed_so_far == total_records_in_input`
        # so this final save might be redundant if the total count aligns, but it's a good safety net.
        print(f"\n最终保存: 正在保存缓冲区中剩余的 {len(current_batch_results)} 条记录到 {output_path}...")
        try:
            # Re-load existing data to ensure it's up-to-date before final append
            existing_data_for_final_save = []
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_data_for_final_save = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: 现有输出文件 '{output_path}' 格式不正确，将尝试重新写入最终结果。")
                    existing_data_for_final_save = []
                except Exception as e:
                    print(f"读取现有输出文件时发生错误: {e}。将尝试重新写入最终结果。")
                    existing_data_for_final_save = []

            # Append only records that are not already in final_results_buffer (safety check)
            # This logic should generally not be needed if previous saves and current_batch_results are handled correctly
            # But ensures we don't accidentally duplicate
            existing_data_for_final_save.extend(current_batch_results)

            temp_output_path = output_path.with_suffix('.tmp')
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data_for_final_save, f, ensure_ascii=False, indent=4)
            Path(temp_output_path).rename(output_path)
            print(f"已成功保存所有 {len(existing_data_for_final_save)} 条记录到 {output_path}")
            current_batch_results.clear()
        except Exception as e:
            print(f"最终保存数据时发生严重错误: {e}")
            import traceback
            traceback.print_exc()

    print("处理完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generates specific questions for AlphaFin articles using an LLM, including context summarization and metadata extraction, with structured JSON output. Supports checkpointing/resume functionality.",
        epilog="Consider using smaller models like Phi-3-mini for CPU testing."
    )
    
    parser.add_argument("--input_file", type=str, default="data/alphafin/alphafin_rag_ready.json", help="Path to the input preprocessed data file.")
    parser.add_argument("--output_file", type=str, default="data/alphafin/alphafin_summarized_and_structured_qa.json", help="Path to the output RAG-ready file with summaries, generated questions, and metadata. Will be used for checkpointing.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct", help="Name of the Hugging Face model to use. Default is Qwen/Qwen2-7B-Instruct.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process for testing. None to process all.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save progress every N batches.") 
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Falling back to CPU.")
        args.device = "cpu"
    elif args.device == "cpu": 
        print("Running on CPU. Consider using a smaller batch size (e.g., --batch_size 1) and a limit (e.g., --limit 10).")

    process_alphafin_data_with_forced_question_metadata(
        input_path=Path(args.input_file),
        output_path=Path(args.output_file),
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        save_interval=args.save_interval
    )