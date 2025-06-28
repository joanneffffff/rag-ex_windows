import json
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
import textwrap
import re 
import sys
import os 

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(model_name: str, device: str):
    print(f"正在加载模型：{model_name}...")
    bnb_config = None
    if device == "cuda":
        print("检测到 CUDA 设备。正在应用 4-bit 量化...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
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
            model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
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
    input_path: Path, output_path: Path, model_name: str, device: str, batch_size: int, limit: int,
    save_interval: int, start_record_index: int, end_record_index: int, failed_output_path: Path 
):
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    print(f"正在从 {input_path} 加载已清理的数据...")
    if not input_path.exists():
        print(f"致命错误：输入文件未找到。请创建 '{input_path.name}'。")
        return
    with open(input_path, 'r', encoding='utf-8') as f:
        full_cleaned_data = json.load(f) 
    if limit:
        print(f"Limiting total records to {limit}.")
        full_cleaned_data = full_cleaned_data[:limit]
    
    if start_record_index is not None or end_record_index is not None: 
        if start_record_index is None: start_record_index = 0
        if end_record_index is None: end_record_index = len(full_cleaned_data)
        if not (0 <= start_record_index < len(full_cleaned_data) or \
                start_record_index == len(full_cleaned_data) and end_record_index == len(full_cleaned_data) ) or \
                start_record_index >= end_record_index or \
                end_record_index > len(full_cleaned_data):
            print(f"警告: 指定的记录范围 ({start_record_index}-{end_record_index}) 无效或超出数据范围 ({0}-{len(full_cleaned_data)-1})。将处理所有记录。")
            cleaned_data = full_cleaned_data
            actual_start_index_in_full_data = 0
            actual_end_index_in_full_data = len(full_cleaned_data)
        else:
            cleaned_data = full_cleaned_data[start_record_index:end_record_index]
            actual_start_index_in_full_data = start_record_index
            actual_end_index_in_full_data = end_record_index
        print(f"将处理原始数据中索引从 {actual_start_index_in_full_data} 到 {actual_end_index_in_full_data-1} 的记录。")
    else:
        cleaned_data = full_cleaned_data
        actual_start_index_in_full_data = 0
        actual_end_index_in_full_data = len(full_cleaned_data)

    total_records_in_slice = len(cleaned_data)
    if total_records_in_slice == 0:
        print("没有记录需要处理。")
        return

    slice_start_index_for_resume = 0 
    if output_path.exists():
        print(f"检测到输出文件 '{output_path}'。尝试加载已处理记录以支持断点续跑...")
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                slice_start_index_for_resume = len(existing_data)
            print(f"已从 '{output_path}' 加载 {slice_start_index_for_resume} 条已处理记录。")
            if slice_start_index_for_resume >= total_records_in_slice:
                 print("当前切片中的所有记录已处理完成。")
                 return
            print(f"将从当前切片的第 {slice_start_index_for_resume} 条记录开始处理。")
        except json.JSONDecodeError:
            print(f"警告: 现有输出文件 '{output_path}' 格式不正确。将从头开始处理当前切片。")
            slice_start_index_for_resume = 0 
        except Exception as e:
            print(f"读取现有输出文件时发生错误: {e}。将从头开始处理当前切片。")
            slice_start_index_for_resume = 0
    
    system_prompt = textwrap.dedent("""
        你是一名专业的金融分析师。你的任务是根据提供的中文金融新闻文章（Context）和人工编写的答案（Answer），严格、清晰地完成以下三项任务：
        
        **任务 1：生成简洁、完整的上下文总结 (summary)**
        对“Context”和“Answer”的核心语义信息进行简洁、精炼的总结。这个总结必须是**一个或多个完整、通顺的中文句子**，精准捕捉文章的关键点和“Answer”所表达的核心内容。请确保总结流畅且信息完整。
        
        **任务 2：生成一个具体、完整的问题 (generated_question)**
        根据“Context”和“Answer”生成一个单一的、具体的问题。这个问题必须是**一个完整、清晰的中文问句**，能够被给定的“Answer”直接且完全地回答，并且只能使用“Context”中包含的信息。
        **如果“Context”中明确提及了公司名称和/或股票代码，你生成的问题中必须明确提及这些信息，以提高问题特异性。例如，如果Context提到了“腾讯控股”和“0700.HK”，你的问题可以是“关于腾讯控股(0700.HK)的最新财报显示了哪些信息？”。如果Context中没有明确的公司名称或股票代码，则问题中无需包含，但仍需确保问题是一个完整且有意义的句子。**
        
        **特别注意：如果“Answer”是关于股票“涨”、“跌”或“波动”的预测，请确保生成的问题是关于该股票的“下月涨跌预测”、“未来收益表现预测”或“近期市场走势预测”等相关主题。**
        
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

    current_batch_results_buffer = [] 
    failed_records_buffer = []       
    
    print(f"正在批量处理切片中的记录，批大小为 {batch_size}...")
    
    for batch_idx_in_slice in tqdm(
        range(slice_start_index_for_resume, total_records_in_slice, batch_size),
        initial=slice_start_index_for_resume // batch_size, # 初始批次索引
        total=(total_records_in_slice + batch_size - 1) // batch_size, # 总批次数量
        desc="处理数据"
    ):
        batch_records = cleaned_data[batch_idx_in_slice : batch_idx_in_slice + batch_size]
        
        messages_batch = []
        original_records_in_batch = []
        for record in batch_records:
            context = record.get('context', '').strip()
            answer = record.get('answer', '').strip()
            original_question = record.get('query', '').strip() 
            # 关键：获取 original_instruction
            original_instruction_text = record.get('original_instruction', '').strip() # <<< 这里获取 original_instruction
            
            context = re.sub(r'\s+', ' ', context).strip()
            answer = re.sub(r'\s+', ' ', answer).strip()

            if context and answer:
                user_content = f"Context:\n{context}\n\n"
                # 关键：将 original_instruction 加入 user_content
                if original_instruction_text:
                    user_content += f"Original Task Instruction:\n{original_instruction_text}\n\n" # <<< 这里加入
                
                user_content += f"Answer:\n{answer}"

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
            template_args["enable_thinking"] = False 

        prompts = [
            tokenizer.apply_chat_template(
                conversation=m, **template_args
            ) for m in messages_batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": 400, # <<-- 已根据 JSON 截断问题调整为 400
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

        for idx, generated_text in enumerate(decoded_outputs):
            clean_text = generated_text.split('<|im_end|>')[0].replace(tokenizer.eos_token, '').strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[len("```json"):].strip()
            if clean_text.endswith("```"):
                clean_text = clean_text[:-len("```")].strip()
            
            success_parsing = False
            try:
                generated_json = json.loads(clean_text)
                expected_fields = ["summary", "generated_question", "company_name", "stock_code", "report_date"]
                if all(field in generated_json for field in expected_fields) and \
                   all(generated_json.get(field) is not None and generated_json.get(field).strip() != '' for field in ["summary", "generated_question"]):
                    success_parsing = True
                    original_record_full_idx = actual_start_index_in_full_data + batch_idx_in_slice + idx
                    record_to_append = {
                        'original_context': original_records_in_batch[idx].get('context', '').strip(), 
                        'original_answer': original_records_in_batch[idx].get('answer', '').strip(), 
                        'original_question': original_records_in_batch[idx].get('query', '').strip(), 
                        'doc_id': original_records_in_batch[idx].get('doc_id', f"generated_doc_{original_record_full_idx}"), 
                        'original_split': original_records_in_batch[idx].get('original_split', None),
                        'original_instruction': original_records_in_batch[idx].get('original_instruction', None), # 确保 original_instruction 被带到输出
                        'summary': generated_json.get('summary', '').strip(),
                        'generated_question': generated_json.get('generated_question', '').strip(),
                        'company_name': generated_json.get('company_name', '').strip() if generated_json.get('company_name') else None,
                        'stock_code': generated_json.get('stock_code', '').strip() if generated_json.get('stock_code') else None,
                        'report_date': generated_json.get('report_date', '').strip() if generated_json.get('report_date') else None 
                    }
                    current_batch_results_buffer.append(record_to_append)
                else:
                    print(f"警告：跳过无效生成输出 (JSON 缺少期望字段或 summary/generated_question 为空)。原始切片索引: {batch_idx_in_slice + idx}\n原始输出：{clean_text}")

            except json.JSONDecodeError:
                print(f"警告：无法解析 JSON 输出，可能格式不正确。原始切片索引: {batch_idx_in_slice + idx}\n原始输出：{clean_text}")
            except Exception as e:
                print(f"处理生成输出时发生错误：{e}。原始切片索引: {batch_idx_in_slice + idx}\n原始输出：{clean_text}")

            if not success_parsing:
                failed_original_record = original_records_in_batch[idx].copy() 
                failed_original_record['_failed_reason'] = clean_text 
                failed_original_record['_error_message'] = str(e) if 'e' in locals() else 'JSON parsing failed'
                failed_records_buffer.append(failed_original_record)


        total_records_processed_in_this_run = (batch_idx_in_slice - slice_start_index_for_resume) + len(current_batch_results_buffer)
        
        should_save = False
        if len(current_batch_results_buffer) >= (save_interval * batch_size): 
            should_save = True
        elif (batch_idx_in_slice + batch_size >= total_records_in_slice) and current_batch_results_buffer: 
            should_save = True
        
        if should_save:
            print(f"\n--- 周期性保存: 正在将 {len(current_batch_results_buffer)} 条新处理的记录追加到 {output_path} (当前切片总处理: {batch_idx_in_slice + len(current_batch_results_buffer)}) ---")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            temp_output_path = output_path.with_suffix('.tmp')
            
            existing_data_for_save = []
            if output_path.exists():
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        existing_data_for_save = json.load(f)
                except json.JSONDecodeError:
                    print(f"警告: 现有输出文件 '{output_path}' 格式不正确，将尝试追加到新文件。请检查源文件。")
                    existing_data_for_save = [] 
                except Exception as e:
                    print(f"读取现有输出文件时发生错误: {e}。将尝试追加到新文件。")
                    existing_data_for_save = []
            
            existing_data_for_save.extend(current_batch_results_buffer) 
            
            try:
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data_for_save, f, ensure_ascii=False, indent=4)
                
                Path(temp_output_path).rename(output_path)
                print(f"已成功保存 {len(existing_data_for_save)} 条记录到 {output_path}")
                current_batch_results_buffer.clear() 
            except Exception as e:
                print(f"保存数据时发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                
    if current_batch_results_buffer: 
        print(f"\n最终保存 (主输出): 正在保存缓冲区中剩余的 {len(current_batch_results_buffer)} 条记录到 {output_path}...")
        try:
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

            existing_data_for_final_save.extend(current_batch_results_buffer)

            temp_output_path = output_path.with_suffix('.tmp')
            with open(temp_output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data_for_final_save, f, ensure_ascii=False, indent=4)
            Path(temp_output_path).rename(output_path)
            print(f"已成功保存所有 {len(existing_data_for_final_save)} 条记录到 {output_path}")
            current_batch_results_buffer.clear()
        except Exception as e:
            print(f"最终保存主输出数据时发生严重错误: {e}")
            import traceback
            traceback.print_exc()
    
    if failed_records_buffer:
        print(f"\n最终保存 (失败样本): 正在保存 {len(failed_records_buffer)} 条生成失败的原始记录到 {failed_output_path}...")
        failed_output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            temp_failed_output_path = failed_output_path.with_suffix('.tmp_failed')
            with open(temp_failed_output_path, 'w', encoding='utf-8') as f:
                json.dump(failed_records_buffer, f, ensure_ascii=False, indent=4)
            Path(temp_failed_output_path).rename(failed_output_path)
            print(f"已成功保存 {len(failed_records_buffer)} 条失败样本到 {failed_output_path}")
            failed_records_buffer.clear()
        except Exception as e:
            print(f"最终保存失败样本数据时发生严重错误: {e}")
            import traceback
            traceback.print_exc()

    print("处理完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generates specific questions for AlphaFin articles using an LLM, including context summarization and metadata extraction, with structured JSON output. Supports checkpointing/resume functionality. Can process a specific range of records (slice). Collects and saves failed records separately.",
        epilog="Consider using smaller models like Phi-3-mini for CPU testing."
    )
    
    parser.add_argument("--input_file", type=str, default="data/alphafin/alphafin_rag_ready.json", help="Path to the input preprocessed data file.")
    parser.add_argument("--output_file", type=str, default="data/alphafin/alphafin_summarized_and_structured_qa.json", help="Path to the output RAG-ready file with summaries, generated questions and metadata. Will be used for checkpointing.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-7B-Instruct", help="Name of the Hugging Face model to use. Default is Qwen/Qwen2-7B-Instruct.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for generation.")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process for testing. None to process all.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save progress every N batches.") 
    parser.add_argument("--start_record_index", type=int, default=None, help="Starting index (inclusive) of records to process from the input file. 0-based.")
    parser.add_argument("--end_record_index", type=int, default=None, help="Ending index (exclusive) of records to process from the input file. (e.g., use total_records for end).")
    parser.add_argument("--failed_output_path", type=str, default="data/alphafin/failed_llm_outputs.json", help="Path to save original records that failed LLM generation/parsing.") 
    
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
        save_interval=args.save_interval,
        start_record_index=args.start_record_index,
        end_record_index=args.end_record_index,
        failed_output_path=Path(args.failed_output_path)
    )