import json
from pathlib import Path
import re
import hashlib

def generate_content_hash(record: dict, is_original: bool = False) -> str:
    """
    为记录生成一个基于其核心内容的哈希值，用于唯一标识记录。
    根据记录类型（原始或生成）使用不同的键。
    """
    if is_original:
        # 原始文件中通常是 'context', 'query', 'answer'
        # 请根据您的原始JSON文件结构调整这些键名！
        context = record.get('context', '') 
        question = record.get('query', '')
        answer = record.get('answer', '')
    else:
        # LLM生成文件中通常是 'original_context', 'original_question', 'original_answer'
        context = record.get('original_context', '')
        question = record.get('original_question', '')
        answer = record.get('original_answer', '')
    
    # 清理空格和换行符，确保哈希值稳定
    clean_context = re.sub(r'\s+', ' ', context).strip()
    clean_question = re.sub(r'\s+', ' ', question).strip()
    clean_answer = re.sub(r'\s+', ' ', answer).strip()
    clean_content = f"{clean_context}|{clean_question}|{clean_answer}"
    return hashlib.md5(clean_content.encode('utf-8')).hexdigest()

def merge_and_check_missing_data(
    original_json_path: Path, 
    generated_json_paths: list[Path], 
    merged_output_path: Path,
    missing_records_output_path: Path
):
    """
    合并多个LLM生成的JSON文件（部分有doc_id，部分没有），
    并与原始JSON文件进行对比，找出在生成过程中"漏掉"的原始记录。
    采用混合去重和匹配策略：优先doc_id，回退到内容哈希。
    
    Args:
        original_json_path (Path): 原始JSON文件的路径。
        generated_json_paths (list[Path]): LLM生成的JSON文件路径列表。
        merged_output_path (Path): 合并后的生成数据JSON文件的保存路径。
        missing_records_output_path (Path): 漏掉的原始记录保存路径。
    """

    # --- 1. 合并多个生成的 JSON 文件，并进行去重 ---
    print(f"正在合并生成的JSON文件：{[str(p) for p in generated_json_paths]}...")
    
    unique_generated_records_map = {} 

    for gen_path in generated_json_paths:
        try:
            with open(gen_path, 'r', encoding='utf-8') as f:
                current_gen_data = json.load(f)
                for record in current_gen_data:
                    record_key = None
                    # 优先使用 doc_id 去重，如果存在且有效
                    if 'doc_id' in record and record['doc_id'] and str(record['doc_id']).strip() != '' and str(record['doc_id']).strip().lower() != 'none':
                        record_key = f"doc_id_{record['doc_id']}"
                    # 否则回退到内容哈希去重（确保核心字段存在）
                    elif all(k in record for k in ['original_context', 'original_question', 'original_answer']):
                        record_key = f"hash_{generate_content_hash(record, is_original=False)}"
                    
                    if record_key:
                        unique_generated_records_map[record_key] = record # 后面的会覆盖前面的，达到去重效果
                    else:
                        print(f"警告：跳过在文件 {gen_path} 中无法生成唯一键的记录（缺少doc_id或核心内容字段）。")

        except FileNotFoundError:
            print(f"警告：生成文件未找到：{gen_path}")
        except json.JSONDecodeError:
            print(f"错误：生成文件格式不正确：{gen_path}")
        except Exception as e:
            print(f"处理生成文件 {gen_path} 时发生错误: {e}")

    merged_generated_data = list(unique_generated_records_map.values())
    print(f"合并完成。总计 {len(merged_generated_data)} 条唯一的生成记录。")

    # 保存合并后的文件
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_generated_data, f, ensure_ascii=False, indent=4)
    print(f"合并后的生成数据已保存到：{merged_output_path}")

    # --- 2. 加载原始 JSON 数据 ---
    print(f"正在加载原始文件：{original_json_path}...")
    original_data = []
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：原始文件未找到：{original_json_path}")
        return
    except json.JSONDecodeError:
        print(f"错误：原始文件格式不正确：{original_json_path}")
        return
    except Exception as e:
        print(f"加载原始文件 {original_json_path} 时发生错误: {e}")
        return

    print(f"原始文件加载完成。总计 {len(original_data)} 条原始记录。")

    # --- 3. 构建已生成记录的查找集合 (使用内容哈希作为键) ---
    # 这一步是为了与原始数据进行匹配，所以我们统一使用 generate_content_hash
    generated_content_hash_set = set()
    for record in merged_generated_data:
        # LLM生成的数据结构中，原始context等信息在'original_context'等键下
        # 确保这些键存在，否则该条记录无法参与哈希匹配
        if all(k in record for k in ['original_context', 'original_question', 'original_answer']):
             generated_content_hash_set.add(generate_content_hash(record, is_original=False))
        else:
            print(f"警告：生成记录 (doc_id: {record.get('doc_id', 'N/A')}) 缺少 'original_context', 'original_question' 或 'original_answer' 字段，无法生成哈希进行匹配。")


    # --- 4. 找出漏掉的原始记录 ---
    missing_records = []
    for i, original_record in enumerate(original_data):
        # 原始记录的键名可能与LLM生成的不同，这里需要根据原始文件结构来获取
        # 假设您的原始JSON文件结构是 {"context": "...", "query": "...", "answer": "..."}
        # 或者已经是 {"original_context": "...", "original_question": "...", "original_answer": "..."}
        # 
        # 这里尝试同时从原始文件可能的两种键名中获取内容，以增加兼容性
        temp_record_for_hash = {
            'context': original_record.get('original_context', original_record.get('context', '')), 
            'query': original_record.get('original_question', original_record.get('query', '')),
            'answer': original_record.get('original_answer', original_record.get('answer', ''))
        }
        
        # 只有当关键字段都存在且非空时才尝试生成哈希进行匹配
        # 这里判断的是 temp_record_for_hash 中的键是否存在且非空字符串
        if (temp_record_for_hash['context'].strip() != '' and 
            temp_record_for_hash['query'].strip() != '' and 
            temp_record_for_hash['answer'].strip() != ''):
            
            original_record_hash = generate_content_hash(temp_record_for_hash, is_original=True)
            
            if original_record_hash not in generated_content_hash_set:
                missing_records.append(original_record)
        else:
            print(f"警告：原始记录 {i} 缺少核心匹配字段或字段值为空，无法匹配。将其视为漏掉。")
            missing_records.append(original_record) # 如果无法生成哈希进行匹配，也将其视为漏掉

    print(f"检查完成。发现 {len(missing_records)} 条漏掉的原始记录。")

    # --- 5. 保存漏掉的原始记录 ---
    if missing_records:
        missing_records_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(missing_records_output_path, 'w', encoding='utf-8') as f:
            json.dump(missing_records, f, ensure_ascii=False, indent=4)
        print(f"漏掉的原始记录已保存到：{missing_records_output_path}")
    else:
        print("没有发现漏掉的原始记录。")

if __name__ == '__main__':
    # --- 请在这里配置您的文件路径 ---

    # 原始数据文件路径 (LLM处理前的cleaned数据文件)
    # 示例: Path("data/alphafin/alphafin_rag_ready.json")
    original_data_file = Path("data/alphafin/alphafin_rag_ready_0627.json") 

    # LLM生成的多个分文件路径列表
    # 请确保这里包含了您所有分批生成的JSON文件，例如:
    # [Path("data/alphafin/part1.json"), Path("data/alphafin/part2.json")]
    generated_files = [
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0627_b8_s50_fullsentence.json"), 
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0627_colab_backward.json"),
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0628_colab_backward.json"), # 如果有更多部分，请在这里添加
        Path("data/alphafin/alphafin_summarized_and_structured_qa_0628_colab_missing.json"), # 如果有更多部分，请在这里添加p
    ]

    # 合并后的生成数据输出文件路径
    # 示例: Path("data/alphafin/alphafin_merged_generated_qa.json")
    merged_generated_output = Path("data/alphafin/alphafin_merged_generated_qa.json")

    # 漏掉的原始记录输出文件路径
    # 示例: Path("data/alphafin/alphafin_missing_original_records.json")
    missing_records_output = Path("data/alphafin/alphafin_missing_original_records.json")

    # --- 调用整合函数 ---
    merge_and_check_missing_data(
        original_json_path=original_data_file,
        generated_json_paths=generated_files,
        merged_output_path=merged_generated_output,
        missing_records_output_path=missing_records_output
    )

    print("\n任务完成。请检查输出文件。")