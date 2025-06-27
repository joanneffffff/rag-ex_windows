import json
import re
import os
import random
import ast
from pathlib import Path
from tqdm import tqdm

def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    """
    将原始 JSON 字符串形式的 context 转换为更自然的语言分块。
    处理研报格式、字典格式和纯文本格式，并清除特定的标记和字面换行符。
    """
    chunks = []
    
    if not json_str_context or not json_str_context.strip():
        return chunks

    # === 新增：处理字面上的 "\n" 字符，将其替换为实际的换行符 ===
    processed_str_context = json_str_context.replace("\\n", "\n")

    # 对处理过的 context 进行通用预清理，删除所有匹配到的 【问题】: 和 【答案】:
    cleaned_initial = re.sub(re.escape("【问题】:"), "", processed_str_context)
    cleaned_initial = re.sub(re.escape("【答案】:"), "", cleaned_initial).strip()
    
    # 其他通用字符替换和空格清理
    # 注意：这里您的代码将句号“。”替换为了逗号“,”。
    # 对于中文语义，通常不建议将句号直接替换为逗号，这会改变句子的完整性。
    # 我在这里将其改回不替换句号，只替换其他特殊空格和括号。
    # 如果您确认原始数据中“。”确实代表需要被当做分隔符的“,”，请自行调整。
    cleaned_initial = cleaned_initial.replace('，', ',')
    cleaned_initial = cleaned_initial.replace('：', ':')
    # cleaned_initial = cleaned_initial.replace('。', ',') # 暂时注释掉，保留中文句号
    cleaned_initial = cleaned_initial.replace('【', '') 
    cleaned_initial = cleaned_initial.replace('】', '') 
    cleaned_initial = cleaned_initial.replace('\u3000', ' ')
    cleaned_initial = cleaned_initial.replace('\xa0', ' ').strip()
    # 额外处理可能由 "\n" 替换引入的多余空格或换行
    cleaned_initial = re.sub(r'\s+', ' ', cleaned_initial).strip()

    # --- 尝试解析研报格式 ---
    # 确保 re.DOTALL 标志，让 '.' 匹配换行符
    report_match = re.match(
        r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
        cleaned_initial, 
        re.DOTALL
    )
    
    if report_match:
        report_title_full = report_match.group(1).strip()
        report_date = report_match.group(2).strip()
        report_raw_content = report_match.group(3).strip() 

        content_after_second_title_match = re.match(r"研报题目是:(.+)", report_raw_content, re.DOTALL)
        if content_after_second_title_match:
            report_content_preview = content_after_second_title_match.group(1).strip()
        else:
            report_content_preview = report_raw_content 
            
        # 研报内容内部再次清理
        report_content_preview = re.sub(re.escape("【问题】:"), "", report_content_preview)
        report_content_preview = re.sub(re.escape("【答案】:"), "", report_content_preview).strip()
        report_content_preview = re.sub(r'\s+', ' ', report_content_preview).strip() # 再次清理空格


        company_stock_match = re.search(r"(.+?)（(\d{6}\.\w{2})）", report_title_full)
        company_info = ""
        if company_stock_match:
            report_company_name = company_stock_match.group(1).strip()
            report_stock_code = company_stock_match.group(2).strip()
            company_info = f"，公司名称：{report_company_name}，股票代码：{report_stock_code}"
            report_title_main = re.sub(r"（\d{6}\.\w{2}）", "", report_title_full).strip()
        else:
            report_title_main = report_title_full

        chunk_text = f"一份发布日期为 {report_date} 的研究报告，其标题是：“{report_title_main}”{company_info}。报告摘要内容：{report_content_preview.rstrip('...') if report_content_preview.endswith('...') else report_content_preview}。"
        chunks.append(chunk_text)
        return chunks 

    # --- 优先尝试字典解析 --- (保留您的强大逻辑)
    extracted_dict_str = None
    parsed_data = None 

    # 替换 Timestamp(...) 为字符串形式，使其可以被 ast.literal_eval 或 json.loads 解析
    temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_initial) 
    all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 

    for potential_dict_str in all_dict_matches:
        cleaned_potential_dict_str = potential_dict_str.strip()
        
        # 尝试 JSON 解析
        json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
        try:
            parsed_data_temp = json.loads(json_compatible_str_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break # 成功解析，跳出循环
        except json.JSONDecodeError:
            pass 

        # 尝试 ast.literal_eval 解析 (处理非严格 JSON 格式，如单引号、无引号的键等)
        # 修复 ast.literal_eval 对带有前导零的数字字符串的误判，将其视为字符串
        fixed_for_ast_eval_temp = re.sub(
            r"(?<!['\"\w.])\b(0[1-9]\d*)\b(?![\d.]|['\"\w.])", 
            r"'\1'", 
            cleaned_potential_dict_str
        )
        try:
            parsed_data_temp = ast.literal_eval(fixed_for_ast_eval_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break # 成功解析，跳出循环
        except (ValueError, SyntaxError):
            pass 

    if extracted_dict_str is not None and isinstance(parsed_data, dict):
        for metric_name, time_series_data in parsed_data.items():
            if not isinstance(metric_name, str):
                metric_name = str(metric_name)

            cleaned_metric_name = re.sub(r'（.*?）', '', metric_name).strip()
            
            if not isinstance(time_series_data, dict):
                if time_series_data is not None and str(time_series_data).strip():
                    chunks.append(f"{company_name}的{cleaned_metric_name}数据为：{time_series_data}。")
                continue
            if not time_series_data:
                continue
            
            try:
                sorted_dates = sorted(time_series_data.keys(), key=str)
            except TypeError: # Fallback for non-string keys
                sorted_dates = [str(k) for k in time_series_data.keys()]
                
            description_parts = []
            for date in sorted_dates:
                value = time_series_data[date]
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                description_parts.append(f"在{date}为{formatted_value}")
            
            if description_parts:
                if len(description_parts) <= 3:
                    full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                else:
                    first_part = "，".join(description_parts[:3])
                    last_part = "，".join(description_parts[-3:])
                    if len(sorted_dates) > 6:
                        full_description = f"{company_name}的{cleaned_metric_name}数据从{sorted_dates[0]}到{sorted_dates[-1]}，主要变化为：{first_part}，...，{last_part}。"
                    else:
                        full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                chunks.append(full_description)
        return chunks # Return after successfully parsing and processing a dictionary

    # --- 纯文本回退 ---
    # 如果研报和字典解析均失败，则将其视为纯文本并进一步清理
    pure_text = cleaned_initial
    # 移除日期时间戳等前缀
    pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", pure_text, 1).strip()
    pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()
    pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()
    pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()

    if pure_text: 
        chunks.append(pure_text)
    else:
        # 如果经过所有清理后仍然为空，或者看起来仍像未解析的结构
        print(f"警告：未能在 context 字符串中找到有效结构 (字典、研报或纯文本)。原始字符串（前100字符）：{json_str_context[:100]}...")
        # 返回一个带有特定标记的字符串，以便后续判断为解析失败
        chunks.append(f"原始格式，解析失败或无有效结构：{json_str_context.strip()[:100]}...")

    return chunks


# --- 主脚本逻辑 ---
input_path = 'data/alphafin/alphafin_rag_ready_generated_cleaned.json'
output_dir = 'evaluate_mrr'
os.makedirs(output_dir, exist_ok=True)
eval_jsonl_path = os.path.join(output_dir, 'alphafin_eval.jsonl')
train_jsonl_path = os.path.join(output_dir, 'alphafin_train_qc.jsonl')
train_ratio = 0.8
seed = 42

print(f"正在读取原始数据文件: {input_path}")
with open(input_path, 'r', encoding='utf-8') as f: 
    data = json.load(f)
print(f"原始数据文件包含 {len(data)} 条目。")

qca_list = []
skipped_count = 0

print(f"正在转换数据和清理文本并分配 doc_id...")
for idx, item in tqdm(enumerate(data), total=len(data), desc="Processing items"):
    # === 关键修复：确保 doc_id 被添加到每个条目中 ===
    doc_id = str(idx) # 使用原始索引作为 doc_id

    q_raw = item.get('question', '')
    # 对 question 字段进行全面清理，包括字面上的 "\n"
    q = q_raw.replace("\\n", "\n") # 替换字面换行符
    q = re.sub(re.escape("【问题】:"), "", q).strip() 
    q = re.sub(r'\s+', ' ', q).strip() # 清理多余空格

    a_raw = item.get('answer', '')
    # 对 answer 字段进行全面清理，包括字面上的 "\n"
    a = a_raw.replace("\\n", "\n") # 替换字面换行符
    a = re.sub(re.escape("【答案】："), "", a).strip() 
    a = re.sub(r'\s+', ' ', a).strip() # 清理多余空格

    company_name = item.get('stock_name', '公司')
    
    original_json_context_str = item.get('context', '')
    natural_language_chunks = convert_json_context_to_natural_language_chunks(
        original_json_context_str, company_name=company_name
    )
    
    # 检查 context 是否解析失败 (与 evaluate 脚本中的逻辑一致)
    is_parse_failed_chunk = False
    if not natural_language_chunks:
        is_parse_failed_chunk = True
    elif len(natural_language_chunks) == 1 and (
        natural_language_chunks[0].startswith("原始格式，无有效字典") or # 兼容旧的警告
        natural_language_chunks[0].startswith("原始格式，解析失败") or
        natural_language_chunks[0].startswith("原始格式，无有效结构")
    ):
        is_parse_failed_chunk = True
            
    c = "\n\n".join(natural_language_chunks) # 将所有分块合并为一个 context 字符串
    
    # 只有当 Q、A 存在且 Context 没有解析失败时，才将其添加到列表中
    if q and a and not is_parse_failed_chunk: 
        qca_list.append({
            'query': q, 
            'context': c, 
            'answer': a,
            'doc_id': doc_id # === 关键：在这里添加 doc_id ===
        })
    else:
        skipped_count += 1
        # 打印跳过原因，便于调试
        reason = []
        if not q: reason.append("问题为空")
        if not a: reason.append("答案为空")
        if is_parse_failed_chunk: reason.append("上下文解析失败")
        # print(f"警告：跳过原始 ID {doc_id} 的条目，原因：{'; '.join(reason)}。问题: {q[:50]}..., 答案: {a[:50]}..., 原始 context: {original_json_context_str[:100]}...")


print(f"已处理 {len(qca_list)} 个有效条目。")
print(f"因数据不完整或上下文解析失败而跳过 {skipped_count} 个条目。")

random.seed(seed)
random.shuffle(qca_list)
n_train = int(len(qca_list) * train_ratio)
train_list = qca_list[:n_train]
eval_list = qca_list[n_train:]

print(f"正在写入评测集: {eval_jsonl_path}")
with open(eval_jsonl_path, 'w', encoding='utf-8') as f_eval:
    for item in eval_list:
        f_eval.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"正在写入训练集: {train_jsonl_path}")
with open(train_jsonl_path, 'w', encoding='utf-8') as f_train:
    for item in train_list:
        # 训练集通常只需要 query 和 context
        f_train.write(json.dumps({
            'query': item['query'],
            'context': item['context'],
            'doc_id': item['doc_id'] # 训练集也可以包含 doc_id，方便追溯
        }, ensure_ascii=False) + '\n')

print(f"已生成评测集: {eval_jsonl_path}，共 {len(eval_list)} 条样本")
print(f"已生成训练集: {train_jsonl_path}，共 {len(train_list)} 条样本")