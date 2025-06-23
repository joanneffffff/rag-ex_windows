import json
import os
import random
import ast
import re
from pathlib import Path
from tqdm import tqdm

# === 请将上面最新版本的 convert_json_context_to_natural_language_chunks 函数复制到这里 ===
def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    chunks = []
    
    if not json_str_context or not json_str_context.strip():
        return chunks

    # 步骤1：预清理一些常见的前缀和标点，以便更准确地提取字典和纯文本
    cleaned_for_processing = json_str_context.replace("【问题】:", "").replace("【答案】:", "").strip()
    cleaned_for_processing = cleaned_for_processing.replace('，', ',')
    cleaned_for_processing = cleaned_for_processing.replace('：', ':')
    cleaned_for_processing = cleaned_for_processing.replace('。', ',')
    cleaned_for_processing = cleaned_for_processing.replace('【', '')
    cleaned_for_processing = cleaned_for_processing.replace('】', '')
    cleaned_for_processing = cleaned_for_processing.replace('\u3000', ' ')
    cleaned_for_processing = cleaned_for_processing.replace('\xa0', ' ').strip() # 处理 &nbsp; 这样的特殊空格

    # --- NEW: 尝试解析研报格式 (优化后的正则表达式) ---
    # 匹配 '这是以...为题目,在...日期发布的研究报告。研报内容如下: 研报题目是:...'
    # 调整研报内容捕获，使其能更鲁棒地处理后面的内容，包括重复的“研报题目是”
    # 将第三个捕获组从 `(.+)` 调整为更具体地从 `研报题目是:` 之后开始捕获
    report_match = re.match(
        r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
        cleaned_for_processing, 
        re.DOTALL
    )
    
    if report_match:
        report_title_full = report_match.group(1).strip()
        report_date = report_match.group(2).strip()
        report_raw_content = report_match.group(3).strip() # 捕获到原始的“研报题目是”部分

        # 从 report_raw_content 中再次提取实际的“研报题目是”后面的内容
        content_after_second_title_match = re.match(r"研报题目是:(.+)", report_raw_content, re.DOTALL)
        if content_after_second_title_match:
            report_content_preview = content_after_second_title_match.group(1).strip()
        else:
            report_content_preview = report_raw_content # 如果没有第二个“研报题目是:”，则使用捕获到的全部内容

        # 进一步从 report_title_full 中提取公司名称和股票代码，如果存在的话
        company_stock_match = re.search(r"(.+?)（(\d{6}\.\w{2})）", report_title_full)
        company_info = ""
        if company_stock_match:
            report_company_name = company_stock_match.group(1).strip()
            report_stock_code = company_stock_match.group(2).strip()
            company_info = f"，公司名称：{report_company_name}，股票代码：{report_stock_code}"
            # 移除标题中的公司信息，只保留主题
            report_title_main = re.sub(r"（\d{6}\.\w{2}）", "", report_title_full).strip()
        else:
            report_title_main = report_title_full

        chunk_text = f"一份发布日期为 {report_date} 的研究报告，其标题是：“{report_title_main}”{company_info}。报告摘要内容：{report_content_preview.rstrip('...') if report_content_preview.endswith('...') else report_content_preview}。"
        chunks.append(chunk_text)
        return chunks # 如果匹配到研报格式，直接返回

    # --- 优先尝试字典解析 (与之前的逻辑相同) ---
    extracted_dict_str = None
    parsed_data = None 

    temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_for_processing)
    all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 

    for potential_dict_str in all_dict_matches:
        cleaned_potential_dict_str = potential_dict_str.strip()
        
        # 尝试 json.loads
        json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
        try:
            parsed_data_temp = json.loads(json_compatible_str_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break 
        except json.JSONDecodeError:
            pass 

        # 尝试 ast.literal_eval
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
                break 
        except (ValueError, SyntaxError):
            pass 

    if extracted_dict_str is not None and isinstance(parsed_data, dict):
        # 如果成功解析出字典，则按字典方式处理
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
            except TypeError: # handles cases where keys might be mixed types
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
    else:
        # 如果没有成功解析出字典，也没有匹配到研报格式，则假定是普通纯文本内容
        # 移除可能存在的日期前缀 (例如 "2022-04-27_;" 或 "2023-08-14 16:48:44_;")
        pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", cleaned_for_processing, 1).strip()
        
        # 进一步清理可能存在的像 "华尔街见闻/严婷2015年05月25日23:19:02" 这种更复杂的前缀
        # 匹配开头是中文、斜杠、日期、时间、中文、逗号的模式
        pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()

        # 处理像 "市场资金进出截至周一收盘," 这样的头部
        pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()

        # 处理像 "渤海活塞中期净利预减60%-70%渤海活塞7月24日晚间公告," 这样的头部
        pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()


        if pure_text: 
            chunks.append(pure_text)
        else:
            # 如果清理后还是空，或者就是完全无法识别的格式，则返回一个通用失败信息
            print(f"警告：未能在 context 字符串中找到有效结构 (字典、研报或纯文本)。原始字符串（前100字符）：{json_str_context[:100]}...")
            chunks.append(f"{company_name}的相关数据（原始格式，无有效结构）：{json_str_context.strip()}")

    return chunks


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
for item in tqdm(data, desc="正在转换 context 为自然语言"):
    q = item.get('question', '')
    a = item.get('answer', '')
    company_name = item.get('stock_name', '公司')
    
    original_json_context_str = item.get('context', '')
    natural_language_chunks = convert_json_context_to_natural_language_chunks(
        original_json_context_str, company_name=company_name
    )
    
    # 检查是否包含表示解析失败的通用错误信息，如果有，则跳过
    is_parse_failed_chunk = False
    if not natural_language_chunks: # chunks 为空
        is_parse_failed_chunk = True
    elif len(natural_language_chunks) == 1 and ("原始格式，无有效字典" in natural_language_chunks[0] or "原始格式，解析失败" in natural_language_chunks[0] or "原始格式，无有效结构" in natural_language_chunks[0]):
        is_parse_failed_chunk = True
        
    c = "\n\n".join(natural_language_chunks) 
    
    if q and a and not is_parse_failed_chunk: 
        qca_list.append({'query': q, 'context': c, 'answer': a})
    else:
        if is_parse_failed_chunk:
            print(f"警告：跳过无效 QCA 条目（context 解析失败）。问题: {q[:50]}, 答案: {a[:50]}. 原始 context: {original_json_context_str[:100]}...")
        else:
            print(f"警告：跳过无效 QCA 条目（问题或答案为空）。问题: {q[:50]}, 答案: {a[:50]}")


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
        f_train.write(json.dumps({
            'query': item['query'],
            'context': item['context']
        }, ensure_ascii=False) + '\n')

print(f"已生成评测集: {eval_jsonl_path}，共{len(eval_list)}条样本")
print(f"已生成训练集: {train_jsonl_path}，共{len(train_list)}条样本")