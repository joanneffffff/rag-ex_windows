import json
from pathlib import Path
from tqdm import tqdm
import re

def extract_unit_from_paragraph(paragraphs):
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD"
    return ""

def table_to_natural_text(table_dict, caption="", unit_info=""):
    rows = table_dict.get("table", [])
    lines = []

    if caption:
        lines.append(f"Table Topic: {caption}.")

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]

    for i, row in enumerate(data_rows):
        if not row or all(str(v).strip() == "" for v in row):
            continue

        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Table Category: {str(row[0]).strip()}.")
            continue

        row_name = str(row[0]).strip().replace('.', '')

        data_descriptions = []
        for h_idx, v in enumerate(row):
            if h_idx == 0:
                continue
            
            header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"
            value = str(v).strip()

            if value:
                if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value): 
                    formatted_value = value.replace('$', '')
                    if unit_info:
                        if formatted_value.startswith('(') and formatted_value.endswith(')'):
                             formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                        else:
                             formatted_value = f"${formatted_value} {unit_info}"
                    else:
                        formatted_value = f"${formatted_value}"
                else:
                    formatted_value = value
                
                data_descriptions.append(f"{header} is {formatted_value}")

        if row_name and data_descriptions:
            lines.append(f"Details for item {row_name}: {'; '.join(data_descriptions)}.")
        elif data_descriptions:
            lines.append(f"Other data item: {'; '.join(data_descriptions)}.")
        elif row_name:
            lines.append(f"Data item: {row_name}.")

    return "\n".join(lines)

def process_tatqa_to_qca_enhanced(input_paths, output_path):
    all_data = []
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))

    processed_qa_chunks = []

    for item in tqdm(all_data, desc=f"Processing {Path(output_path).name}"):
        doc_paragraphs = item.get("paragraphs", [])
        doc_tables = item.get("tables", [])
        # 处理单个表格的情况
        if "table" in item and not doc_tables:
            doc_tables = [item["table"]]
        
        qa_pairs = item.get("qa_pairs", item.get("questions", []))

        doc_unit_info = extract_unit_from_paragraph(doc_paragraphs)
        
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "")
            
            if isinstance(answer, list):
                answer_str = "; ".join(str(a) for a in answer)
            elif not isinstance(answer, str):
                answer_str = str(answer)
            else:
                answer_str = answer.strip()

            if not question or not answer_str:
                continue

            correct_chunk_content = ""
            answer_type = qa.get("answer_from")
            rel_paragraphs = qa.get("rel_paragraphs", [])
            relevant_doc_ids = []
            
            if answer_type == "text" and rel_paragraphs:
                try:
                    # rel_paragraphs是字符串索引，需要转换为整数
                    p_idx = int(rel_paragraphs[0]) - 1  # 转换为0-based索引
                    if p_idx < len(doc_paragraphs):
                        correct_chunk_content = doc_paragraphs[p_idx].get("text", "")
                        # 使用段落的真实uid作为relevant_doc_ids
                        para_uid = doc_paragraphs[p_idx].get("uid")
                        if para_uid:
                            relevant_doc_ids.append(para_uid)
                except (ValueError, IndexError):
                    pass
            elif answer_type == "table-text":
                t_idx = 0 
                if t_idx < len(doc_tables):
                    correct_chunk_content = table_to_natural_text(doc_tables[t_idx], doc_tables[t_idx].get("caption", ""), doc_unit_info)
                    # 使用表格的真实uid作为relevant_doc_ids
                    table_uid = doc_tables[t_idx].get("uid")
                    if table_uid:
                        relevant_doc_ids.append(table_uid)
            
            if correct_chunk_content.strip():
                # 为当前chunk分配一个唯一的doc_id
                chunk_doc_id = f"chunk_{len(processed_qa_chunks) + 1}"
                
                processed_qa_chunks.append({
                    "query": question,
                    "context": correct_chunk_content.strip(),
                    "answer": answer_str,
                    "doc_id": chunk_doc_id,
                    "relevant_doc_ids": relevant_doc_ids
                })

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in processed_qa_chunks:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated enhanced Q-C-A data (total {len(processed_qa_chunks)} pairs): {output_path}")

if __name__ == "__main__":
    base_raw_data_path = "data/tatqa_dataset_raw/"
    base_output_path = "evaluate_mrr/"

    # 确保输出目录存在
    Path(base_output_path).mkdir(parents=True, exist_ok=True)

    train_dev_inputs = [
        Path(base_raw_data_path) / "tatqa_dataset_train.json",
        Path(base_raw_data_path) / "tatqa_dataset_dev.json"
    ]
    
    # 处理训练和验证数据
    process_tatqa_to_qca_enhanced(
        input_paths=train_dev_inputs,
        output_path=Path(base_output_path) / "tatqa_train_qc_enhanced.jsonl"
    )
    
    eval_inputs = [
        Path(base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    
    # 处理测试数据
    process_tatqa_to_qca_enhanced(
        input_paths=eval_inputs,
        output_path=Path(base_output_path) / "tatqa_eval_enhanced.jsonl"
    ) 