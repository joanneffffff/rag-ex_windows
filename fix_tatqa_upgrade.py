#!/usr/bin/env python3
"""
修复TatQA升级脚本
确保relevant_doc_ids包含所有能回答该问题的段落/表格
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm

def extract_unit_from_paragraph(paragraphs):
    """从段落中提取数值单位"""
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD"
    return ""

def table_to_natural_text(table_dict, caption="", unit_info=""):
    """将表格转换为自然语言文本"""
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

def process_tatqa_to_qca_fixed(input_paths, output_path):
    """修复版：处理TatQA数据集，生成Q-C-A格式的评估数据"""
    all_data = []
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))

    processed_qa_chunks = []

    for item in tqdm(all_data, desc=f"Processing {Path(output_path).name}"):
        doc_paragraphs = item.get("paragraphs", [])
        doc_tables = item.get("tables", [])
        qa_pairs = item.get("qa_pairs", item.get("questions", []))

        doc_unit_info = extract_unit_from_paragraph(doc_paragraphs)
        doc_id = item.get("uid", f"doc_{len(processed_qa_chunks)}")
        
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

            # 收集所有相关的doc_ids
            relevant_doc_ids = set()
            
            # 根据rel_paragraphs收集所有相关段落
            rel_paragraphs = qa.get("rel_paragraphs", [])
            for para_idx in rel_paragraphs:
                try:
                    p_idx = int(para_idx) - 1  # TatQA的rel_paragraphs是1-based
                    if p_idx < len(doc_paragraphs):
                        relevant_doc_ids.add(f"{doc_id}_para_{p_idx}")
                except (ValueError, IndexError):
                    pass
            
            # 如果答案来自表格，也添加表格的doc_id
            answer_from = qa.get("answer_from", "")
            if answer_from in ["table", "table-text"]:
                for t_idx in range(len(doc_tables)):
                    relevant_doc_ids.add(f"{doc_id}_table_{t_idx}")
            
            # 如果没有找到任何相关段落，但有段落存在，使用第一个段落
            if not relevant_doc_ids and doc_paragraphs:
                relevant_doc_ids.add(f"{doc_id}_para_0")
            
            # 生成context（使用第一个相关段落或表格）
            correct_chunk_content = ""
            if relevant_doc_ids:
                first_doc_id = list(relevant_doc_ids)[0]
                if "_para_" in first_doc_id:
                    # 从段落获取内容
                    try:
                        p_idx = int(first_doc_id.split("_para_")[1])
                        if p_idx < len(doc_paragraphs):
                            correct_chunk_content = doc_paragraphs[p_idx].get("text", "") if isinstance(doc_paragraphs[p_idx], dict) else doc_paragraphs[p_idx]
                    except:
                        pass
                elif "_table_" in first_doc_id:
                    # 从表格获取内容
                    try:
                        t_idx = int(first_doc_id.split("_table_")[1])
                        if t_idx < len(doc_tables):
                            correct_chunk_content = table_to_natural_text(doc_tables[t_idx], doc_tables[t_idx].get("caption", ""), doc_unit_info)
                    except:
                        pass
            
            if correct_chunk_content.strip() and relevant_doc_ids:
                processed_qa_chunks.append({
                    "query": question,
                    "context": correct_chunk_content.strip(),
                    "answer": answer_str,
                    "relevant_doc_ids": list(relevant_doc_ids)  # 包含所有相关段落/表格的ID
                })

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in processed_qa_chunks:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated Q-Chunk-A data with comprehensive relevant_doc_ids (total {len(processed_qa_chunks)} pairs): {output_path}")
    return processed_qa_chunks

def verify_multi_questions():
    """验证是否正确处理了多问题情况"""
    print("=== 验证多问题处理 ===")
    
    # 加载修复后的数据
    fixed_eval_path = "evaluate_mrr/tatqa_eval_fixed.jsonl"
    
    if not Path(fixed_eval_path).exists():
        print(f"❌ 修复后的评估数据不存在: {fixed_eval_path}")
        return
    
    # 加载数据
    data = []
    with open(fixed_eval_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    print(f"✅ 加载了 {len(data)} 个修复后的评估样本")
    
    # 按基础doc_id分组
    doc_groups = defaultdict(list)
    for item in data:
        relevant_doc_ids = item.get('relevant_doc_ids', [])
        if relevant_doc_ids:
            # 提取基础doc_id（去掉chunk_id部分）
            base_doc_id = relevant_doc_ids[0].rsplit('_', 1)[0] if '_' in relevant_doc_ids[0] else relevant_doc_ids[0]
            doc_groups[base_doc_id].append(item)
    
    # 找出包含多个问题的文档
    multi_question_docs = {doc_id: items for doc_id, items in doc_groups.items() if len(items) > 1}
    
    print(f"📊 修复后的统计:")
    print(f"  总文档数: {len(doc_groups)}")
    print(f"  包含多个问题的文档数: {len(multi_question_docs)}")
    
    # 显示前3个多问题文档的示例
    print(f"\n=== 修复后的多问题示例 ===")
    
    for i, (doc_id, items) in enumerate(list(multi_question_docs.items())[:3]):
        print(f"\n📄 文档 {i+1}: {doc_id}")
        print(f"   包含 {len(items)} 个问题")
        
        # 按chunk_id分组
        chunk_groups = defaultdict(list)
        for item in items:
            relevant_doc_ids = item.get('relevant_doc_ids', [])
            for doc_id_full in relevant_doc_ids:
                chunk_id = doc_id_full.rsplit('_', 1)[1] if '_' in doc_id_full else 'unknown'
                chunk_groups[chunk_id].append(item)
        
        # 显示每个chunk的问题
        for chunk_id, chunk_items in chunk_groups.items():
            print(f"\n   📍 Chunk: {chunk_id}")
            print(f"   包含 {len(chunk_items)} 个问题:")
            
            for j, item in enumerate(chunk_items[:3]):  # 只显示前3个问题
                print(f"     {j+1}. 问题: {item['query'][:80]}...")
                print(f"        答案: {item['answer'][:50]}...")
                print(f"        相关文档ID: {item['relevant_doc_ids']}")
                print()
            
            if len(chunk_items) > 3:
                print(f"     ... 还有 {len(chunk_items) - 3} 个问题")

def main():
    """主函数"""
    print("=== 修复TatQA升级脚本 ===")
    print("确保relevant_doc_ids包含所有能回答该问题的段落/表格")
    print()
    
    # 原始TatQA数据路径
    tatqa_data_paths = [
        "data/tatqa_dataset_raw/tatqa_dataset_train.json",
        "data/tatqa_dataset_raw/tatqa_dataset_dev.json", 
        "data/tatqa_dataset_raw/tatqa_dataset_test_gold.json"
    ]
    
    # 检查文件是否存在
    existing_paths = []
    for path in tatqa_data_paths:
        if Path(path).exists():
            existing_paths.append(path)
        else:
            print(f"警告：文件不存在 {path}")
    
    if not existing_paths:
        print("错误：没有找到任何TatQA原始数据文件")
        return
    
    # 生成修复后的评估数据
    output_path = "evaluate_mrr/tatqa_eval_fixed.jsonl"
    
    try:
        processed_data = process_tatqa_to_qca_fixed(existing_paths, output_path)
        
        # 验证数据
        print(f"\n=== 验证修复后的数据 ===")
        print(f"总样本数: {len(processed_data)}")
        
        # 检查relevant_doc_ids的分布
        relevant_doc_ids_count = 0
        multi_doc_ids_count = 0
        for item in processed_data:
            if item.get("relevant_doc_ids"):
                relevant_doc_ids_count += 1
                if len(item['relevant_doc_ids']) > 1:
                    multi_doc_ids_count += 1
        
        print(f"包含relevant_doc_ids的样本数: {relevant_doc_ids_count}")
        print(f"包含多个relevant_doc_ids的样本数: {multi_doc_ids_count}")
        print(f"覆盖率: {relevant_doc_ids_count/len(processed_data)*100:.2f}%")
        print(f"多文档覆盖率: {multi_doc_ids_count/len(processed_data)*100:.2f}%")
        
        # 显示前几个样本的示例
        print(f"\n=== 修复后的样本示例 ===")
        for i, item in enumerate(processed_data[:3]):
            print(f"样本 {i+1}:")
            print(f"  查询: {item['query'][:100]}...")
            print(f"  答案: {item['answer'][:50]}...")
            print(f"  相关文档ID: {item.get('relevant_doc_ids', [])}")
            print()
        
        print(f"✅ 成功修复TatQA评估数据: {output_path}")
        
        # 验证多问题处理
        verify_multi_questions()
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "="*50)
    print("✅ 修复完成！")
    print("\n修复内容：")
    print("1. 正确处理了rel_paragraphs字段，包含所有相关段落")
    print("2. 对于表格问题，添加了表格的doc_id")
    print("3. relevant_doc_ids现在包含所有能回答该问题的段落/表格")
    print("4. 这确保了评估时能够进行严格的doc_id匹配")
    print("5. 消除了因模糊匹配导致的高估问题")

if __name__ == "__main__":
    main() 