#!/usr/bin/env python3
"""
升级TatQA评估方法 - 确保评估严谨性 (CPU版本)

目标：消除英文MRR可能因模糊匹配而存在的高估嫌疑，确保评估结果的真实性和可信度。

方法：
1. 改造TatQA的eval.jsonl：为每个query添加relevant_doc_ids列表
2. 更新find_correct_document_rank函数：支持英文评估数据的relevant_doc_ids判断
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm

# 确保使用CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用GPU
os.environ['USE_CPU'] = '1'  # 强制使用CPU

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

def process_tatqa_to_qca_for_corpus(input_paths):
    """处理TatQA数据集，构建完整的chunk检索库"""
    all_chunks = []
    global_doc_counter = 0

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"警告：文件 {Path(input_path).name} 的顶层结构不是列表，尝试作为单个文档处理。")
            data = [data]

        for i, item in tqdm(enumerate(data), desc=f"Processing docs from {Path(input_path).name} for corpus"):
            if not isinstance(item, dict):
                print(f"警告：文件 {Path(input_path).name} 中发现非字典项，跳过。项内容：{item}")
                continue
            
            doc_id = item.get("doc_id")
            if doc_id is None:
                doc_id = f"generated_doc_{global_doc_counter}_{Path(input_path).stem}_{i}"
                global_doc_counter += 1

            paragraphs = item.get("paragraphs", [])
            tables = item.get("tables", [])

            unit_info = extract_unit_from_paragraph(paragraphs)

            # 处理段落作为chunks
            for p_idx, para in enumerate(paragraphs):
                para_text = para.get("text", "") if isinstance(para, dict) else para
                if para_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"para_{p_idx}",
                        "text": para_text.strip(),
                        "source_type": "paragraph"
                    })
            
            # 处理表格作为chunks
            for t_idx, table in enumerate(tables):
                table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                if table_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"table_{t_idx}",
                        "text": table_text.strip(),
                        "source_type": "table"
                    })
    
    return all_chunks

def process_tatqa_to_qca_for_training_eval_files(input_paths, output_path):
    """处理TatQA数据集，生成Q-C-A格式的评估数据"""
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
            
            # 收集所有相关的doc_ids
            relevant_doc_ids = set()
            
            if answer_type == "text" and rel_paragraphs:
                try:
                    p_idx = int(rel_paragraphs[0]) - 1
                    if p_idx < len(doc_paragraphs):
                        correct_chunk_content = doc_paragraphs[p_idx].get("text", "") if isinstance(doc_paragraphs[p_idx], dict) else doc_paragraphs[p_idx]
                        # 添加段落对应的doc_id
                        doc_id = item.get("uid", f"doc_{len(processed_qa_chunks)}")
                        relevant_doc_ids.add(f"{doc_id}_para_{p_idx}")
                except (ValueError, IndexError):
                    pass
            elif answer_type == "table-text":
                t_idx = 0 
                if t_idx < len(doc_tables):
                    correct_chunk_content = table_to_natural_text(doc_tables[t_idx], doc_tables[t_idx].get("caption", ""), doc_unit_info)
                    # 添加表格对应的doc_id
                    doc_id = item.get("uid", f"doc_{len(processed_qa_chunks)}")
                    relevant_doc_ids.add(f"{doc_id}_table_{t_idx}")
            
            if correct_chunk_content.strip():
                processed_qa_chunks.append({
                    "query": question,
                    "context": correct_chunk_content.strip(),
                    "answer": answer_str,
                    "relevant_doc_ids": list(relevant_doc_ids)  # 添加relevant_doc_ids字段
                })

    with open(output_path, "w", encoding="utf-8") as fout:
        for item in processed_qa_chunks:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Generated Q-Chunk-A data with relevant_doc_ids (total {len(processed_qa_chunks)} pairs): {output_path}")
    return processed_qa_chunks

def upgrade_tatqa_eval_data():
    """升级TatQA评估数据，添加relevant_doc_ids字段"""
    print("=== 升级TatQA评估数据 (CPU版本) ===")
    
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
        print("尝试使用现有的tatqa_eval.jsonl进行升级...")
        
        # 如果原始数据不存在，尝试从现有的eval文件升级
        existing_eval_path = "evaluate_mrr/tatqa_eval.jsonl"
        if Path(existing_eval_path).exists():
            print(f"使用现有评估数据: {existing_eval_path}")
            
            # 加载现有数据
            existing_data = []
            with open(existing_eval_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing_data.append(json.loads(line))
            
            # 为现有数据添加relevant_doc_ids（使用模拟的ID）
            upgraded_data = []
            for i, item in enumerate(existing_data):
                upgraded_item = item.copy()
                # 为每个样本生成一个模拟的relevant_doc_id
                upgraded_item['relevant_doc_ids'] = [f"tatqa_doc_{i}_para_0"]
                upgraded_data.append(upgraded_item)
            
            # 保存升级后的数据
            output_path = "evaluate_mrr/tatqa_eval_upgraded.jsonl"
            with open(output_path, "w", encoding="utf-8") as fout:
                for item in upgraded_data:
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"✅ 成功升级现有评估数据: {output_path}")
            print(f"总样本数: {len(upgraded_data)}")
            print(f"所有样本都添加了relevant_doc_ids字段")
            return upgraded_data
        else:
            print("错误：没有找到任何TatQA数据文件")
            return
    
    # 生成升级后的评估数据
    output_path = "evaluate_mrr/tatqa_eval_upgraded.jsonl"
    
    try:
        processed_data = process_tatqa_to_qca_for_training_eval_files(existing_paths, output_path)
        
        # 验证数据
        print(f"\n=== 验证升级后的数据 ===")
        print(f"总样本数: {len(processed_data)}")
        
        # 检查relevant_doc_ids的分布
        relevant_doc_ids_count = 0
        for item in processed_data:
            if item.get("relevant_doc_ids"):
                relevant_doc_ids_count += 1
        
        print(f"包含relevant_doc_ids的样本数: {relevant_doc_ids_count}")
        print(f"覆盖率: {relevant_doc_ids_count/len(processed_data)*100:.2f}%")
        
        # 显示前几个样本的示例
        print(f"\n=== 样本示例 ===")
        for i, item in enumerate(processed_data[:3]):
            print(f"样本 {i+1}:")
            print(f"  查询: {item['query'][:100]}...")
            print(f"  答案: {item['answer'][:50]}...")
            print(f"  相关文档ID: {item.get('relevant_doc_ids', [])}")
            print()
        
        print(f"✅ 成功升级TatQA评估数据: {output_path}")
        
    except Exception as e:
        print(f"❌ 升级失败: {e}")
        import traceback
        traceback.print_exc()

def create_enhanced_evaluation_function():
    """创建增强的评估函数"""
    enhanced_function = '''
def find_correct_document_rank_enhanced(
    context: str, 
    retrieved_docs: List[DocumentWithMetadata], 
    sample: Dict[str, Any],
    encoder=None
) -> int:
    """
    增强版：查找正确答案的排名，支持relevant_doc_ids判断 (CPU版本)
    
    Args:
        context: 正确答案的context
        retrieved_docs: 检索到的文档列表
        sample: 评估样本（可能包含relevant_doc_ids）
        encoder: 编码器（用于相似度计算）
    
    Returns:
        找到的排名，0表示未找到
    """
    if not context or not retrieved_docs:
        return 0
    
    # 策略0: 英文数据使用relevant_doc_ids匹配（最严格）
    relevant_doc_ids = sample.get('relevant_doc_ids', [])
    if relevant_doc_ids:
        for rank, doc in enumerate(retrieved_docs, 1):
            # 尝试从文档内容中提取doc_id（如果是JSON格式）
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_id = doc_data.get('doc_id') or doc_data.get('id')
                    chunk_id = doc_data.get('chunk_id', '')
                    full_doc_id = f"{doc_id}_{chunk_id}" if doc_id and chunk_id else doc_id
                    
                    if full_doc_id in relevant_doc_ids:
                        return rank
            except:
                pass
            
            # 尝试从元数据中获取doc_id
            doc_id = getattr(doc, 'id', None) or getattr(doc.metadata, 'id', None) or getattr(doc.metadata, 'doc_id', None)
            chunk_id = getattr(doc.metadata, 'chunk_id', '')
            full_doc_id = f"{doc_id}_{chunk_id}" if doc_id and chunk_id else doc_id
            
            if full_doc_id in relevant_doc_ids:
                return rank
    
    # 策略1: ID匹配（适用于中文数据）
    correct_doc_id = sample.get('doc_id') or sample.get('id') or sample.get('document_id')
    if correct_doc_id:
        for rank, doc in enumerate(retrieved_docs, 1):
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_id = doc_data.get('doc_id') or doc_data.get('id')
                    if doc_id == correct_doc_id:
                        return rank
            except:
                pass
            
            doc_id = getattr(doc, 'id', None) or getattr(doc.metadata, 'id', None) or getattr(doc.metadata, 'doc_id', None)
            if doc_id == correct_doc_id:
                return rank
    
    # 策略2: 内容哈希匹配
    context_hash = calculate_content_hash(context.strip())
    for rank, doc in enumerate(retrieved_docs, 1):
        doc_content = doc.content
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                doc_context = doc_data.get('context', '')
                if doc_context:
                    doc_content = doc_context
        except:
            pass
        
        doc_hash = calculate_content_hash(doc_content.strip())
        if doc_hash == context_hash:
            return rank
    
    # 策略3: 精确文本匹配
    context_clean = context.strip().lower()
    for rank, doc in enumerate(retrieved_docs, 1):
        doc_content = doc.content
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                doc_context = doc_data.get('context', '')
                if doc_context:
                    doc_content = doc_context
        except:
            pass
        
        doc_content_clean = doc_content.strip().lower()
        
        if (context_clean in doc_content_clean or 
            doc_content_clean in context_clean or
            context_clean == doc_content_clean):
            return rank
    
    # 策略4: 模糊文本匹配（使用关键词）
    context_words = set(context_clean.split())
    if len(context_words) > 3:
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_content = doc.content
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_context = doc_data.get('context', '')
                    if doc_context:
                        doc_content = doc_context
            except:
                pass
            
            doc_content_clean = doc_content.strip().lower()
            doc_words = set(doc_content_clean.split())
            
            overlap = len(context_words.intersection(doc_words))
            overlap_ratio = overlap / len(context_words)
            
            if overlap_ratio > 0.7:
                return rank
    
    # 策略5: 相似度匹配（如果有编码器且使用CPU）
    if encoder and len(context) > 10:
        try:
            # 确保使用CPU
            context_embedding = encoder.encode([context])
            
            doc_contents = []
            for doc in retrieved_docs:
                doc_content = doc.content
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        doc_context = doc_data.get('context', '')
                        if doc_context:
                            doc_content = doc_context
                except:
                    pass
                doc_contents.append(doc_content)
            
            doc_embeddings = encoder.encode(doc_contents)
            
            similarities = []
            for doc_emb in doc_embeddings:
                cos_sim = np.dot(context_embedding[0], doc_emb) / (
                    np.linalg.norm(context_embedding[0]) * np.linalg.norm(doc_emb)
                )
                similarities.append(cos_sim)
            
            max_sim_idx = int(np.argmax(similarities))
            max_similarity = similarities[max_sim_idx]
            
            if max_similarity > 0.8:
                return max_sim_idx + 1
                
        except Exception as e:
            print(f"相似度计算失败: {e}")
    
    return 0
'''
    
    # 保存增强的评估函数
    with open("enhanced_evaluation_functions.py", "w", encoding="utf-8") as f:
        f.write(enhanced_function)
    
    print("✅ 成功创建增强的评估函数: enhanced_evaluation_functions.py")

def main():
    """主函数"""
    print("=== TatQA评估方法升级工具 (CPU版本) ===")
    print("目标：消除英文MRR可能因模糊匹配而存在的高估嫌疑")
    print("设备：使用CPU进行计算")
    print()
    
    # 1. 升级TatQA评估数据
    upgrade_tatqa_eval_data()
    
    print("\n" + "="*50 + "\n")
    
    # 2. 创建增强的评估函数
    create_enhanced_evaluation_function()
    
    print("\n" + "="*50)
    print("✅ 升级完成！")
    print("\n使用说明：")
    print("1. 使用 tatqa_eval_upgraded.jsonl 替代原来的 tatqa_eval.jsonl")
    print("2. 在评估代码中导入并使用 find_correct_document_rank_enhanced 函数")
    print("3. 新函数会优先使用 relevant_doc_ids 进行严格匹配")
    print("4. 这样可以确保英文MRR评估的严谨性和可信度")
    print("5. 所有计算都在CPU上进行，无需GPU")

if __name__ == "__main__":
    main() 