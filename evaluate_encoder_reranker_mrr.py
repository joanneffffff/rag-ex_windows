import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
# 导入 CrossEncoder 类
from sentence_transformers import CrossEncoder 
import torch
import numpy as np
import os
import re

# ==================== 兼容性函数 (与之前相同) ====================
def get_question_or_query(item_dict):
    """
    尝试从字典中获取 'question' 或 'query' 的值。
    优先返回 'question' 的值，如果不存在则返回 'query' 的值。
    如果两者都不存在，则返回 None。
    """
    if "question" in item_dict:
        return item_dict["question"]
    elif "query" in item_dict:
        return item_dict["query"]
    return None

# ==================== 自定义 Chunking 和 Table to Text 逻辑 (与之前相同) ====================
def extract_unit_from_paragraph(paragraphs):
    """
    Attempts to extract numerical units from paragraphs, such as "in millions" or "in millions except per share amounts".
    """
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        # Look for patterns like "dollars in millions" or "in millions"
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            # Extract the matched unit word and standardize it
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD" # Assuming USD for financial data
    return ""

def table_to_natural_text(table_dict, caption="", unit_info=""):
    """
    Converts a TatQA table into more conversational/descriptive natural language, suitable for LLMs.
    This function assumes the first sublist in table_dict['table'] is the header, and the rest are data rows.
    It attempts to organize more natural sentences, integrate unit information, and handle empty values and category header rows.
    """
    rows = table_dict.get("table", [])
    lines = []

    if caption:
        lines.append(f"Table Topic: {caption}.") # Added a period for sentence completion

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]

    for i, row in enumerate(data_rows):
        # Skip completely empty rows
        if not row or all(str(v).strip() == "" for v in row):
            continue

        # Identify and handle category header rows (e.g., "Current assets" where subsequent cells are empty)
        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Table Category: {str(row[0]).strip()}.")
            continue

        # Process core data rows
        row_name = str(row[0]).strip().replace('.', '') # Clean up row name from trailing periods

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

            for p_idx, para in enumerate(paragraphs):
                para_text = para.get("text", "") if isinstance(para, dict) else para
                if para_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"para_{p_idx}",
                        "text": para_text.strip(),
                        "source_type": "paragraph"
                    })
            
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

# ==================== 结束：自定义 Chunking 和 Table to Text 逻辑 ====================


def compute_mrr_with_reranker(encoder_model, reranker_model, eval_data, corpus_chunks, top_k_retrieval=100, top_k_rerank=10):
    """
    计算使用 Encoder 进行初步检索和 Reranker 进行重排序后的 Mean Reciprocal Rank (MRR)。
    """
    query_texts = [get_question_or_query(item) for item in eval_data]
    correct_chunk_contents = [item["context"] for item in eval_data]

    print("编码评估查询 (Queries) with Encoder...")
    query_embeddings = encoder_model.encode(query_texts, show_progress_bar=True, convert_to_tensor=True)
    
    print("编码检索库上下文 (Chunks) with Encoder...")
    corpus_texts = [chunk["text"] for chunk in corpus_chunks]
    corpus_embeddings = encoder_model.encode(corpus_texts, show_progress_bar=True, convert_to_tensor=True)

    mrr_scores = []
    
    print(f"Evaluating MRR with Reranker (Retrieval Top-{top_k_retrieval}, Rerank Top-{top_k_rerank}):")
    for i, query_emb in tqdm(enumerate(query_embeddings), total=len(query_embeddings)):
        current_query_text = query_texts[i]
        correct_chunk_content = correct_chunk_contents[i]

        # 1. 初步检索 (Encoder)
        cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        # 获取 top_k_retrieval 个最相似的 chunk 的索引
        top_retrieved_indices = torch.topk(cos_scores, k=min(top_k_retrieval, len(corpus_chunks)))[1].tolist()
        
        # 准备 Reranker 的输入
        reranker_input_pairs = []
        retrieved_chunks = []
        for idx in top_retrieved_indices:
            chunk = corpus_chunks[idx]
            reranker_input_pairs.append([current_query_text, chunk["text"]])
            retrieved_chunks.append(chunk)

        if not reranker_input_pairs: # Handle case where no chunks were retrieved
            mrr_scores.append(0)
            continue

        # 2. 重排序 (Reranker)
        # 交叉编码器会为每个 (query, document) 对生成一个分数
        # predict 方法会自动处理批处理
        reranker_scores = reranker_model.predict(reranker_input_pairs, show_progress_bar=False, convert_to_tensor=True) # convert_to_tensor=True for sorting
        
        # 将 reranker scores 与对应的 chunk 关联起来
        # Note: reranker_scores are typically logits or similarity scores, higher means more relevant
        scored_retrieved_chunks_with_scores = []
        for j, score in enumerate(reranker_scores):
            scored_retrieved_chunks_with_scores.append({"chunk": retrieved_chunks[j], "score": score.item()}) # .item() to get scalar

        # 按 Reranker 分数降序排序
        scored_retrieved_chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)

        # 3. 计算 MRR (在重排序后的 Top-K 中查找)
        found_rank = -1
        # 只考虑 reranked top_k_rerank 结果
        for rank, item in enumerate(scored_retrieved_chunks_with_scores[:top_k_rerank]):
            if item["chunk"]["text"] == correct_chunk_content:
                found_rank = rank + 1
                break
        
        if found_rank != -1:
            mrr_scores.append(1 / found_rank)
        else:
            mrr_scores.append(0) # Not found in reranked top_k

    return np.mean(mrr_scores)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_name", type=str, required=True, help="Path to the fine-tuned Encoder model.")
    parser.add_argument("--reranker_model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="Path to the Reranker model or Hugging Face ID.")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="Path to the evaluation JSONL file.")
    parser.add_argument("--base_raw_data_path", type=str, default="data/tatqa_dataset_raw/", help="Base path to raw TatQA dataset files for corpus building.")
    parser.add_argument("--top_k_retrieval", type=int, default=100, help="Number of top documents to retrieve from Encoder before reranking.")
    parser.add_argument("--top_k_rerank", type=int, default=10, help="Number of top documents to consider after reranking for MRR calculation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ==================== 加载 Encoder 模型 ====================
    print(f"加载 Encoder 模型: {args.encoder_model_name}")
    try:
        encoder_model = SentenceTransformer(args.encoder_model_name) 
        print("Encoder 模型加载成功。")
        encoder_model.to(device)
    except Exception as e:
        print(f"加载 Encoder 模型失败。错误信息: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==================== 加载 Reranker 模型 ====================
    # 导入 CrossEncoder 类
    # from sentence_transformers import CrossEncoder # 已经放在文件顶部，这里可以省略，但放这里也无害
    
    print(f"加载 Reranker 模型: {args.reranker_model_name}")
    try:
        # 使用 CrossEncoder 类加载交叉编码器模型
        reranker_model = CrossEncoder(args.reranker_model_name) 
        print("Reranker 模型加载成功。")
        # CrossEncoder 内部会处理 device，通常不需要手动调用 .to(device)
        # reranker_model.to(device) 
    except Exception as e:
        print(f"加载 Reranker 模型失败。错误信息: {e}")
        import traceback
        traceback.print_exc()
        return


    # ==================== 增强评估数据加载的健壮性 (兼容 'question' 和 'query') ====================
    print(f"加载评估数据：{args.eval_jsonl}")
    raw_eval_data = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            raw_eval_data.append(json.loads(line))
    
    eval_data = []
    for item in raw_eval_data:
        is_valid_item = isinstance(item, dict) and "context" in item and \
                        (get_question_or_query(item) is not None)
        
        if is_valid_item:
            eval_data.append(item) 
        else:
            print(f"警告：评估数据中发现无效样本，缺少 'context' 或缺少 'question'/'query' 键，或不是字典。样本内容：{item}")
    print(f"加载了 {len(eval_data)} 个有效评估样本。 (共 {len(raw_eval_data)} 个原始样本)")

    if not eval_data:
        print("错误：没有找到任何有效的评估样本，无法计算 MRR。请检查您的 JSONL 文件格式或其内容。")
        return

    # 构建完整的 Chunk 检索库 (包含所有段落和表格)
    print(f"从原始数据路径构建完整 Chunk 检索库：{Path(args.base_raw_data_path)}")
    corpus_input_paths = [
        Path(args.base_raw_data_path) / "tatqa_dataset_train.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_dev.json",
        Path(args.base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    corpus_chunks = process_tatqa_to_qca_for_corpus(corpus_input_paths)
    print(f"构建了 {len(corpus_chunks)} 个 Chunk 作为检索库。")

    # 计算 MRR (带 Reranker)
    mrr = compute_mrr_with_reranker(encoder_model, reranker_model, eval_data, corpus_chunks, args.top_k_retrieval, args.top_k_rerank)
    print(f"\nMean Reciprocal Rank (MRR) with Reranker: {mrr:.4f}")

if __name__ == "__main__":
    main()
