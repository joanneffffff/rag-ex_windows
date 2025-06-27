# import json
# import argparse
# import os
# import csv
# import numpy as np
# from sentence_transformers import SentenceTransformer, InputExample, losses, util
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from sentence_transformers.evaluation import SentenceEvaluator

# # --- 数据加载函数 ---
# def load_qc_pairs(jsonl_path, max_samples=None):
#     """加载训练用的 Q-C 对"""
#     pairs = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if max_samples and i >= max_samples:
#                 break
#             try:
#                 item = json.loads(line)
#                 q_raw = item['query'] if 'query' in item else item.get('question', '')
#                 q = str(q_raw).strip() 
                
#                 c_raw = item.get('context', '')
#                 c = str(c_raw).strip()
                
#                 if q and c:
#                     pairs.append(InputExample(texts=[q, c], label=1.0))
#                 else:
#                     print(f"警告：训练数据行 {i+1}，因为 'query'/'question' 或 'context' 为空或无效。原始行: {line.strip()}")
#             except json.JSONDecodeError:
#                 print(f"错误：跳过训练数据行 {i+1}，非法的 JSON 格式: {line.strip()}")
#             except Exception as e:
#                 print(f"处理训练数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
#     return pairs

# def load_qca_for_eval(jsonl_path, max_samples=None):
#     """加载评估用的 Q-C-A 数据，供 MRREvaluator 使用"""
#     data = []
#     with open(jsonl_path, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             if max_samples and i >= max_samples:
#                 break
#             try:
#                 item = json.loads(line)
#                 q_raw = item['query'] if 'query' in item else item.get('question', '')
#                 q = str(q_raw).strip() 
                
#                 c_raw = item.get('context', '')
#                 c = str(c_raw).strip()
                
#                 a_raw = item.get('answer', '')
#                 a = str(a_raw).strip() 

#                 if q and c: # 确保问题和上下文都非空
#                     data.append({'query': q, 'context': c, 'answer': a})
#                 else:
#                     print(f"警告：评估数据行 {i+1}，因为 'query'/'question' 或 'context' 为空或无效。原始行: {line.strip()}")
#             except json.JSONDecodeError:
#                 print(f"错误：跳过评估数据行 {i+1}，非法的 JSON 格式: {line.strip()}")
#             except Exception as e:
#                 print(f"处理评估数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
#     return data

# # --- MRR 评估器类 ---
# class MRREvaluator(SentenceEvaluator):
#     """
#     对给定的 (query, context, answer) 数据集计算 Mean Reciprocal Rank (MRR)。
#     假设每个 item['context'] 是其 item['query'] 的正确上下文。
#     """
#     def __init__(self, dataset, name='', show_progress_bar=False, write_csv=True):
#         self.dataset = dataset
#         self.name = name
#         self.show_progress_bar = show_progress_bar
#         self.write_csv = write_csv

#         self.queries = [item['query'] for item in dataset]
#         self.contexts = [item['context'] for item in dataset]
#         self.answers = [item['answer'] for item in dataset] 

#         self.csv_file: str = None
#         self.csv_headers = ["epoch", "steps", "MRR"]


#     def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
#         if epoch != -1:
#             if self.write_csv:
#                 self.csv_file = os.path.join(output_path, self.name + "_mrr_evaluation_results.csv")
#                 if not os.path.isfile(self.csv_file) or epoch == 0:
#                     with open(self.csv_file, newline="", mode="w", encoding="utf-8") as f:
#                         writer = csv.writer(f)
#                         writer.writerow(self.csv_headers)
                        
#         print(f"\n--- 开始 MRR 评估 (Epoch: {epoch}, Steps: {steps}) ---")

#         if not self.dataset:
#             print("警告：评估数据集为空，MRR为0。")
#             mrr = 0.0
#         else:
#             print(f"编码 {len(self.contexts)} 个评估上下文...")
#             # >>>>> 这里是修正过的行 <<<<<
#             context_embeddings = model.encode(self.contexts, batch_size=64, convert_to_tensor=True,
#                                               show_progress_bar=self.show_progress_bar)

#             mrrs = []
#             iterator = tqdm(self.dataset, desc='评估 MRR', disable=not self.show_progress_bar)
#             for i, item in enumerate(iterator):
#                 query_emb = model.encode(item['query'], convert_to_tensor=True)
#                 scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()

#                 target_context_idx = i 

#                 sorted_indices = np.argsort(scores)[::-1]
                
#                 rank = -1
#                 for r, idx in enumerate(sorted_indices):
#                     if idx == target_context_idx:
#                         rank = r + 1
#                         break
                
#                 if rank != -1:
#                     mrr_score = 1.0 / rank
#                     mrrs.append(mrr_score)
#                 else:
#                     mrrs.append(0.0) 
            
#             mrr = np.mean(mrrs) if mrrs else 0.0 

#         print(f"MRR (Epoch: {epoch}, Steps: {steps}): {mrr:.4f}")
#         print(f"--- MRR 评估结束 ---")

#         if output_path is not None and self.write_csv:
#             with open(self.csv_file, newline="", mode="a", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([epoch, steps, round(mrr, 4)])

#         return mrr

# # --- 主微调函数 ---
# def finetune_encoder(model_name, train_jsonl, eval_jsonl, output_dir, batch_size=32, epochs=1, max_samples=None, eval_steps=500):
#     print(f"正在加载训练数据：{train_jsonl}")
#     train_examples = load_qc_pairs(train_jsonl, max_samples)
#     if not train_examples:
#         print("警告：没有加载到有效的训练样本，请检查数据文件和键名。")
#         return

#     print(f"正在加载评估数据：{eval_jsonl}")
#     eval_data = load_qca_for_eval(eval_jsonl, max_samples) 
#     if not eval_data:
#         print("警告：没有加载到有效的评估样本，无法进行MRR评估。")
#         evaluator = None
#     else:
#         print(f"加载了 {len(eval_data)} 个评估样本。")
#         evaluator = MRREvaluator(dataset=eval_data, name='mrr_eval', show_progress_bar=True)


#     print(f"加载模型：{model_name}")
#     try:
#         model = SentenceTransformer(model_name)
#     except Exception as e:
#         print(f"错误：无法加载 SentenceTransformer 模型 '{model_name}'。请确保模型路径正确或模型ID有效。")
#         print(f"详细错误信息: {e}")
#         return

#     print(f"训练样本数量：{len(train_examples)}")
#     train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
#     train_loss = losses.MultipleNegativesRankingLoss(model) 
    
#     print(f"开始模型微调，共 {epochs} 个 epoch...")
#     model.fit(
#         train_objectives=[(train_dataloader, train_loss)], 
#         epochs=epochs,
#         warmup_steps=100,
#         evaluator=evaluator,          
#         evaluation_steps=eval_steps,  
#         output_path=output_dir,       
#         show_progress_bar='batches'   
#     )
    
#     os.makedirs(output_dir, exist_ok=True)
#     model.save(output_dir)
#     print(f"模型已保存到：{output_dir}")

# # --- 主执行块 ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, required=True, help='预训练模型名或路径')
#     parser.add_argument('--train_jsonl', type=str, required=True, help='训练数据jsonl文件路径')
#     parser.add_argument('--eval_jsonl', type=str, required=True, help='评估数据jsonl文件路径') 
#     parser.add_argument('--output_dir', type=str, required=True, help='微调模型输出目录')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--epochs', type=int, default=1)
#     parser.add_argument('--max_samples', type=int, default=None, help='训练和评估的最大样本数')
#     parser.add_argument('--eval_steps', type=int, default=0, help='每隔多少训练步评估一次 (0表示只在每个epoch结束时评估)') 
    
#     args = parser.parse_args()

#     # 执行模型微调
#     finetune_encoder(
#         model_name=args.model_name,
#         train_jsonl=args.train_jsonl,
#         eval_jsonl=args.eval_jsonl,
#         output_dir=args.output_dir,
#         batch_size=args.batch_size,
#         epochs=args.epochs,
#         max_samples=args.max_samples,
#         eval_steps=args.eval_steps
#     )


import json
import argparse
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, losses, util
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import re

# 导入 InformationRetrievalEvaluator
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# ==================== 兼容性函数 ====================
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

# ==================== 开始：自定义 Chunking 和 Table to Text 逻辑 ====================
# 这部分代码与您之前提供的 convert_tatqa_to_qca.py 中的逻辑一致
# 确保与 convert_tatqa_to_qca.py 中的逻辑保持同步，以保证数据生成的一致性

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
            if h_idx == 0: # Skip the first element as it's the row name
                continue
            
            header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}" # Fallback for headers
            value = str(v).strip()

            if value: # Only process non-empty values
                # Attempt to add units and currency symbols
                # Improved regex to robustly match numbers (including negative, comma-separated, parenthetical negatives)
                if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value): 
                    formatted_value = value.replace('$', '') # Remove $ for consistent re-adding
                    if unit_info:
                        # Handle parenthetical negative numbers before adding unit
                        if formatted_value.startswith('(') and formatted_value.endswith(')'):
                             formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                        else:
                             formatted_value = f"${formatted_value} {unit_info}"
                    else:
                        formatted_value = f"${formatted_value}" # If no unit info, keep currency symbol
                else:
                    formatted_value = value
                
                data_descriptions.append(f"{header} is {formatted_value}") # More natural phrasing

        # Combine row descriptions
        if row_name and data_descriptions:
            lines.append(f"Details for item {row_name}: {'; '.join(data_descriptions)}.")
        elif data_descriptions: # If row name is empty but there's data
            lines.append(f"Other data item: {'; '.join(data_descriptions)}.")
        elif row_name: # Only row name, no data (should be caught by empty row check usually)
            lines.append(f"Data item: {row_name}.")

    return "\n".join(lines)


def process_tatqa_to_qca_for_corpus(input_paths):
    """
    Processes TatQA dataset(s) for corpus building, integrating paragraph and table content into chunks.
    This version focuses on creating the full context/chunks for retrieval.
    """
    all_chunks = []
    global_doc_counter = 0 # 用于生成唯一的 doc_id

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 确保 data 是一个列表，即使文件只包含一个文档
        if not isinstance(data, list):
            print(f"警告：文件 {Path(input_path).name} 的顶层结构不是列表，尝试作为单个文档处理。")
            data = [data] # 将单个字典封装成列表

        for i, item in tqdm(enumerate(data), desc=f"Processing docs from {Path(input_path).name} for corpus"):
            # 确保 item 是一个字典
            if not isinstance(item, dict):
                print(f"警告：文件 {Path(input_path).name} 中发现非字典项，跳过。项内容：{item}")
                continue
            
            doc_id = item.get("doc_id")
            if doc_id is None:
                # 如果 doc_id 不存在，生成一个唯一的 ID
                doc_id = f"generated_doc_{global_doc_counter}_{Path(input_path).stem}_{i}"
                global_doc_counter += 1
                # print(f"警告：文件 {Path(input_path).name} 中发现缺少 'doc_id' 的文档，已生成 ID: {doc_id}") # 避免过多打印

            paragraphs = item.get("paragraphs", [])
            tables = item.get("tables", [])

            # Extract unit information, typically found in a descriptive paragraph
            unit_info = extract_unit_from_paragraph(paragraphs)

            # Process paragraphs as chunks
            for p_idx, para in enumerate(paragraphs):
                para_text = para.get("text", "") if isinstance(para, dict) else para
                if para_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id, # 使用确定的 doc_id
                        "chunk_id": f"para_{p_idx}",
                        "text": para_text.strip(),
                        "source_type": "paragraph"
                    })
            
            # Process tables as chunks
            for t_idx, table in enumerate(tables):
                table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                if table_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id, # 使用确定的 doc_id
                        "chunk_id": f"table_{t_idx}",
                        "text": table_text.strip(),
                        "source_type": "table"
                    })
    return all_chunks
# ==================== 结束：自定义 Chunking 和 Table to Text 逻辑 ====================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Path to the base model (e.g., 'ProsusAI/finbert').")
    parser.add_argument("--train_jsonl", type=str, required=True, help="Path to the training JSONL file.")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="Path to the evaluation JSONL file.")
    parser.add_argument("--output_dir", type=str, default="models/finetuned_model", help="Directory to save the finetuned model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of training samples to use (for debugging).")
    parser.add_argument("--eval_steps", type=int, default=0, help="Evaluate model every N steps. 0 for no in-training evaluation.")
    parser.add_argument("--base_raw_data_path", type=str, default="data/tatqa_dataset_raw/", help="Base path to raw TatQA dataset files for corpus building.")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # ==================== 最终模型加载方式 ====================
    print(f"加载 Encoder 模型: {args.model_name}")
    try:
        # SentenceTransformer 可以直接从本地路径加载模型
        model = SentenceTransformer(args.model_name)
        print("SentenceTransformer 模型加载成功。")
        model.to(device) # 确保SentenceTransformer模型也移到指定设备
    except Exception as e:
        print(f"加载模型失败。错误信息: {e}")
        import traceback
        traceback.print_exc()
        return # 如果模型加载失败，直接退出
    
    # ==================== 增强训练数据加载的健壮性 (兼容 'question' 和 'query') ====================
    print(f"加载训练数据：{args.train_jsonl}")
    raw_train_data = []
    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            raw_train_data.append(json.loads(line))
    
    train_data = []
    for item in raw_train_data:
        # 修正：检查 'context' 键，以及 'question' 或 'query' 至少一个存在
        is_valid_item = isinstance(item, dict) and "context" in item and \
                        (get_question_or_query(item) is not None)

        if is_valid_item:
            train_data.append(item)
        else:
            print(f"警告：训练数据中发现无效样本，缺少 'context' 或缺少 'question'/'query' 键，或不是字典。样本内容：{item}")
    
    if args.max_samples is not None:
        train_data = train_data[:args.max_samples]
    print(f"加载了 {len(train_data)} 个有效训练样本。 (共 {len(raw_train_data)} 个原始样本)")

    # 检查是否有足够的有效样本进行训练
    if not train_data:
        print("错误：没有找到任何有效的训练样本，无法进行训练。请检查您的 JSONL 文件格式或其内容。")
        return

    # Prepare training examples
    train_examples = []
    # 修正：InputExample 的 texts 属性使用兼容函数
    for item in train_data:
        question_text = get_question_or_query(item)
        # 这里理论上 question_text 不会是 None，因为前面已经过滤了
        train_examples.append(InputExample(texts=[question_text, item["context"]]))
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    # 关键修改：使用 MultipleNegativesRankingLoss
    train_loss = losses.MultipleNegativesRankingLoss(model=model) 

    evaluator = None
    if args.eval_steps > 0:
        print(f"加载评估数据：{args.eval_jsonl}")
        raw_eval_data_for_eval = []
        with open(args.eval_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                raw_eval_data_for_eval.append(json.loads(line))
        
        eval_data = []
        for item in raw_eval_data_for_eval:
            # 修正：检查 'context' 键，以及 'question' 或 'query' 至少一个存在
            is_valid_item = isinstance(item, dict) and "context" in item and \
                            (get_question_or_query(item) is not None)
            if is_valid_item:
                eval_data.append(item)
            else:
                print(f"警告：评估数据（训练时）中发现无效样本，缺少 'context' 或缺少 'question'/'query' 键，或不是字典。样本内容：{item}")
        print(f"加载了 {len(eval_data)} 个有效评估样本 (训练时评估)。 (共 {len(raw_eval_data_for_eval)} 个原始样本)")

        if not eval_data:
            print("警告：没有找到任何有效的评估样本用于训练时的评估。跳过训练过程中的评估。")
        else:
            # 构建完整的 Chunk 检索库 (包含所有段落和表格)
            print(f"从原始数据路径构建完整 Chunk 检索库：{Path(args.base_raw_data_path)}")
            corpus_input_paths = [
                Path(args.base_raw_data_path) / "tatqa_dataset_train.json",
                Path(args.base_raw_data_path) / "tatqa_dataset_dev.json",
                Path(args.base_raw_data_path) / "tatqa_dataset_test_gold.json"
            ]
            corpus_chunks = process_tatqa_to_qca_for_corpus(corpus_input_paths)
            print(f"构建了 {len(corpus_chunks)} 个 Chunk 作为检索库。")

            # 准备 InformationRetrievalEvaluator 所需的数据
            queries = {}
            corpus = {}
            relevant_docs = {} # qrels: {query_id: {doc_id, doc_id, ...}}

            # 填充 queries 和 corpus
            for i, query_item in enumerate(eval_data):
                query_id = f"q_{i}" # 生成唯一的查询 ID
                queries[query_id] = get_question_or_query(query_item)
                
                # 寻找正确的 chunk_id
                correct_context = query_item["context"]
                found_chunk_id = None
                # 由于 corpus_chunks 是列表，这里需要线性搜索，效率不高，但对于评估集大小通常可接受
                # 更优化的做法是先将 corpus_chunks 转换为 {text: chunk_id} 的字典
                for chunk in corpus_chunks:
                    if chunk["text"] == correct_context:
                        # 确保 chunk_id 在全局语料库中是唯一的
                        found_chunk_id = f"doc_{chunk['doc_id']}_{chunk['chunk_id']}"
                        break
                
                if found_chunk_id:
                    if query_id not in relevant_docs:
                        relevant_docs[query_id] = set()
                    relevant_docs[query_id].add(found_chunk_id)
                else:
                    print(f"警告：找不到查询 '{queries[query_id]}' 对应的正确上下文：'{correct_context}' 在语料库中。此查询将被排除在评估之外。")
            
            # 填充 corpus 字典，使用统一的 Chunk ID
            for chunk in corpus_chunks:
                corpus_id = f"doc_{chunk['doc_id']}_{chunk['chunk_id']}"
                corpus[corpus_id] = chunk["text"]

            # 创建 InformationRetrievalEvaluator 实例
            # 这里的 batch_size 是指评估时的查询批次大小，可以设大一些
            evaluator = InformationRetrievalEvaluator(
                queries=queries, 
                corpus=corpus, 
                relevant_docs=relevant_docs, 
                batch_size=16, # 可以根据需要调整评估时的批次大小
                main_score_function=None, # 默认是CosineSimilarity, 这里可以不指定
                name='mrr_eval' # 评估结果的文件名后缀
            )

    # Start training
    print("开始训练...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=100, # A small number of warmup steps
              output_path=args.output_dir,
              evaluator=evaluator, # 将 evaluator 实例传递给 fit 方法
              evaluation_steps=args.eval_steps,
              callback=None # 不再需要自定义回调函数
             )

    print(f"微调模型已保存到：{args.output_dir}")

if __name__ == "__main__":
    main()