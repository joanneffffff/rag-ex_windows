import argparse
import json
import os
import re
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator
# from transformers import BitsAndBytesConfig # <--- 这行被注释掉了，因为我们不再使用 8-bit 量化
import torch
import gc
import ast
from collections import defaultdict # <--- 确保这一行是存在的

# --- Utility Function for Context Conversion (保持不变) ---
def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    chunks = []
    if not json_str_context or not json_str_context.strip():
        return chunks
    processed_str_context = json_str_context.replace("\\n", "\n")
    cleaned_initial = re.sub(re.escape("【问题】:"), "", processed_str_context)
    cleaned_initial = re.sub(re.escape("【答案】:"), "", cleaned_initial).strip()
    cleaned_initial = cleaned_initial.replace('，', ',')
    cleaned_initial = cleaned_initial.replace('：', ':')
    cleaned_initial = cleaned_initial.replace('【', '') 
    cleaned_initial = cleaned_initial.replace('】', '') 
    cleaned_initial = cleaned_initial.replace('\u3000', ' ')
    cleaned_initial = cleaned_initial.replace('\xa0', ' ').strip()
    cleaned_initial = re.sub(r'\s+', ' ', cleaned_initial).strip()
    
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
        report_content_preview = re.sub(re.escape("【问题】:"), "", report_content_preview)
        report_content_preview = re.sub(re.escape("【答案】:"), "", report_content_preview).strip()
        report_content_preview = re.sub(r'\s+', ' ', report_content_preview).strip() 
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

    extracted_dict_str = None
    parsed_data = None 
    temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_initial) 
    all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 
    for potential_dict_str in all_dict_matches:
        cleaned_potential_dict_str = potential_dict_str.strip()
        json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
        try:
            parsed_data_temp = json.loads(json_compatible_str_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break 
        except json.JSONDecodeError:
            pass 
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
            except TypeError:
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
        return chunks 

    pure_text = cleaned_initial
    pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", pure_text, 1).strip()
    pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()
    pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()
    pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()

    if pure_text: 
        chunks.append(pure_text)
    else:
        print(f"警告：未能在 context 字符串中找到有效结构 (字典、研报或纯文本)。原始字符串（前100字符）：{json_str_context[:100]}...")
        chunks.append(f"原始格式，解析失败或无有效结构：{json_str_context.strip()[:100]}...")
    return chunks

# --- Main Finetuning Script ---

def main():
    parser = argparse.ArgumentParser(description="Finetune Chinese Encoder for RAG with in-training evaluation.")
    parser.add_argument("--base_encoder_model_name", type=str, default="Langboat/mengzi-bert-base-fin",
                        help="Base encoder model to finetune (e.g., 'Langboat/mengzi-bert-base-fin').")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to the training JSONL file (e.g., 'alphafin_train_qc.jsonl').")
    parser.add_argument("--eval_jsonl", type=str, required=True,
                        help="Path to the evaluation JSONL file (e.g., 'alphafin_eval.jsonl').")
    parser.add_argument("--base_raw_data_path", type=str, required=True,
                        help="Path to the original raw data JSON file (e.g., alphafin_rag_ready_generated_cleaned.json), needed for context lookup.")
    parser.add_argument("--output_model_path", type=str, default="./fine-tuned_chinese_encoder",
                        help="Path to save the fine-tuned encoder model.")
    parser.add_argument("--train_batch_size", type=int, default=16,
                        help="Batch size for training. Adjust based on GPU memory.")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for training.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length for tokenization.")
    parser.add_argument("--limit_train_data", type=int, default=0,
                        help="Limit the number of training examples to load. Set 0 for no limit.")
    parser.add_argument("--limit_eval_data", type=int, default=0,
                        help="Limit the number of evaluation examples to load. Set 0 for no limit. Recommended for quick checks.")
    parser.add_argument("--eval_top_k", type=int, default=100,
                        help="Top K for MRR calculation during evaluation.")


    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 移除了 8-bit 量化配置 ===
    # quantization_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )
    
    # 打印信息也修改为 FP16
    print(f"Loading base encoder model: {args.base_encoder_model_name} with float16...") 
    
    model = SentenceTransformer(
        model_name_or_path=args.base_encoder_model_name,
        # model_kwargs={'quantization_config': quantization_config}, # <--- 移除了量化配置
        # 直接指定 torch_dtype 为 float16，确保模型以半精度加载
        model_kwargs={'torch_dtype': torch.float16}, # <--- 添加此行
        device=str(device)
    )
    # 确保模型被推送到 GPU，并且对于 FP16 训练，模型需要是 FP16
    model.to(device)
    
    model.max_seq_length = args.max_seq_length

    print(f"Model loaded. Max sequence length set to: {model.max_seq_length}")
    print(f"Model will be saved to: {args.output_model_path}")

    # --- Load Raw Data for Context Lookup (Corpus) ---
    print(f"Loading and processing raw data for context lookup from: {args.base_raw_data_path}...")
    corpus_documents = {}
    with open(args.base_raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    for idx, item in enumerate(tqdm(raw_data, desc="Building corpus map for training data")):
        doc_id = str(idx)
        company_name = item.get('stock_name', '公司')
        original_json_context_str = item.get('context', '')
        natural_language_chunks = convert_json_context_to_natural_language_chunks(
            original_json_context_str, company_name=company_name
        )
        is_parse_failed_chunk = False
        if not natural_language_chunks:
            is_parse_failed_chunk = True
        elif len(natural_language_chunks) == 1 and (
            natural_language_chunks[0].startswith("原始格式，无有效字典") or 
            natural_language_chunks[0].startswith("原始格式，解析失败") or
            natural_language_chunks[0].startswith("原始格式，无有效结构")
        ):
            is_parse_failed_chunk = True
            
        if not is_parse_failed_chunk:
            corpus_documents[doc_id] = "\n\n".join(natural_language_chunks)
        
    print(f"Corpus map built with {len(corpus_documents)} documents.")

    # --- Prepare Training Data ---
    print(f"Loading training data from: {args.train_jsonl}...")
    train_examples = []
    skipped_train_examples = 0
    with open(args.train_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.limit_train_data > 0 and i >= args.limit_train_data:
                print(f"Limited training data to {args.limit_train_data} examples.")
                break
            data = json.loads(line)
            query_text = data.get('query', '').strip()
            ground_truth_doc_id = data.get('doc_id') 

            if not query_text or ground_truth_doc_id is None:
                skipped_train_examples += 1
                continue
            
            positive_doc_text = corpus_documents.get(str(ground_truth_doc_id))
            
            if positive_doc_text:
                train_examples.append(InputExample(texts=[query_text, positive_doc_text]))
            else:
                skipped_train_examples += 1

    print(f"Loaded {len(train_examples)} training examples. Skipped {skipped_train_examples} due to missing data or context.")

    if not train_examples:
        print("No valid training examples found. Exiting.")
        return

    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size)

    # --- Prepare Evaluation Data for InformationRetrievalEvaluator ---
    print(f"Loading evaluation data from: {args.eval_jsonl} for in-training evaluation...")
    
    # 构建评估器所需的 queries, corpus, relevant_docs
    eval_queries = {} # query_id -> query_text
    eval_corpus = {}  # doc_id -> doc_text
    # 重新从 base_raw_data_path 加载语料库，因为 eval_jsonl 中的 doc_id 对应的是原始数据的索引
    temp_eval_corpus_raw_data = {} 
    with open(args.base_raw_data_path, 'r', encoding='utf-8') as f:
        raw_data_full_eval_corpus = json.load(f)
        for idx, item in enumerate(tqdm(raw_data_full_eval_corpus, desc="Building full corpus map for evaluation")):
            doc_id = str(idx)
            company_name = item.get('stock_name', '公司')
            original_json_context_str = item.get('context', '')
            natural_language_chunks = convert_json_context_to_natural_language_chunks(
                original_json_context_str, company_name=company_name
            )
            is_parse_failed_chunk = False
            if not natural_language_chunks:
                is_parse_failed_chunk = True
            elif len(natural_language_chunks) == 1 and (
                natural_language_chunks[0].startswith("原始格式，无有效字典") or 
                natural_language_chunks[0].startswith("原始格式，解析失败") or
                natural_language_chunks[0].startswith("原始格式，无有效结构")
            ):
                is_parse_failed_chunk = True
            
            if not is_parse_failed_chunk:
                temp_eval_corpus_raw_data[doc_id] = "\n\n".join(natural_language_chunks)

    # 从 eval_jsonl 加载查询和相关文档信息
    relevant_docs = defaultdict(set) # query_id -> set(relevant_doc_ids)
    eval_query_id_counter = 0 # 为评估查询生成一个唯一的ID
    skipped_eval_examples = 0

    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if args.limit_eval_data > 0 and i >= args.limit_eval_data:
                print(f"Limited evaluation data to {args.limit_eval_data} examples.")
                break
            
            data = json.loads(line)
            query_text = data.get('query', '').strip()
            ground_truth_doc_id = data.get('doc_id') 

            if not query_text or ground_truth_doc_id is None:
                skipped_eval_examples += 1
                continue
            
            # 只有当 ground_truth_doc_id 存在于我们构建的评估语料库中时才添加
            if str(ground_truth_doc_id) in temp_eval_corpus_raw_data:
                q_id = f"q_{eval_query_id_counter}"
                eval_queries[q_id] = query_text
                relevant_docs[q_id].add(str(ground_truth_doc_id))
                eval_query_id_counter += 1
                # 将 ground_truth_doc_id 对应的文档添加到 eval_corpus 中
                eval_corpus = temp_eval_corpus_raw_data # 将完整的语料库作为评估语料库
            else:
                skipped_eval_examples += 1
    
    print(f"Loaded {len(eval_queries)} evaluation queries. Skipped {skipped_eval_examples} due to missing data or context in evaluation.")
    print(f"Evaluation corpus contains {len(eval_corpus)} documents.")

    if not eval_queries:
        print("No valid evaluation queries found. Evaluation will be skipped.")
        evaluator = None
    else:
        # InformationRetrievalEvaluator 需要 (queries, corpus, relevant_docs, name, show_progress_bar, **kwargs)
        evaluator = InformationRetrievalEvaluator(
            eval_queries, 
            eval_corpus, 
            relevant_docs, 
            name='finetune-eval',
            show_progress_bar=True,
            mrr_at_k=[args.eval_top_k] # 只保留 mrr_at_k
            # 移除了 precision_at_k=[] 和 ndcg_at_k=[]
        )

    # --- Define Loss Function ---
    train_loss = MultipleNegativesRankingLoss(model=model)

    # --- Fine-tune the model ---
    print(f"Starting fine-tuning for {args.num_epochs} epochs...")
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=args.num_epochs,
            warmup_steps=int(len(train_dataloader) * args.num_epochs * 0.1),
            output_path=args.output_model_path,
            checkpoint_save_steps=len(train_dataloader) // 2 if len(train_dataloader) > 1 else 0,
            checkpoint_save_total_limit=2,
            evaluation_steps=len(train_dataloader) // 4 if len(train_dataloader) > 1 else 0,
            show_progress_bar=True,
            optimizer_params={'lr': args.learning_rate, 'eps': 1e-6}, # 移除了 'correct_bias'
            # fp16=True # <--- 新增此行，启用 FP16 训练
            # save_best_model=True,
            # measure='mrr@k'
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
        del model
        if 'train_dataset' in locals(): del train_dataset
        if 'train_dataloader' in locals(): del train_dataloader
        if 'train_loss' in locals(): del train_loss
        if 'eval_queries' in locals(): del eval_queries
        if 'eval_corpus' in locals(): del eval_corpus
        if 'relevant_docs' in locals(): del relevant_docs
        if 'evaluator' in locals(): del evaluator
        torch.cuda.empty_cache()
        gc.collect()
        print("Attempted to clean up memory after error.")
        return

    print("Fine-tuning completed successfully!")

    print(f"Fine-tuned model saved to: {args.output_model_path}")
    print("\n--- Next Steps ---")
    print(f"You can now use this fine-tuned model '{args.output_model_path}' as your encoder in evaluate_chinese_encoder_reranker_mrr.py.")
    print("Example: python evaluate_chinese_encoder_reranker_mrr.py --encoder_model_name {args.output_model_path} ...")

if __name__ == "__main__":
    main()