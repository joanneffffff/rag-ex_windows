import argparse
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig 
import torch
import torch.nn.functional as F
from collections import defaultdict
import ast 

# --- Utility Function for Context Conversion (保持不变) ---
def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    """
    Parses a JSON string context from AlphaFin and converts it into a list of
    natural language chunks, handling various formats and cleaning.
    """
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


# --- Main Evaluation Script Logic ---

def calculate_mrr(rankings):
    """
    Calculates Mean Reciprocal Rank (MRR).
    Args:
        rankings: A list of ranks for relevant documents. E.g., [1, 3, 0] means
                  first relevant doc was at rank 1, second at rank 3, third not found.
                  0 indicates not found.
    Returns:
        The MRR score.
    """
    reciprocal_ranks = []
    num_queries = 0
    for rank in rankings:
        if rank > 0:  # Rank 0 means not found
            reciprocal_ranks.append(1.0 / rank)
            num_queries += 1
    
    if num_queries == 0:
        return 0.0 
    return sum(reciprocal_ranks) / num_queries

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Chinese Encoder and Reranker MRR.")
    parser.add_argument("--encoder_model_name", type=str, required=True,
                        help="Path or name of the encoder model (e.g., 'fine-tuned_model_path').")
    parser.add_argument("--reranker_model_name", type=str, required=True,
                        help="Path or name of the reranker model (e.g., 'Qwen/Qwen3-Reranker-0.6B').")
    parser.add_argument("--eval_jsonl", type=str, required=True,
                        help="Path to the evaluation JSONL file (output from split_alphafin_data.py).")
    parser.add_argument("--base_raw_data_path", type=str, required=True,
                        help="Path to the original raw data JSON file (e.g., alphafin_rag_ready_generated_cleaned.json).")
    parser.add_argument("--top_k_retrieval", type=int, default=100,
                        help="Top K documents to retrieve from the corpus.")
    parser.add_argument("--top_k_rerank", type=int, default=10,
                        help="Top K documents after reranking for MRR calculation.")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === 配置量化参数 (8-bit 或 4-bit) ===
    # 根据您的显存情况选择。4-bit 量化节省更多，但可能对精度影响更大。
    # 推荐先尝试 8-bit 量化 (load_in_8bit=True)
    # 如果 8-bit 仍不够，再尝试 4-bit 量化 (load_in_4bit=True)
    # 注意：不能同时设置 load_in_8bit 和 load_in_4bit
    
    # 量化配置 (用于 AutoModel，如 Encoder)
    quantization_config_encoder = BitsAndBytesConfig(
        load_in_8bit=True, # 尝试 8-bit 量化
        # 或者 load_in_4bit=True, # 如果 8-bit 不够，尝试 4-bit
        # bnb_4bit_quant_type="nf4", # 仅当 load_in_4bit=True 时生效
        # bnb_4bit_compute_dtype=torch.float16 # 仅当 load_in_4bit=True 时生效
    )

    # 量化配置 (用于 AutoModelForCausalLM，如 Qwen Reranker)
    quantization_config_reranker = BitsAndBytesConfig(
        load_in_8bit=True, # 尝试 8-bit 量化
        # 或者 load_in_4bit=True, # 如果 8-bit 不够，尝试 4-bit
        # bnb_4bit_quant_type="nf4", 
        # bnb_4bit_compute_dtype=torch.float16 
    )


    # --- Load Encoder Model and Tokenizer ---
    print(f"Loading encoder model: {args.encoder_model_name} with quantization...")
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    encoder_model = AutoModel.from_pretrained(
        args.encoder_model_name,
        quantization_config=quantization_config_encoder, 
        torch_dtype=torch.float16 
    ) 
    encoder_model.eval()

    encoder_max_length = encoder_tokenizer.model_max_length
    if encoder_max_length > 1024: 
        print(f"Warning: Encoder model_max_length is {encoder_max_length}. Capping to 512 for typical BERT models.")
        encoder_max_length = 512 

    # --- Load Reranker Model and Tokenizer (Using AutoModelForCausalLM as per Qwen's example) ---
    print(f"Loading reranker model: {args.reranker_model_name} with quantization...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name, padding_side='left') 
    reranker_model = AutoModelForCausalLM.from_pretrained(
        args.reranker_model_name,
        quantization_config=quantization_config_reranker, 
        torch_dtype=torch.float16 
    ) 
    reranker_model.eval()

    token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    reranker_max_length = 8192 

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    # 我们不需要单独的 prefix_tokens 和 suffix_tokens 列表了，因为我们会一次性处理字符串

    def format_instruction(instruction, query, doc):
        fixed_instruction = 'Given a web search query, retrieve relevant passages that answer the query' 
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=fixed_instruction, query=query, doc=doc)
        return output

    # === 修改后的 process_reranker_inputs 函数 ===
    def process_reranker_inputs(pairs):
        # 这里的 pairs 是一个列表，每个元素是格式化后的 (instruct, query, doc) 字符串
        # 例如: ["<Instruct>: ... <Query>: ... <Document>: ...", ...]

        # 将前缀和后缀直接加到每个输入字符串上
        full_inputs = [prefix + p + suffix for p in pairs]

        inputs = reranker_tokenizer(
            full_inputs, 
            padding='max_length', # 明确指定填充到 max_length
            truncation=True,      # 启用截断
            max_length=reranker_max_length, 
            return_tensors='pt'
        )
        for key in inputs:
            inputs[key] = inputs[key].to(reranker_model.device)
        return inputs

    @torch.no_grad()
    def compute_reranker_logits(inputs):
        batch_scores = reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist() 
        return scores


    # --- Load and Process Corpus (base_raw_data_path) ---
    print(f"Loading and processing corpus from: {args.base_raw_data_path} (mimicking split_alphafin_data.py logic)...")
    corpus_documents = {}
    
    with open(args.base_raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    processed_corpus_count = 0
    skipped_corpus_items = 0

    for idx, item in enumerate(tqdm(raw_data, desc="Building corpus map with aligned IDs")):
        doc_id = str(idx) 

        q_raw = item.get('question', '')
        q = q_raw.replace("\\n", "\n") 
        q = re.sub(re.escape("【问题】:"), "", q).strip() 
        q = re.sub(r'\s+', ' ', q).strip() 

        a_raw = item.get('answer', '')
        a = a_raw.replace("\\n", "\n") 
        a = re.sub(re.escape("【答案】:"), "", a).strip() 
        a = re.sub(r'\s+', ' ', a).strip() 

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
            
        c = "\n\n".join(natural_language_chunks) 
        
        if q and a and not is_parse_failed_chunk: 
            corpus_documents[doc_id] = c 
            processed_corpus_count += 1
        else:
            skipped_corpus_items += 1
            
    print(f"Corpus built with {len(corpus_documents)} documents (after filtering).")
    print(f"Skipped {skipped_corpus_items} items from raw data while building corpus.")
    
    # Generate embeddings for all corpus documents
    corpus_ids = list(corpus_documents.keys())
    corpus_texts = [corpus_documents[doc_id] for doc_id in corpus_ids]
    corpus_embeddings = []

    print("Generating embeddings for corpus documents...")
    batch_size = 4
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding corpus"):
        batch_texts = corpus_texts[i:i + batch_size] 
        with torch.no_grad():
            # === 修改 Encoder 分词器调用方式 ===
            encoded_input = encoder_tokenizer(
                batch_texts, 
                padding='max_length', 
                truncation=True,      
                max_length=encoder_max_length, 
                return_tensors='pt'
            ).to(device)
            model_output = encoder_model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1) 
            corpus_embeddings.append(embeddings.cpu()) 
    
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0).to(device) 
    print(f"Generated {corpus_embeddings.shape[0]} corpus embeddings.")

    # --- Load Evaluation Data ---
    print(f"Loading evaluation data from: {args.eval_jsonl}...")
    eval_data = []
    LIMIT_EVAL_DATA = 100 
    
    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= LIMIT_EVAL_DATA: 
                break
            eval_data.append(json.loads(line))
    print(f"Loaded {len(eval_data)} evaluation queries (limited to {LIMIT_EVAL_DATA} for testing).")

    all_retrieval_ranks = []
    all_rerank_ranks = []
    skipped_queries_count = 0 

    print("Starting evaluation...")
    for item in tqdm(eval_data, desc="Evaluating queries"):
        query_text = item.get('query', '').strip()
        ground_truth_doc_id = item.get('doc_id') 

        if not query_text:
            skipped_queries_count += 1
            continue
        if ground_truth_doc_id is None:
            skipped_queries_count += 1
            print(f"Warning: Missing 'doc_id' for query: '{query_text[:50]}...'. Skipping.")
            continue
        if ground_truth_doc_id not in corpus_documents:
            skipped_queries_count += 1
            continue
        
        # --- 1. Retrieval (Encoder) ---
        with torch.no_grad():
            # === 修改 Encoder 分词器调用方式 ===
            query_encoded = encoder_tokenizer(
                query_text, 
                padding='max_length', 
                truncation=True, 
                max_length=encoder_max_length, 
                return_tensors='pt'
            ).to(device)
            model_output = encoder_model(**query_encoded)
            embeddings = mean_pooling(model_output, query_encoded['attention_mask'])
            query_embedding = F.normalize(embeddings, p=2, dim=1)

            similarities = torch.matmul(query_embedding, corpus_embeddings.transpose(0, 1))
            
            top_k_retrieval_values, top_k_retrieval_indices = torch.topk(similarities, min(args.top_k_retrieval, len(corpus_documents)), dim=1)
            
            retrieved_doc_ids_and_scores = []
            for i, idx in enumerate(top_k_retrieval_indices[0]):
                doc_id = corpus_ids[idx.item()]
                score = top_k_retrieval_values[0][i].item()
                retrieved_doc_ids_and_scores.append((doc_id, score))

        retrieval_rank = 0
        for rank, (doc_id, _) in enumerate(retrieved_doc_ids_and_scores, 1):
            if doc_id == ground_truth_doc_id:
                retrieval_rank = rank
                break
        all_retrieval_ranks.append(retrieval_rank)


        # --- 2. Reranking (Reranker Model) ---
        rerank_data_for_qwen = [] 
        for doc_id, _ in retrieved_doc_ids_and_scores:
            doc_text = corpus_documents.get(doc_id, "") 
            if doc_text:
                formatted_text = format_instruction(None, query_text, doc_text)
                # 这里只将格式化后的字符串添加到列表中，不再预先添加 prefix 和 suffix
                rerank_data_for_qwen.append((formatted_text, doc_id)) 

        if not rerank_data_for_qwen:
            all_rerank_ranks.append(0)
            continue
        
        reranked_results = []
        reranker_batch_size = 4 
        for j in range(0, len(rerank_data_for_qwen), reranker_batch_size):
            # current_batch_pairs 此时只包含格式化后的 (instruct, query, doc) 字符串
            current_batch_pairs_content = [item[0] for item in rerank_data_for_qwen[j:j + reranker_batch_size]]
            current_batch_doc_ids = [item[1] for item in rerank_data_for_qwen[j:j + reranker_batch_size]]

            # 直接调用修改后的 process_reranker_inputs
            rerank_inputs_batch = process_reranker_inputs(current_batch_pairs_content)
            current_batch_scores = compute_reranker_logits(rerank_inputs_batch)

            for k, score in enumerate(current_batch_scores):
                reranked_results.append({'doc_id': current_batch_doc_ids[k], 'score': score})


        reranked_results.sort(key=lambda x: x['score'], reverse=True)

        rerank_rank = 0
        for rank, res in enumerate(reranked_results, 1):
            if res['doc_id'] == ground_truth_doc_id:
                rerank_rank = rank
                break
            if rank >= args.top_k_rerank: 
                break
        all_rerank_ranks.append(rerank_rank)

    # --- Calculate MRR Scores ---
    mrr_retrieval = calculate_mrr(all_retrieval_ranks)
    mrr_rerank = calculate_mrr(all_rerank_ranks)

    print("\n--- Evaluation Results ---")
    print(f"Total queries processed: {len(eval_data)}")
    print(f"Queries skipped (missing query/doc_id/corpus_match): {skipped_queries_count}")
    print(f"MRR @{args.top_k_retrieval} (Retrieval): {mrr_retrieval:.4f}")
    print(f"MRR @{args.top_k_rerank} (Reranking): {mrr_rerank:.4f}")

if __name__ == "__main__":
    main()