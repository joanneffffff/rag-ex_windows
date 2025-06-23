import argparse
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections import defaultdict
import ast # 导入 ast 模块

# --- Utility Function for Context Conversion (from split_alphafin_data.py) ---
def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    """
    Parses a JSON string context from AlphaFin and converts it into a list of
    natural language chunks, handling various formats and cleaning.
    """
    if not json_str_context:
        return []

    # Replace literal "\n" with actual newline character for proper parsing
    processed_str_context = json_str_context.replace("\\n", "\n")

    chunks = []
    
    # Attempt to parse as a list of dicts first
    try:
        data_list = json.loads(processed_str_context)
        if isinstance(data_list, list):
            for item in data_list:
                if isinstance(item, dict):
                    # Prefer 'content' if available, otherwise iterate through k-v pairs
                    if 'content' in item and isinstance(item['content'], str):
                        chunks.append(item['content'])
                    else:
                        chunk_parts = []
                        for k, v in item.items():
                            if isinstance(v, str) and v.strip():
                                chunk_parts.append(f"{k}: {v.strip()}")
                            elif isinstance(v, (int, float)):
                                chunk_parts.append(f"{k}: {v}")
                        if chunk_parts:
                            chunks.append("。".join(chunk_parts))
                elif isinstance(item, str) and item.strip():
                    chunks.append(item.strip())
            if chunks:
                final_context = "\n\n".join(chunks)
                # Apply general cleaning to the final joined context
                final_context = re.sub(r'【.*?】', '', final_context) # Remove specific brackets
                final_context = re.sub(r'[\u3000\xa0]', ' ', final_context) # Replace unicode spaces
                final_context = re.sub(r'\s+', ' ', final_context).strip() # Normalize all whitespace
                return [final_context] # Return as a single chunk for simplicity in this case if joined
    except json.JSONDecodeError:
        pass # Not a valid JSON list, try other formats

    # Attempt to parse as a single dictionary
    try:
        data_dict = json.loads(processed_str_context)
        if isinstance(data_dict, dict):
            chunk_parts = []
            if 'content' in data_dict and isinstance(data_dict['content'], str):
                chunk_parts.append(data_dict['content'])
            else:
                for k, v in data_dict.items():
                    if isinstance(v, str) and v.strip():
                        chunk_parts.append(f"{k}: {v.strip()}")
                    elif isinstance(v, (int, float)):
                        chunk_parts.append(f"{k}: {v}")
            if chunk_parts:
                final_context = "。".join(chunk_parts)
                final_context = re.sub(r'【.*?】', '', final_context)
                final_context = re.sub(r'[\u3000\xa0]', ' ', final_context)
                final_context = re.sub(r'\s+', ' ', final_context).strip()
                return [final_context]
    except json.JSONDecodeError:
        pass # Not a valid JSON dict, try other formats

    # Try literal_eval for cases that look like Python lists/dicts but not strict JSON
    try:
        parsed_data = ast.literal_eval(processed_str_context)
        if isinstance(parsed_data, list):
            for item in parsed_data:
                if isinstance(item, dict):
                    chunk_parts = []
                    if 'content' in item and isinstance(item['content'], str):
                        chunk_parts.append(item['content'])
                    else:
                        for k, v in item.items():
                            if isinstance(v, str) and v.strip():
                                chunk_parts.append(f"{k}: {v.strip()}")
                            elif isinstance(v, (int, float)):
                                chunk_parts.append(f"{k}: {v}")
                    if chunk_parts:
                        chunks.append("。".join(chunk_parts))
                elif isinstance(item, str) and item.strip():
                    chunks.append(item.strip())
            if chunks:
                final_context = "\n\n".join(chunks)
                final_context = re.sub(r'【.*?】', '', final_context)
                final_context = re.sub(r'[\u3000\xa0]', ' ', final_context)
                final_context = re.sub(r'\s+', ' ', final_context).strip()
                return [final_context]
        elif isinstance(parsed_data, dict):
            chunk_parts = []
            if 'content' in parsed_data and isinstance(parsed_data['content'], str):
                chunk_parts.append(parsed_data['content'])
            else:
                for k, v in parsed_data.items():
                    if isinstance(v, str) and v.strip():
                        chunk_parts.append(f"{k}: {v.strip()}")
                    elif isinstance(v, (int, float)):
                        chunk_parts.append(f"{k}: {v}")
            if chunk_parts:
                final_context = "。".join(chunk_parts)
                final_context = re.sub(r'【.*?】', '', final_context)
                final_context = re.sub(r'[\u3000\xa0]', ' ', final_context)
                final_context = re.sub(r'\s+', ' ', final_context).strip()
                return [final_context]
    except (ValueError, SyntaxError):
        pass # Not a valid literal, treat as raw text

    # Fallback: Treat as raw text if no structured parsing is successful
    # Apply general cleaning similar to how Q and A are cleaned
    cleaned_text = processed_str_context.replace('，', ',')
    cleaned_text = cleaned_text.replace('：', ':')
    cleaned_text = cleaned_text.replace('。', ',') # This might be aggressive, depending on desired outcome
    cleaned_text = re.sub(r'【.*?】', '', cleaned_text) # Remove specific brackets
    cleaned_text = re.sub(r'[\u3000\xa0]', ' ', cleaned_text) # Replace unicode spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip() # Normalize all whitespace

    if "原始格式" in cleaned_text or not cleaned_text: # Check for default unparsed messages or empty
        # If the original text is still looking like unparsed JSON structure,
        # or it's empty after all attempts, mark it as '解析失败'
        return [f"原始格式，解析失败或无有效结构：{cleaned_text[:50]}..."] # Added a snippet for debug
    return [cleaned_text] # Return the cleaned raw text as a single chunk


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

    # --- Load Encoder Model and Tokenizer ---
    print(f"Loading encoder model: {args.encoder_model_name}...")
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_model_name)
    encoder_model = AutoModel.from_pretrained(args.encoder_model_name).to(device)
    encoder_model.eval()

    encoder_max_length = encoder_tokenizer.model_max_length
    if encoder_max_length > 1024: 
        print(f"Warning: Encoder model_max_length is {encoder_max_length}. Capping to 512 for typical BERT models.")
        encoder_max_length = 512 

    # --- Load Reranker Model and Tokenizer (Using AutoModelForCausalLM as per Qwen's example) ---
    print(f"Loading reranker model: {args.reranker_model_name}...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name, padding_side='left') 
    reranker_model = AutoModelForCausalLM.from_pretrained(args.reranker_model_name).to(device) 
    reranker_model.eval()

    token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    reranker_max_length = 8192 

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

    def format_instruction(instruction, query, doc):
        fixed_instruction = 'Given a web search query, retrieve relevant passages that answer the query' 
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=fixed_instruction, query=query, doc=doc)
        return output

    def process_reranker_inputs(pairs):
        inputs = reranker_tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=reranker_max_length - len(prefix_tokens) - len(suffix_tokens) 
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=reranker_max_length) 
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
    # !!! 核心改动在这里：模仿 split_alphafin_data.py 的逻辑来构建语料库 !!!
    print(f"Loading and processing corpus from: {args.base_raw_data_path} (mimicking split_alphafin_data.py logic)...")
    corpus_documents = {}
    
    with open(args.base_raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    processed_corpus_count = 0
    skipped_corpus_items = 0

    for idx, item in enumerate(tqdm(raw_data, desc="Building corpus map with aligned IDs")):
        # Assign doc_id based on original index, consistent with split_alphafin_data.py
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
            "原始格式，无有效字典" in natural_language_chunks[0] or 
            "原始格式，解析失败" in natural_language_chunks[0] or 
            "原始格式，无有效结构" in natural_language_chunks[0]
        ):
            is_parse_failed_chunk = True
            
        c = "\n\n".join(natural_language_chunks) 
        
        # Only add to corpus if it would have been included in eval.jsonl by split_alphafin_data.py
        if q and a and not is_parse_failed_chunk: 
            corpus_documents[doc_id] = c # Use the processed context
            processed_corpus_count += 1
        else:
            skipped_corpus_items += 1
            # print(f"Warning: Skipping corpus item with original ID {doc_id} due to invalid Q/A/context. Q: {q[:50]}, A: {a[:50]}. Context: {original_json_context_str[:100]}...") # Too verbose if many
            
    print(f"Corpus built with {len(corpus_documents)} documents (after filtering).")
    print(f"Skipped {skipped_corpus_items} items from raw data while building corpus.")
    
    # Generate embeddings for all corpus documents
    corpus_ids = list(corpus_documents.keys())
    corpus_texts = [corpus_documents[doc_id] for doc_id in corpus_ids]
    corpus_embeddings = []

    print("Generating embeddings for corpus documents...")
    batch_size = 16 
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding corpus"):
        batch_texts = corpus_texts[i:i + batch_size] 
        with torch.no_grad():
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
    with open(args.eval_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            eval_data.append(json.loads(line))
    print(f"Loaded {len(eval_data)} evaluation queries.")

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
        # 语料库构建逻辑已与eval.jsonl一致，此处判断应该更少触发
        if ground_truth_doc_id not in corpus_documents:
            skipped_queries_count += 1
            # 仅在实际发生时打印，避免大量重复警告
            # print(f"Warning: 'doc_id': '{ground_truth_doc_id}' from eval_jsonl not found in CORPUS_DOCUMENTS. Skipping query: '{query_text[:50]}...'.")
            continue
        
        # --- 1. Retrieval (Encoder) ---
        with torch.no_grad():
            query_encoded = encoder_tokenizer(
                query_text, 
                padding='max_length', 
                truncation=True, 
                max_length=encoder_max_length, 
                return_tensors='pt'
            ).to(device)
            query_embedding = mean_pooling(encoder_model(**query_encoded), query_encoded['attention_mask'])
            query_embedding = F.normalize(query_embedding, p=2, dim=1)

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
                rerank_data_for_qwen.append((formatted_text, doc_id)) 

        if not rerank_data_for_qwen:
            all_rerank_ranks.append(0)
            continue
        
        reranked_results = []
        reranker_batch_size = 16 
        for j in range(0, len(rerank_data_for_qwen), reranker_batch_size):
            current_batch_pairs = [item[0] for item in rerank_data_for_qwen[j:j + reranker_batch_size]]
            current_batch_doc_ids = [item[1] for item in rerank_data_for_qwen[j:j + reranker_batch_size]]

            rerank_inputs_batch = process_reranker_inputs(current_batch_pairs)
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