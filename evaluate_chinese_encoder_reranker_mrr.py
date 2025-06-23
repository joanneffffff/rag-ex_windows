import argparse
import json
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from collections import defaultdict

# 移除了 clean_text_for_corpus 函数，假设 split_alphafin_data.py 已完成所有清理

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

    # Get max sequence length for the encoder model (e.g., 512 for BERT-base)
    # This is important to ensure truncation
    encoder_max_length = encoder_tokenizer.model_max_length
    if encoder_max_length > 1024: # Some models might have very large max_length, cap it if needed
        print(f"Warning: Encoder model_max_length is {encoder_max_length}. Capping to 512 for typical BERT models.")
        encoder_max_length = 512 # Or adjust based on your specific encoder model's true capability

    # --- Load Reranker Model and Tokenizer (Using AutoModelForCausalLM as per Qwen's example) ---
    print(f"Loading reranker model: {args.reranker_model_name}...")
    reranker_tokenizer = AutoTokenizer.from_pretrained(args.reranker_model_name, padding_side='left') # 注意 padding_side
    reranker_model = AutoModelForCausalLM.from_pretrained(args.reranker_model_name).to(device) # 使用 AutoModelForCausalLM
    reranker_model.eval()

    # Define Reranker specific tokens and logic (from Qwen's example)
    token_false_id = reranker_tokenizer.convert_tokens_to_ids("no")
    token_true_id = reranker_tokenizer.convert_tokens_to_ids("yes")
    # For Qwen Reranker, max_length typically refers to the combined length including prefix/suffix
    reranker_max_length = 8192 # Adjust if needed, based on Qwen reranker's true max length

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = reranker_tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = reranker_tokenizer.encode(suffix, add_special_tokens=False)

    # Function to format instruction for Qwen reranker
    def format_instruction(instruction, query, doc):
        # instruction for reranker should be fixed
        fixed_instruction = 'Given a web search query, retrieve relevant passages that answer the query' 
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=fixed_instruction, query=query, doc=doc)
        return output

    # Function to process inputs for Qwen reranker
    def process_reranker_inputs(pairs):
        inputs = reranker_tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=reranker_max_length - len(prefix_tokens) - len(suffix_tokens) # Use reranker_max_length
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = reranker_tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=reranker_max_length) # Use reranker_max_length
        for key in inputs:
            inputs[key] = inputs[key].to(reranker_model.device)
        return inputs

    # Function to compute logits for Qwen reranker
    @torch.no_grad()
    def compute_reranker_logits(inputs):
        batch_scores = reranker_model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist() # Get probability of "yes"
        return scores


    # --- Load and Process Corpus (base_raw_data_path) ---
    print(f"Loading and processing corpus from: {args.base_raw_data_path}...")
    corpus_documents = {}
    with open(args.base_raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        for idx, item in enumerate(tqdm(raw_data, desc="Building corpus map")):
            doc_id = item.get('id', str(idx)) 
            corpus_documents[doc_id] = str(item.get('context', '')) 
    print(f"Corpus loaded with {len(corpus_documents)} documents.")
    
    # Generate embeddings for all corpus documents
    corpus_ids = list(corpus_documents.keys())
    corpus_texts = [corpus_documents[doc_id] for doc_id in corpus_ids]
    corpus_embeddings = []

    print("Generating embeddings for corpus documents...")
    batch_size = 32 # Adjust based on GPU memory
    for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Embedding corpus"):
        batch_texts = corpus_texts[i:i + batch_size] 
        with torch.no_grad():
            # Pass max_length to encoder_tokenizer for explicit truncation
            encoded_input = encoder_tokenizer(
                batch_texts, 
                padding='max_length', # Pad to max_length
                truncation=True,      # Truncate if longer than max_length
                max_length=encoder_max_length, # Explicitly set max_length for the encoder
                return_tensors='pt'
            ).to(device)
            model_output = encoder_model(**encoded_input)
            embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1) # L2 normalization
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
        if ground_truth_doc_id not in corpus_documents:
            skipped_queries_count += 1
            print(f"Warning: 'doc_id': '{ground_truth_doc_id}' from eval_jsonl not found in corpus. Skipping query: '{query_text[:50]}...'.")
            continue
        
        # --- 1. Retrieval (Encoder) ---
        with torch.no_grad():
            # Pass max_length to encoder_tokenizer for query
            query_encoded = encoder_tokenizer(
                query_text, 
                padding='max_length', 
                truncation=True, 
                max_length=encoder_max_length, # Explicitly set max_length
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