import json
import argparse
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

def load_qca(jsonl_path, max_samples=None):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            q = item['query'] if 'query' in item else item.get('question', '')
            c = item['context']
            a = item.get('answer', '')
            data.append({'query': q, 'context': c, 'answer': a})
    return data

def compute_mrr(model, data, top_k=10, print_examples=5):
    queries = [item['query'] for item in data]
    contexts = [item['context'] for item in data]
    answers = [item['answer'] for item in data]
    context_embeddings = model.encode(contexts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    mrrs = []
    for i, item in enumerate(tqdm(data, desc='Evaluating MRR')):
        query_emb = model.encode(item['query'], convert_to_tensor=True)
        scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()
        sorted_idx = np.argsort(scores)[::-1]
        rank = None
        for r, idx in enumerate(sorted_idx):
            # 答案是否在context中出现，answer转为str
            if item['answer'] and str(item['answer']) in contexts[idx]:
                rank = r + 1
                break
        if rank:
            mrrs.append(1.0 / rank)
        else:
            mrrs.append(0.0)
        if i < print_examples:
            print(f"Q: {item['query']}")
            print(f"GT Answer: {item['answer']}")
            print(f"Top-3 Retrieved:")
            for j in range(3):
                print(f"  Rank {j+1}: {contexts[sorted_idx[j]][:80]} ... Score: {scores[sorted_idx[j]]:.4f}")
            print(f"Reciprocal Rank: {1.0/rank if rank else 0.0}\n{'-'*40}")
    mrr = np.mean(mrrs)
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")
    return mrr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_jsonl', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    model = SentenceTransformer(args.model_name)
    data = load_qca(args.eval_jsonl, args.max_samples)
    compute_mrr(model, data)
# 用法示例：
# python evaluate_encoder_mrr.py --model_name Langboat/mengzi-bert-base-fin --eval_jsonl data/alphafin/alphafin_eval.jsonl
# python evaluate_encoder_mrr.py --model_name sentence-transformers/all-mpnet-base-v2 --eval_jsonl data/tatqa/tatqa_eval.jsonl 