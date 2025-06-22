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
            try:
                item = json.loads(line)
                
                # 强制转换为字符串再 strip()
                q_raw = item['query'] if 'query' in item else item.get('question', '')
                q = str(q_raw).strip() 
                
                c_raw = item.get('context', '')
                c = str(c_raw).strip()
                
                a_raw = item.get('answer', '')
                a = str(a_raw).strip() # 答案也可能不是字符串

                if q and c: # 确保问题和上下文都非空
                    data.append({'query': q, 'context': c, 'answer': a})
                else:
                    print(f"警告：跳过数据行 {i+1}，因为 'query'/'question' 或 'context' 字段为空或无效。原始行: {line.strip()}")
            except json.JSONDecodeError:
                print(f"错误：跳过非法的 JSON 行 {i+1}: {line.strip()}")
            except Exception as e:
                print(f"处理数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
    return data

# compute_mrr 函数和 if __name__ == "__main__": 部分与之前提供的一致，无需改动
# （为了简洁，这里省略了，请将它们添加到你的文件中）

def compute_mrr(model, data, top_k=10, print_examples=5):
    if not data:
        print("评估数据为空，无法计算MRR。")
        return 0.0

    queries = [item['query'] for item in data]
    contexts = [item['context'] for item in data]
    
    print(f"编码 {len(contexts)} 个上下文...")
    context_embeddings = model.encode(contexts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    mrrs = []
    for i, item in enumerate(tqdm(data, desc='Evaluating MRR')):
        query_emb = model.encode(item['query'], convert_to_tensor=True)
        
        scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()
        
        target_context_idx = i 

        sorted_indices = np.argsort(scores)[::-1]
        
        rank = -1
        for r, idx in enumerate(sorted_indices):
            if idx == target_context_idx: 
                rank = r + 1 
                break
        
        if rank != -1:
            mrr_score = 1.0 / rank
            mrrs.append(mrr_score)
        else:
            mrrs.append(0.0)

        if i < print_examples:
            print(f"Q: {item['query']}")
            print(f"GT Context: {item['context'][:80]}...") 
            print(f"GT Answer: {item['answer']}")
            print(f"Top-{top_k} Retrieved:") 
            for j in range(min(top_k, len(sorted_indices))):
                retrieved_idx = sorted_indices[j]
                is_gt = " (GT)" if retrieved_idx == target_context_idx else ""
                print(f"  Rank {j+1}: {contexts[retrieved_idx][:80]} ... Score: {scores[retrieved_idx]:.4f}{is_gt}")
            
            print(f"Target Context Rank: {rank if rank != -1 else 'Not Found'}")
            print(f"Reciprocal Rank: {mrr_score if rank != -1 else 0.0}\n{'-'*40}")

    final_mrr = np.mean(mrrs)
    print(f"\nMean Reciprocal Rank (MRR): {final_mrr:.4f}")
    return final_mrr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='预训练模型名或路径')
    parser.add_argument('--eval_jsonl', type=str, required=True, help='评估数据jsonl文件路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评估样本数')
    args = parser.parse_args()
    
    print(f"加载模型: {args.model_name}")
    try:
        model = SentenceTransformer(args.model_name)
    except Exception as e:
        print(f"错误：无法加载 SentenceTransformer 模型 '{args.model_name}'。请确保模型路径正确或模型ID有效。")
        print(f"详细错误信息: {e}")
        exit()

    print(f"加载评估数据：{args.eval_jsonl}")
    data = load_qca(args.eval_jsonl, args.max_samples)
    
    if not data:
        print("没有加载到有效的评估样本，请检查数据文件和键名。")
    else:
        print(f"加载了 {len(data)} 个评估样本。")
        compute_mrr(model, data)