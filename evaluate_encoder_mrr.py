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
                q = item['query'] if 'query' in item else item.get('question', '')
                c = item.get('context', '') # 使用.get()，并提供默认值
                a = item.get('answer', '')
                # 确保q和c非空，否则跳过
                if q and c:
                    data.append({'query': q.strip(), 'context': c.strip(), 'answer': a.strip()})
                else:
                    print(f"警告：跳过数据行 {i+1}，因为 'query'/'question' 或 'context' 为空。行内容: {line.strip()}")
            except json.JSONDecodeError:
                print(f"错误：跳过非法的 JSON 行 {i+1}: {line.strip()}")
            except Exception as e:
                print(f"处理数据行 {i+1} 时发生未知错误: {e} - 行内容: {line.strip()}")
    return data

def compute_mrr(model, data, top_k=10, print_examples=5):
    if not data:
        print("评估数据为空，无法计算MRR。")
        return 0.0

    queries = [item['query'] for item in data]
    contexts = [item['context'] for item in data]
    # 注意：answers在这里不直接用于MRR计算，但在打印示例时有用

    print(f"编码 {len(contexts)} 个上下文...")
    context_embeddings = model.encode(contexts, batch_size=32, convert_to_tensor=True, show_progress_bar=True)
    
    mrrs = []
    for i, item in enumerate(tqdm(data, desc='Evaluating MRR')):
        query_emb = model.encode(item['query'], convert_to_tensor=True)
        
        # 计算当前查询与所有上下文的相似度
        scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()
        
        # 找到当前查询对应的“正确”上下文在 `contexts` 列表中的索引
        # 假设每个 item['context'] 是其 item['query'] 的正确上下文
        try:
            # 找到当前 item['context'] 在所有 contexts 列表中的位置
            # 这要求 contexts 列表是唯一的，或者正确上下文不会重复
            # 否则，这里可能会找到第一个匹配的索引，而不是唯一的那个
            # 更严谨的做法是，如果你的数据是 QCA 结构，那么每个 Q 都有一个唯一的 C
            # 这里的 target_context_idx 就是当前循环的索引 i
            target_context_idx = i 
        except ValueError:
            # 如果 item['context'] 不在 contexts 列表中，这是个数据问题
            print(f"警告：查询 '{item['query']}' 的正确上下文未在总上下文中找到。")
            mrrs.append(0.0)
            continue

        # 将得分按降序排列，获取索引
        sorted_indices = np.argsort(scores)[::-1]
        
        # 找到正确上下文的排名
        rank = -1
        for r, idx in enumerate(sorted_indices):
            if idx == target_context_idx: # 如果当前检索到的索引就是目标索引
                rank = r + 1 # 排名是从1开始的
                break
        
        if rank != -1:
            mrr_score = 1.0 / rank
            mrrs.append(mrr_score)
        else:
            # 理论上应该不会发生，除非 target_context_idx 无法通过排序找到
            # 这表明正确上下文没有在 top-k 中，或者数据有问题
            mrrs.append(0.0)

        # 打印示例
        if i < print_examples:
            print(f"Q: {item['query']}")
            print(f"GT Context: {item['context'][:80]}...") # 打印原始的GT Context
            print(f"GT Answer: {item['answer']}")
            print(f"Top-{top_k} Retrieved:") # 打印 top_k 个检索结果
            for j in range(min(top_k, len(sorted_indices))):
                retrieved_idx = sorted_indices[j]
                # 标记是否是正确上下文
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