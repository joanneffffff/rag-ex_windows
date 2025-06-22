# import argparse
# import sys
# import os

# # 尝试直接import评测函数
# try:
#     from evaluate_encoder_mrr import load_qca, compute_mrr
# except ImportError:
#     print("请确保evaluate_encoder_mrr.py与本脚本在同一目录下，或将其路径加入sys.path。")
#     sys.exit(1)

# def main():
#     parser = argparse.ArgumentParser(description="Encoder自动化MRR评测脚本")
#     parser.add_argument('--model_name', type=str, required=True, help='编码器模型名')
#     parser.add_argument('--eval_jsonl', type=str, required=True, help='评测集jsonl文件')
#     parser.add_argument('--max_samples', type=int, default=None, help='最大评测样本数')
#     parser.add_argument('--top_k', type=int, default=10, help='MRR评测top_k')
#     parser.add_argument('--print_examples', type=int, default=5, help='输出样例数')
#     args = parser.parse_args()

#     from sentence_transformers import SentenceTransformer
#     print(f"\n加载模型: {args.model_name}")
#     model = SentenceTransformer(args.model_name)
#     print(f"加载评测集: {args.eval_jsonl}")
#     data = load_qca(args.eval_jsonl, args.max_samples)
#     print(f"共加载{len(data)}条样本，开始评测...")
#     mrr = compute_mrr(model, data, top_k=args.top_k, print_examples=args.print_examples)
#     print(f"\n最终MRR分数: {mrr:.4f}")

# if __name__ == "__main__":
#     main() 

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# 加载 QCA 数据（Query, Context, Answer）
# 这个函数与 finetune_encoder.py 中的 load_qca_for_eval 相同，确保一致性
def load_qca(jsonl_path, max_samples=None):
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                q_raw = item['query'] if 'query' in item else item.get('question', '')
                q = str(q_raw).strip() 
                
                c_raw = item.get('context', '')
                c = str(c_raw).strip()
                
                a_raw = item.get('answer', '')
                a = str(a_raw).strip() 

                if q and c:
                    data.append({'query': q, 'context': c, 'answer': a})
                else:
                    print(f"警告：评估数据行 {i+1}，因为 'query'/'question' 或 'context' 为空或无效。原始行: {line.strip()}")
            except json.JSONDecodeError:
                print(f"错误：跳过评估数据行 {i+1}，非法的 JSON 格式: {line.strip()}")
            except Exception as e:
                print(f"处理评估数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
    return data

def evaluate_encoder_mrr(model_name, eval_jsonl, max_samples=None, print_examples=0):
    print(f"--- 开始评估模型: {model_name} ---")
    
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"错误：无法加载 SentenceTransformer 模型 '{model_name}'。请确保模型路径正确或模型ID有效。")
        print(f"详细错误信息: {e}")
        return

    print(f"加载评估数据：{eval_jsonl}")
    eval_data = load_qca(eval_jsonl, max_samples)

    if not eval_data:
        print("警告：没有加载到有效的评估样本，无法进行MRR评估。")
        return

    queries = [item['query'] for item in eval_data]
    contexts = [item['context'] for item in eval_data]
    answers = [item['answer'] for item in eval_data]

    print(f"编码 {len(contexts)} 个上下文...")
    context_embeddings = model.encode(contexts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

    mrrs = []
    example_count = 0

    print(f"评估 {len(queries)} 个查询的 MRR...")
    iterator = tqdm(eval_data, desc='计算 MRR', disable=False) # 始终显示进度条
    for i, item in enumerate(iterator):
        query_emb = model.encode(item['query'], convert_to_tensor=True)
        scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()

        # 假设 item['context'] 是其 item['query'] 的正确上下文
        # 因此正确上下文在原始 contexts 列表中的索引就是当前的 i
        target_context_idx = i 

        # 获取按分数降序排列的索引
        sorted_indices = np.argsort(scores)[::-1]
        
        rank = -1
        for r, idx in enumerate(sorted_indices):
            if idx == target_context_idx:
                rank = r + 1
                break
        
        if rank != -1:
            mrrs.append(1.0 / rank)
        else:
            mrrs.append(0.0) # 如果正确上下文未被检索到，Reciprocal Rank 为 0

        # 打印示例
        if print_examples > 0 and example_count < print_examples:
            print(f"\n--- 示例 {example_count + 1} ---")
            print(f"查询 (Query): {item['query']}")
            print(f"真实上下文 (Gold Context): {item['context']}")
            print(f"真实答案 (Gold Answer): {item['answer']}")
            print("检索结果 (Top 3 Contexts):")
            for k in range(min(3, len(sorted_indices))):
                retrieved_idx = sorted_indices[k]
                retrieved_context = contexts[retrieved_idx]
                retrieved_score = scores[retrieved_idx]
                is_correct = " (正确)" if retrieved_idx == target_context_idx else ""
                print(f"  Top {k+1} (分数: {retrieved_score:.4f}){is_correct}: {retrieved_context}")
            example_count += 1

    final_mrr = np.mean(mrrs) if mrrs else 0.0
    print(f"\n最终 MRR: {final_mrr:.4f}")
    print(f"--- 模型评估结束 ---")
    return final_mrr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='预训练模型名或路径')
    parser.add_argument('--eval_jsonl', type=str, required=True, help='评估数据jsonl文件路径')
    parser.add_argument('--max_samples', type=int, default=None, help='评估的最大样本数')
    parser.add_argument('--print_examples', type=int, default=0, help='打印多少个查询的检索示例')
    args = parser.parse_args()

    evaluate_encoder_mrr(
        model_name=args.model_name,
        eval_jsonl=args.eval_jsonl,
        max_samples=args.max_samples,
        print_examples=args.print_examples
    )