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

# import json
# import argparse
# import numpy as np
# from sentence_transformers import SentenceTransformer, util
# from tqdm import tqdm

# # 加载 QCA 数据（Query, Context, Answer）
# # 这个函数与 finetune_encoder.py 中的 load_qca_for_eval 相同，确保一致性
# def load_qca(jsonl_path, max_samples=None):
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

#                 if q and c:
#                     data.append({'query': q, 'context': c, 'answer': a})
#                 else:
#                     print(f"警告：评估数据行 {i+1}，因为 'query'/'question' 或 'context' 为空或无效。原始行: {line.strip()}")
#             except json.JSONDecodeError:
#                 print(f"错误：跳过评估数据行 {i+1}，非法的 JSON 格式: {line.strip()}")
#             except Exception as e:
#                 print(f"处理评估数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
#     return data

# def evaluate_encoder_mrr(model_name, eval_jsonl, max_samples=None, print_examples=0):
#     print(f"--- 开始评估模型: {model_name} ---")
    
#     try:
#         model = SentenceTransformer(model_name)
#     except Exception as e:
#         print(f"错误：无法加载 SentenceTransformer 模型 '{model_name}'。请确保模型路径正确或模型ID有效。")
#         print(f"详细错误信息: {e}")
#         return

#     print(f"加载评估数据：{eval_jsonl}")
#     eval_data = load_qca(eval_jsonl, max_samples)

#     if not eval_data:
#         print("警告：没有加载到有效的评估样本，无法进行MRR评估。")
#         return

#     queries = [item['query'] for item in eval_data]
#     contexts = [item['context'] for item in eval_data]
#     answers = [item['answer'] for item in eval_data]

#     print(f"编码 {len(contexts)} 个上下文...")
#     context_embeddings = model.encode(contexts, batch_size=64, convert_to_tensor=True, show_progress_bar=True)

#     mrrs = []
#     example_count = 0

#     print(f"评估 {len(queries)} 个查询的 MRR...")
#     iterator = tqdm(eval_data, desc='计算 MRR', disable=False) # 始终显示进度条
#     for i, item in enumerate(iterator):
#         query_emb = model.encode(item['query'], convert_to_tensor=True)
#         scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()

#         # 假设 item['context'] 是其 item['query'] 的正确上下文
#         # 因此正确上下文在原始 contexts 列表中的索引就是当前的 i
#         target_context_idx = i 

#         # 获取按分数降序排列的索引
#         sorted_indices = np.argsort(scores)[::-1]
        
#         rank = -1
#         for r, idx in enumerate(sorted_indices):
#             if idx == target_context_idx:
#                 rank = r + 1
#                 break
        
#         if rank != -1:
#             mrrs.append(1.0 / rank)
#         else:
#             mrrs.append(0.0) # 如果正确上下文未被检索到，Reciprocal Rank 为 0

#         # 打印示例
#         if print_examples > 0 and example_count < print_examples:
#             print(f"\n--- 示例 {example_count + 1} ---")
#             print(f"查询 (Query): {item['query']}")
#             print(f"真实上下文 (Gold Context): {item['context']}")
#             print(f"真实答案 (Gold Answer): {item['answer']}")
#             print("检索结果 (Top 3 Contexts):")
#             for k in range(min(3, len(sorted_indices))):
#                 retrieved_idx = sorted_indices[k]
#                 retrieved_context = contexts[retrieved_idx]
#                 retrieved_score = scores[retrieved_idx]
#                 is_correct = " (正确)" if retrieved_idx == target_context_idx else ""
#                 print(f"  Top {k+1} (分数: {retrieved_score:.4f}){is_correct}: {retrieved_context}")
#             example_count += 1

#     final_mrr = np.mean(mrrs) if mrrs else 0.0
#     print(f"\n最终 MRR: {final_mrr:.4f}")
#     print(f"--- 模型评估结束 ---")
#     return final_mrr

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name', type=str, required=True, help='预训练模型名或路径')
#     parser.add_argument('--eval_jsonl', type=str, required=True, help='评估数据jsonl文件路径')
#     parser.add_argument('--max_samples', type=int, default=None, help='评估的最大样本数')
#     parser.add_argument('--print_examples', type=int, default=0, help='打印多少个查询的检索示例')
#     args = parser.parse_args()

#     evaluate_encoder_mrr(
#         model_name=args.model_name,
#         eval_jsonl=args.eval_jsonl,
#         max_samples=args.max_samples,
#         print_examples=args.print_examples
#     )

import json
import argparse
from sentence_transformers import SentenceTransformer, util, CrossEncoder # 导入 CrossEncoder
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
                
                # 优化：使用 .get() 方法，更健壮地获取键值
                q_raw = item.get('query') or item.get('question', '')
                q = str(q_raw).strip() 
                
                c_raw = item.get('context', '')
                c = str(c_raw).strip()
                
                a_raw = item.get('answer', '')
                a = str(a_raw).strip() 

                if q and c: # 确保问题和上下文都非空
                    data.append({'query': q, 'context': c, 'answer': a})
                else:
                    print(f"警告：跳过数据行 {i+1}，因为 'query'/'question' 或 'context' 字段为空或无效。原始行: {line.strip()}")
            except json.JSONDecodeError:
                print(f"错误：跳过非法的 JSON 行 {i+1}: {line.strip()}")
            except Exception as e:
                print(f"处理数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
    return data

def compute_mrr(model, data, top_k=10, reranker_top_k=100, print_examples=5, reranker_model_name=None): # 新增 reranker_model_name 和 reranker_top_k 参数
    if not data:
        print("评估数据为空，无法计算MRR。")
        return 0.0

    queries = [item['query'] for item in data]
    contexts = [item['context'] for item in data]
    
    print(f"编码 {len(contexts)} 个上下文...")
    # 优化：提高评估时编码的 batch_size
    context_embeddings = model.encode(contexts, batch_size=256, convert_to_tensor=True, show_progress_bar=True) 
    
    # --- Re-Ranker 加载 ---
    reranker = None
    if reranker_model_name:
        print(f"加载 Re-Ranker 模型: {reranker_model_name}")
        try:
            reranker = CrossEncoder(reranker_model_name)
        except Exception as e:
            print(f"错误：无法加载 Re-Ranker 模型 '{reranker_model_name}'。将不使用 Re-Ranker。错误: {e}")
            reranker_model_name = None # 禁用 Re-Ranker
    # ----------------------

    mrrs = []
    for i, item in enumerate(tqdm(data, desc='Evaluating MRR')):
        query_emb = model.encode(item['query'], convert_to_tensor=True)
        
        # 1. 初始检索 (Bi-Encoder)
        scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()
        initial_sorted_indices = np.argsort(scores)[::-1] # 初始排序的索引
        
        target_context_idx = i 

        # 2. Re-Ranking 阶段 (如果 Re-Ranker 存在)
        if reranker:
            # 获取 Re-Ranker 需要的 top-K 候选文档（可以比最终 top_k 大）
            # 这里 reranker_top_k 默认值改为 100，可以根据需要调整
            candidates_for_reranking_indices = initial_sorted_indices[:reranker_top_k]
            candidate_texts = [contexts[idx] for idx in candidates_for_reranking_indices]
            
            # 构建 (query, candidate_text) 对
            sentence_pairs = [[item['query'], text] for text in candidate_texts]
            
            # 使用 Re-Ranker 预测得分
            reranker_scores = reranker.predict(sentence_pairs)
            
            # 根据 Re-Ranker 的得分重新排序这些候选文档
            reranked_candidate_indices_with_scores = sorted(zip(reranker_scores, candidates_for_reranking_indices), key=lambda x: x[0], reverse=True)
            
            # 最终用于 MRR 计算的排序索引来自 Re-Ranker
            final_sorted_indices = [idx for score, idx in reranked_candidate_indices_with_scores]
            
            # 重要：确保即使 Re-Ranker 没有把 GT 包含在 top-reranker_top_k 里，也能在整个语料中找到
            # 这在理论上不应该发生，因为 target_context_idx 应该总在 initial_sorted_indices 里
            # 但作为鲁棒性考虑，如果 reranker_top_k 设得太小，可能会漏掉。
            # 这种情况下，GT会被置于 reranker_top_k 之外，但仍然被考虑在MRR计算中
            if target_context_idx not in final_sorted_indices:
                 final_sorted_indices.extend([idx for idx in initial_sorted_indices if idx not in final_sorted_indices])
                
        else: # 没有 Re-Ranker，就用初始检索结果
            final_sorted_indices = initial_sorted_indices
        
        # 3. 根据最终排序计算 MRR
        rank = -1
        for r, idx in enumerate(final_sorted_indices):
            if idx == target_context_idx: 
                rank = r + 1 
                break
        
        if rank != -1:
            mrr_score = 1.0 / rank
            mrrs.append(mrr_score)
        else:
            mrrs.append(0.0)

        # 打印示例（根据 Re-Ranker 后的结果打印）
        if i < print_examples:
            print(f"Q: {item['query']}")
            print(f"GT Context: {item['context'][:80]}...") 
            print(f"GT Answer: {item['answer']}")
            print(f"Top-{top_k} Retrieved (After Re-Ranking if applied):") 
            # 打印 Re-Ranker 后的 top_k 结果
            for j in range(min(top_k, len(final_sorted_indices))):
                retrieved_idx = final_sorted_indices[j]
                is_gt = " (GT)" if retrieved_idx == target_context_idx else ""
                # 注意这里打印的 score 是 Bi-Encoder 的原始 score，Re-Ranker 的 score 需要另外记录
                # 暂时还是用 Bi-Encoder 的分数，因为 Re-Ranker 的分数只对 top-k 才有，且尺度不同
                print(f"  Rank {j+1}: {contexts[retrieved_idx][:80]} ... Score: {scores[retrieved_idx]:.4f}{is_gt}") 
            
            print(f"Target Context Rank: {rank if rank != -1 else 'Not Found'}")
            print(f"Reciprocal Rank: {mrr_score if rank != -1 else 0.0}\n{'-'*40}")

    final_mrr = np.mean(mrrs)
    print(f"\nMean Reciprocal Rank (MRR): {final_mrr:.4f}")
    return final_mrr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='预训练或微调的Encoder模型名或路径')
    parser.add_argument('--eval_jsonl', type=str, required=True, help='评估数据jsonl文件路径')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评估样本数')
    parser.add_argument('--print_examples', type=int, default=5, help='打印示例数量')
    parser.add_argument('--reranker_model_name', type=str, default=None, help='Re-Ranker 模型名或路径 (可选)') # <-- 这两行是新增的
    parser.add_argument('--reranker_top_k', type=int, default=100, help='Re-Ranker 处理的顶端候选项数量') # <-- 这两行是新增的
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
        compute_mrr(model, data, print_examples=args.print_examples, 
                      reranker_model_name=args.reranker_model_name, # 传入 Re-Ranker 参数
                      reranker_top_k=args.reranker_top_k) # 传入 Re-Ranker 参数