import argparse
import sys
import os

# 尝试直接import评测函数
try:
    from evaluate_encoder_mrr import load_qca, compute_mrr
except ImportError:
    print("请确保evaluate_encoder_mrr.py与本脚本在同一目录下，或将其路径加入sys.path。")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Encoder自动化MRR评测脚本")
    parser.add_argument('--model_name', type=str, required=True, help='编码器模型名')
    parser.add_argument('--eval_jsonl', type=str, required=True, help='评测集jsonl文件')
    parser.add_argument('--max_samples', type=int, default=None, help='最大评测样本数')
    parser.add_argument('--top_k', type=int, default=10, help='MRR评测top_k')
    parser.add_argument('--print_examples', type=int, default=5, help='输出样例数')
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    print(f"\n加载模型: {args.model_name}")
    model = SentenceTransformer(args.model_name)
    print(f"加载评测集: {args.eval_jsonl}")
    data = load_qca(args.eval_jsonl, args.max_samples)
    print(f"共加载{len(data)}条样本，开始评测...")
    mrr = compute_mrr(model, data, top_k=args.top_k, print_examples=args.print_examples)
    print(f"\n最终MRR分数: {mrr:.4f}")

if __name__ == "__main__":
    main() 