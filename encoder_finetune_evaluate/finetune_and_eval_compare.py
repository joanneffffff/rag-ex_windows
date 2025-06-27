import os
import argparse
import subprocess

def run_eval(model_name, eval_jsonl, max_samples, print_examples, tag):
    print(f"\n==== [{tag}] 评测模型: {model_name} ====")
    cmd = [
        "python", "run_encoder_eval.py",
        "--model_name", model_name,
        "--eval_jsonl", eval_jsonl,
        "--max_samples", str(max_samples),
        "--print_examples", str(print_examples)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    # 提取MRR分数
    mrr = None
    for line in result.stdout.splitlines():
        if "Mean Reciprocal Rank" in line or "MRR" in line:
            try:
                mrr = float(line.strip().split()[-1])
            except Exception:
                pass
    return mrr

def run_finetune(model_name, train_jsonl, output_dir, batch_size, epochs, max_samples):
    print(f"\n==== 开始finetune: {model_name} ====")
    cmd = [
        "python", "finetune_encoder.py",
        "--model_name", model_name,
        "--train_jsonl", train_jsonl,
        "--output_dir", output_dir,
        "--batch_size", str(batch_size),
        "--epochs", str(epochs),
        "--max_samples", str(max_samples)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

def main():
    parser = argparse.ArgumentParser(description="一键finetune+评测对比脚本")
    parser.add_argument('--model_name', type=str, required=True, help='原始模型名')
    parser.add_argument('--train_jsonl', type=str, required=True, help='finetune用Q-C对jsonl')
    parser.add_argument('--eval_jsonl', type=str, required=True, help='评测用Q-C-A jsonl')
    parser.add_argument('--output_dir', type=str, default='finetuned_model', help='finetuned模型输出目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=10000)
    parser.add_argument('--eval_samples', type=int, default=1000)
    parser.add_argument('--print_examples', type=int, default=3)
    args = parser.parse_args()

    # 1. 评测原始模型
    mrr_before = run_eval(
        model_name=args.model_name,
        eval_jsonl=args.eval_jsonl,
        max_samples=args.eval_samples,
        print_examples=args.print_examples,
        tag="原始模型"
    )
    print(f"\n[原始模型] MRR: {mrr_before}")

    # 2. finetune
    run_finetune(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_samples=args.max_samples
    )

    # 3. 评测finetuned模型
    mrr_after = run_eval(
        model_name=args.output_dir,
        eval_jsonl=args.eval_jsonl,
        max_samples=args.eval_samples,
        print_examples=args.print_examples,
        tag="Finetuned模型"
    )
    print(f"\n[Finetuned模型] MRR: {mrr_after}")

    # 4. 对比输出
    print("\n==== MRR对比结果 ====")
    print(f"原始模型:   {mrr_before}")
    print(f"Finetuned:  {mrr_after}")
    if mrr_after and mrr_before:
        print(f"提升倍数:   {mrr_after / mrr_before if mrr_before > 0 else 'N/A'}")

if __name__ == "__main__":
    main() 