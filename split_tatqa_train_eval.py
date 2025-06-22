import json
import random

def split_qca_jsonl(
    input_jsonl,
    train_jsonl,
    eval_jsonl,
    train_ratio=0.8,
    seed=42
):
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
    random.seed(seed)
    random.shuffle(lines)
    n_train = int(len(lines) * train_ratio)
    train_lines = lines[:n_train]
    eval_lines = lines[n_train:]

    # 训练集只保留Q-C对
    with open(train_jsonl, "w", encoding="utf-8") as f_train:
        for line in train_lines:
            item = json.loads(line)
            f_train.write(json.dumps({
                "query": item.get("question", ""),
                "context": item.get("context", "")
            }, ensure_ascii=False) + "\n")

    # 评测集保留Q-C-A
    with open(eval_jsonl, "w", encoding="utf-8") as f_eval:
        for line in eval_lines:
            f_eval.write(line)

    print(f"训练集: {train_jsonl} ({len(train_lines)}条)")
    print(f"评测集: {eval_jsonl} ({len(eval_lines)}条)")

if __name__ == "__main__":
    split_qca_jsonl(
        input_jsonl="data/tatqa_rag_ready.jsonl",      # TatQA的Q-C-A数据
        train_jsonl="data/tatqa_train_qc.jsonl",       # 输出TatQA训练集Q-C
        eval_jsonl="data/tatqa_eval.jsonl",            # 输出TatQA评测集Q-C-A
        train_ratio=0.8
    ) 