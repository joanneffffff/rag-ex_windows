import json
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import argparse
import os

def load_qc_pairs(jsonl_path, max_samples=None):
    pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            q = item['query'].strip()
            c = item['context'].strip()
            if q and c:
                pairs.append(InputExample(texts=[q, c], label=1.0))
    return pairs

def finetune_encoder(model_name, train_jsonl, output_dir, batch_size=32, epochs=1, max_samples=None):
    train_examples = load_qc_pairs(train_jsonl, max_samples)
    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='预训练模型名')
    parser.add_argument('--train_jsonl', type=str, required=True, help='训练数据jsonl')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    finetune_encoder(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_samples=args.max_samples
    )
# 示例：
# 中文: python finetune_encoder.py --model_name Langboat/mengzi-bert-base-fin --train_jsonl data/alphafin/alphafin_train.jsonl --output_dir output/mengzi-finetuned
# 英文: python finetune_encoder.py --model_name sentence-transformers/all-mpnet-base-v2 --train_jsonl data/tatqa/tatqa_train.jsonl --output_dir output/mpnet-finetuned 