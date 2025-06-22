#!/bin/bash

# 1. TatQA英文Encoder评测（GPU）
python run_encoder_eval.py --model_name ProsusAI/finbert --eval_jsonl evaluate_mrr/tatqa_eval.jsonl --max_samples 1000 --print_examples 3

# 2. TatQA英文Encoder微调
python finetune_encoder.py --model_name ProsusAI/finbert --train_jsonl evaluate_mrr/tatqa_train_qc.jsonl --output_dir models/finetuned_tatqa_en --batch_size 32 --epochs 2 --max_samples 10000

# 3. TatQA微调后再评测
python run_encoder_eval.py --model_name models/finetuned_tatqa_en --eval_jsonl evaluate_mrr/tatqa_eval.jsonl --max_samples 1000 --print_examples 3

# 4. AlphaFin中文Encoder评测
python run_encoder_eval.py --model_name Langboat/mengzi-bert-base-fin --eval_jsonl evaluate_mrr/alphafin_eval.jsonl --max_samples 1000 --print_examples 3

# 5. AlphaFin中文Encoder微调
python finetune_encoder.py --model_name Langboat/mengzi-bert-base-fin --train_jsonl evaluate_mrr/alphafin_train_qc.jsonl --output_dir models/finetuned_alphafin_zh --batch_size 32 --epochs 2 --max_samples 10000

# 6. AlphaFin微调后再评测
python run_encoder_eval.py --model_name models/finetuned_alphafin_zh --eval_jsonl evaluate_mrr/alphafin_eval.jsonl --max_samples 1000 --print_examples 3
