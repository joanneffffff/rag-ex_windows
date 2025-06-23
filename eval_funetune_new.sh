#!/bin/bash

# 定义日志文件路径和时间戳
LOG_DIR="logs" # 日志文件存放目录
TIMESTAMP=$(date +"%Y%m%d_%H%M%S") # 获取当前时间戳，用于日志文件名
LOG_FILE="${LOG_DIR}/encoder_training_eval_${TIMESTAMP}.log" # 完整的日志文件路径

# 创建日志目录，如果不存在的话
mkdir -p "${LOG_DIR}"

# 将所有后续命令的输出重定向到日志文件，并在屏幕上显示
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "--- 开始执行编码器训练与评估流程 ---"
echo "日志将同时输出到控制台和文件: ${LOG_FILE}"
echo "执行时间: $(date)"
echo "------------------------------------"
echo ""

# 1. TatQA英文Encoder评测（GPU）
echo "--- 1. TatQA 英文Encoder预训练模型评测 ---"
python run_encoder_eval.py --model_name ProsusAI/finbert --eval_jsonl evaluate_mrr/tatqa_eval.jsonl --max_samples 1000 --print_examples 3
echo ""

# ---
# 2. TatQA英文Encoder微调（每个epoch输出Loss和MRR）
echo "--- 2. TatQA 英文Encoder模型微调 ---"
python finetune_encoder.py \
    --model_name ProsusAI/finbert \
    --train_jsonl evaluate_mrr/tatqa_train_qc.jsonl \
    --eval_jsonl evaluate_mrr/tatqa_eval.jsonl \
    --output_dir models/finetuned_tatqa_en \
    --batch_size 32 \
    --epochs 5 \
    --max_samples 10000 \
    --eval_steps 0
echo ""

# ---
# 3. TatQA微调后再评测
echo "--- 3. TatQA 微调后模型评测 ---"
python run_encoder_eval.py --model_name models/finetuned_tatqa_en --eval_jsonl evaluate_mrr/tatqa_eval.jsonl --max_samples 1000 --print_examples 3
echo ""

# # ---
# # 4. AlphaFin中文Encoder评测
# echo "--- 4. AlphaFin 中文Encoder预训练模型评测 ---"
# python run_encoder_eval.py --model_name Langboat/mengzi-bert-base-fin --eval_jsonl evaluate_mrr/alphafin_eval.jsonl --max_samples 1000 --print_examples 3
# echo ""

# # ---
# # 5. AlphaFin中文Encoder微调（每个epoch输出Loss和MRR）
# echo "--- 5. AlphaFin 中文Encoder模型微调 ---"
# python finetune_encoder.py \
#     --model_name Langboat/mengzi-bert-base-fin \
#     --train_jsonl evaluate_mrr/alphafin_train_qc.jsonl \
#     --eval_jsonl evaluate_mrr/alphafin_eval.jsonl \
#     --output_dir models/finetuned_alphafin_zh \
#     --batch_size 32 \
#     --epochs 5 \
#     --max_samples 10000 \
#     --eval_steps 0
# echo ""

# # ---
# # 6. AlphaFin微调后再评测
# echo "--- 6. AlphaFin 微调后模型评测 ---"
# python run_encoder_eval.py --model_name models/finetuned_alphafin_zh --eval_jsonl evaluate_mrr/alphafin_eval.jsonl --max_samples 1000 --print_examples 3
# echo ""

echo "------------------------------------"
echo "所有任务完成！"
echo "结束时间: $(date)"
