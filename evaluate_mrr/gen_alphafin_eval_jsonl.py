import json
import os
import random

input_path = 'data/alphafin/alphafin_rag_ready_generated_cleaned.json'
output_dir = 'data/evaluate_mrr'
os.makedirs(output_dir, exist_ok=True)
eval_jsonl_path = os.path.join(output_dir, 'alphafin_eval.jsonl')
train_jsonl_path = os.path.join(output_dir, 'alphafin_train_qc.jsonl')
train_ratio = 0.8
seed = 42

# 1. 读取原始json
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 转为Q-C-A格式list
qca_list = []
for item in data:
    q = item.get('question', '')
    c = item.get('context', '')
    a = item.get('answer', '')
    qca_list.append({'query': q, 'context': c, 'answer': a})

# 3. 随机划分train/eval
random.seed(seed)
random.shuffle(qca_list)
n_train = int(len(qca_list) * train_ratio)
train_list = qca_list[:n_train]
eval_list = qca_list[n_train:]

# 4. 写入评测集Q-C-A
with open(eval_jsonl_path, 'w', encoding='utf-8') as f_eval:
    for item in eval_list:
        f_eval.write(json.dumps(item, ensure_ascii=False) + '\n')

# 5. 写入训练集Q-C
with open(train_jsonl_path, 'w', encoding='utf-8') as f_train:
    for item in train_list:
        f_train.write(json.dumps({
            'query': item['query'],
            'context': item['context']
        }, ensure_ascii=False) + '\n')

print(f"已生成评测集: {eval_jsonl_path}，共{len(eval_list)}条样本")
print(f"已生成训练集: {train_jsonl_path}，共{len(train_list)}条样本") 