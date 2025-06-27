import json
import os
import random

import re
from pathlib import Path
from tqdm import tqdm # 导入 tqdm 用于显示进度条

def replace_context_prompt_with_regex(file_path: str):
    """
    使用正则表达式从 JSON 文件的 'context' 字段中移除从特定开头到特定结尾的引导性文本。
    """
    path = Path(file_path)
    
    if not path.exists():
        print(f"错误：文件 '{file_path}' 不存在。")
        return

    print(f"正在读取文件：{file_path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是有效的 JSON 格式。")
        return
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return

    initial_count = len(data)
    print(f"原始条目数量：{initial_count}")

    # 定义更精确的正则表达式模式
    # r"你是一个股票分析师.*?如下\s*"
    # ^ 匹配字符串开头
    # .*? 匹配任意字符（包括换行），非贪婪模式
    # 如下 匹配字面量“如下”
    # \s* 匹配“如下”之后可能存在的空白字符（包括换行符），将其一同清除
    # re.DOTALL 使得 . 可以匹配换行符
    # re.IGNORECASE 使得匹配不区分大小写（如果需要的话，但通常中文不需要）
    regex_pattern = re.compile(
        r"你是一个股票分析师.*?如下\s*", 
        re.DOTALL
    )

    modified_count = 0
    new_data = []

    for item in tqdm(data, desc="正在处理 context 字段"):
        original_context = item.get('context', '')
        
        # 使用 sub 方法进行替换
        # 如果匹配到，则替换为空字符串
        # .strip() 会移除替换后可能留下的多余的、不属于原prompt的空白字符
        cleaned_context = regex_pattern.sub("", original_context).strip() 

        if cleaned_context != original_context:
            modified_count += 1
        
        item['context'] = cleaned_context
        new_data.append(item)

    print(f"已修改 {modified_count} 个条目。")
    print(f"总条目数量：{len(new_data)}")

    # 将修改后的数据写回文件
    print(f"正在将数据写回文件：{file_path}")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print("数据写入成功。")
    except Exception as e:
        print(f"写入文件时发生错误：{e}")

# json_file_path = 'data/alphafin/alphafin_rag_ready_generated_cleaned.json'
# replace_context_prompt_with_regex(json_file_path)
# print("remove_specific_questions done")

input_path = 'data/alphafin/alphafin_rag_ready_generated_cleaned.json'
output_dir = 'evaluate_mrr'
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