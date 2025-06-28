import json
import re
from pathlib import Path

# --- 配置你的输入文件路径和切片范围 ---
# input_file_path = Path("data/alphafin/alphafin_summarized_and_structured_qa_0627_b8_s50_fullsentence.json")
input_file_path = Path("data/alphafin/alphafin_summarized_and_structured_qa_0627_colab_backward.json")
start_idx_for_colab = 23524
end_idx_for_colab = 33524
# ----------------------------------------

print(f"正在统计输入文件 '{input_file_path.name}' 中索引 {start_idx_for_colab} 到 {end_idx_for_colab-1} 的切片里，预测类 Q&A 对的数量...")

instruction_keywords = [
    "下个月的涨跌", "进行预测", "下月涨跌概率", "预测下月涨跌", "这个股票下月的涨跌概率"
]
answer_keywords = [
    "下月最终收益结果", "上涨概率", "下跌概率", "涨", "跌"
]

instruction_pattern = re.compile(
    "|".join(r"\b" + re.escape(k) + r"\b" for k in instruction_keywords), re.IGNORECASE
)
answer_pattern = re.compile(
    "|".join(r"\b" + re.escape(k) + r"\b" for k in answer_keywords), re.IGNORECASE
)

count_in_slice = 0
try:
    with open(input_file_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    # 提取指定的切片
    slice_data = full_data[start_idx_for_colab:end_idx_for_colab]
    
    print(f"切片总记录数: {len(slice_data)}")

    for record in slice_data:
        instruction = record.get('original_instruction', '')
        output_answer = record.get('original_answer', record.get('output', '')) # 'output' for raw data, 'original_answer' for preprocessed data

        if instruction_pattern.search(instruction) and answer_pattern.search(output_answer):
            count_in_slice += 1
            
    print(f"\n在输入切片 (索引 {start_idx_for_colab} 到 {end_idx_for_colab-1}) 中，找到 {count_in_slice} 条预测类 Q&A 对。")

except FileNotFoundError:
    print(f"错误: 输入文件 '{input_file_path}' 不存在。")
except json.JSONDecodeError:
    print(f"错误: 无法解析 JSON 文件 '{input_file_path}'。请检查文件内容。")
except Exception as e:
    print(f"发生未知错误: {e}")
