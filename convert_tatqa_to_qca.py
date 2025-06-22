import json
from pathlib import Path
from tqdm import tqdm

def table_to_text(table_dict, caption=""):
    """将TatQA表格转为更口语化/描述性的自然语言，适配LLM"""
    headers = table_dict.get("header", [])
    rows = table_dict.get("table", [])
    lines = []
    if caption:
        lines.append(f"表格主题：{caption}")
    for i, row in enumerate(rows):
        row_desc = []
        for h, v in zip(headers, row):
            row_desc.append(f"{h}为{v}")
        lines.append("，".join(row_desc))
    return "。".join(lines)

def process_tatqa_to_qca(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(output_path, "w", encoding="utf-8") as fout:
        for item in tqdm(data, desc=f"Processing {Path(input_path).name}"):
            # 拼接所有段落
            paragraphs = item.get("paragraphs", [])
            para_texts = []
            for para in paragraphs:
                if isinstance(para, dict):
                    para_texts.append(para.get("text", ""))
                elif isinstance(para, str):
                    para_texts.append(para)
            # 拼接所有表格
            tables = item.get("tables", [])
            table_texts = []
            for table in tables:
                table_texts.append(table_to_text(table, table.get("caption", "")))
            # 合并为context
            context = "\n".join(para_texts + table_texts).strip()
            # 兼容test_gold结构，questions为list，每个有question/answer
            qa_pairs = item.get("qa_pairs")
            if qa_pairs is None:
                qa_pairs = item.get("questions", [])
            for qa in qa_pairs:
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "")
                if isinstance(answer, list):
                    answer = "；".join(str(a) for a in answer)
                if question and context and answer:
                    fout.write(json.dumps({
                        "question": question,
                        "context": context,
                        "answer": answer
                    }, ensure_ascii=False) + "\n")
    print(f"已生成Q-C-A格式数据：{output_path}")

if __name__ == "__main__":
    process_tatqa_to_qca(
        input_path="data/tatqa_dataset_raw/tatqa_dataset_train.json",
        output_path="data/tatqa_rag_ready_train.jsonl"
    )
    process_tatqa_to_qca(
        input_path="data/tatqa_dataset_raw/tatqa_dataset_test_gold.json",
        output_path="data/tatqa_rag_ready_test_gold.jsonl"
    ) 