"""
统一数据处理器 - 整合TatQA和AlphaFin的chunk逻辑
"""

import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

def extract_unit_from_paragraph(paragraphs: List[Dict]) -> str:
    """
    从段落中提取数值单位，如"in millions"或"in millions except per share amounts"
    """
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        # 查找模式如"dollars in millions"或"in millions"
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            # 提取匹配的单位词并标准化
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD"  # 假设金融数据为USD
    return ""

def table_to_natural_text(table_dict: Dict, caption: str = "", unit_info: str = "") -> str:
    """
    将TatQA表格转换为更自然的语言描述，适合LLM理解
    假设table_dict['table']中的第一个子列表是表头，其余是数据行
    """
    rows = table_dict.get("table", [])
    lines = []

    if caption:
        lines.append(f"Table Topic: {caption}.")

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]

    for i, row in enumerate(data_rows):
        # 跳过完全空的行
        if not row or all(str(v).strip() == "" for v in row):
            continue

        # 识别并处理类别标题行（如"Current assets"，后续单元格为空）
        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Table Category: {str(row[0]).strip()}.")
            continue

        # 处理核心数据行
        row_name = str(row[0]).strip().replace('.', '')  # 清理行名中的尾随句点

        data_descriptions = []
        for h_idx, v in enumerate(row):
            if h_idx == 0:  # 跳过第一个元素，因为它是行名
                continue
            
            header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}"  # 表头的后备方案
            value = str(v).strip()

            if value:  # 只处理非空值
                # 尝试添加单位和货币符号
                # 改进的正则表达式以稳健地匹配数字（包括负数、逗号分隔、括号负数）
                if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value): 
                    formatted_value = value.replace('$', '')  # 移除$以便重新添加
                    if unit_info:
                        # 在添加单位之前处理括号负数
                        if formatted_value.startswith('(') and formatted_value.endswith(')'):
                             formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                        else:
                             formatted_value = f"${formatted_value} {unit_info}"
                    else:
                        formatted_value = f"${formatted_value}"  # 如果没有单位信息，保留货币符号
                else:
                    formatted_value = value
                
                data_descriptions.append(f"{header} is {formatted_value}")  # 更自然的措辞

        # 组合行描述
        if row_name and data_descriptions:
            lines.append(f"Details for item {row_name}: {'; '.join(data_descriptions)}.")
        elif data_descriptions:  # 如果行名为空但有数据
            lines.append(f"Other data item: {'; '.join(data_descriptions)}.")
        elif row_name:  # 只有行名，没有数据（通常应该被空行检查捕获）
            lines.append(f"Data item: {row_name}.")

    return "\n".join(lines)

def convert_json_context_to_natural_language_chunks(json_str_context: str, company_name: str = "公司") -> List[str]:
    """
    将AlphaFin的JSON格式context转换为自然语言chunks
    """
    try:
        # 尝试解析JSON
        if isinstance(json_str_context, str):
            context_data = json.loads(json_str_context)
        else:
            context_data = json_str_context
        
        chunks = []
        
        # 处理不同的数据结构
        if isinstance(context_data, dict):
            # 处理单个文档
            chunks.extend(process_alphafin_document(context_data, company_name))
        elif isinstance(context_data, list):
            # 处理文档列表
            for doc in context_data:
                if isinstance(doc, dict):
                    chunks.extend(process_alphafin_document(doc, company_name))
        
        return chunks if chunks else [f"原始格式，解析失败: {json_str_context[:100]}..."]
        
    except json.JSONDecodeError:
        return [f"原始格式，无有效字典: {json_str_context[:100]}..."]
    except Exception as e:
        return [f"原始格式，无有效结构: {str(e)}"]

def process_alphafin_document(doc: Dict[str, Any], company_name: str) -> List[str]:
    """
    处理单个AlphaFin文档
    """
    chunks = []
    
    # 提取基本信息
    stock_name = doc.get('stock_name', company_name)
    stock_code = doc.get('stock_code', '')
    
    # 处理财务报表数据
    financial_data = doc.get('financial_data', {})
    if financial_data:
        chunks.append(f"{stock_name}({stock_code})财务报表信息:")
        
        # 处理资产负债表
        balance_sheet = financial_data.get('balance_sheet', {})
        if balance_sheet:
            chunks.append(f"资产负债表数据:")
            for key, value in balance_sheet.items():
                if value and str(value).strip():
                    chunks.append(f"  {key}: {value}")
        
        # 处理利润表
        income_statement = financial_data.get('income_statement', {})
        if income_statement:
            chunks.append(f"利润表数据:")
            for key, value in income_statement.items():
                if value and str(value).strip():
                    chunks.append(f"  {key}: {value}")
        
        # 处理现金流量表
        cash_flow = financial_data.get('cash_flow', {})
        if cash_flow:
            chunks.append(f"现金流量表数据:")
            for key, value in cash_flow.items():
                if value and str(value).strip():
                    chunks.append(f"  {key}: {value}")
    
    # 处理文本描述
    description = doc.get('description', '')
    if description:
        chunks.append(f"{stock_name}公司描述: {description}")
    
    # 处理其他字段
    for key, value in doc.items():
        if key not in ['financial_data', 'description', 'stock_name', 'stock_code']:
            if value and str(value).strip():
                chunks.append(f"{key}: {value}")
    
    return chunks

def process_tatqa_to_chunks(input_paths: List[str]) -> List[Dict[str, Any]]:
    """
    处理TatQA数据集，将paragraphs和tables转换为chunks
    """
    all_chunks = []
    global_doc_counter = 0

    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 确保data是一个列表，即使文件只包含一个文档
        if not isinstance(data, list):
            print(f"警告：文件 {Path(input_path).name} 的顶层结构不是列表，尝试作为单个文档处理。")
            data = [data]

        for i, item in tqdm(enumerate(data), desc=f"Processing docs from {Path(input_path).name} for corpus"):
            # 确保item是一个字典
            if not isinstance(item, dict):
                print(f"警告：文件 {Path(input_path).name} 中发现非字典项，跳过。项内容：{item}")
                continue
            
            doc_id = item.get("doc_id")
            if doc_id is None:
                # 如果doc_id不存在，生成一个唯一的ID
                doc_id = f"generated_doc_{global_doc_counter}_{Path(input_path).stem}_{i}"
                global_doc_counter += 1

            paragraphs = item.get("paragraphs", [])
            tables = item.get("tables", [])

            # 提取单位信息，通常在描述性段落中找到
            unit_info = extract_unit_from_paragraph(paragraphs)

            # 处理段落作为chunks
            for p_idx, para in enumerate(paragraphs):
                para_text = para.get("text", "") if isinstance(para, dict) else para
                if para_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"para_{p_idx}",
                        "text": para_text.strip(),
                        "source_type": "paragraph",
                        "language": "english"
                    })
            
            # 处理表格作为chunks
            for t_idx, table in enumerate(tables):
                table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                if table_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"table_{t_idx}",
                        "text": table_text.strip(),
                        "source_type": "table",
                        "language": "english"
                    })
    
    return all_chunks

def process_alphafin_to_chunks(input_path: str) -> List[Dict[str, Any]]:
    """
    处理AlphaFin数据集，将JSON格式转换为chunks
    """
    all_chunks = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 确保data是一个列表
    if not isinstance(data, list):
        data = [data]
    
    for i, item in tqdm(enumerate(data), desc=f"Processing AlphaFin data"):
        if not isinstance(item, dict):
            continue
        
        doc_id = item.get("doc_id", f"alphafin_doc_{i}")
        company_name = item.get("stock_name", "公司")
        
        # 转换JSON context为自然语言chunks
        context_str = item.get("context", "")
        if context_str:
            natural_chunks = convert_json_context_to_natural_language_chunks(context_str, company_name)
            
            for c_idx, chunk_text in enumerate(natural_chunks):
                if chunk_text.strip():
                    all_chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": f"chunk_{c_idx}",
                        "text": chunk_text.strip(),
                        "source_type": "financial_data",
                        "language": "chinese",
                        "company_name": company_name
                    })
    
    return all_chunks

def process_unified_data(
    tatqa_paths: Optional[List[str]] = None,
    alphafin_paths: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    统一处理TatQA和AlphaFin数据
    
    Args:
        tatqa_paths: TatQA数据文件路径列表
        alphafin_paths: AlphaFin数据文件路径列表
    
    Returns:
        包含中文和英文chunks的字典
    """
    result = {
        "chinese": [],
        "english": []
    }
    
    # 处理TatQA数据（英文）
    if tatqa_paths:
        print("处理TatQA数据...")
        for path in tatqa_paths:
            if Path(path).exists():
                chunks = process_tatqa_to_chunks([path])
                result["english"].extend(chunks)
                print(f"从 {path} 加载了 {len(chunks)} 个英文chunks")
            else:
                print(f"警告: TatQA文件不存在: {path}")
    
    # 处理AlphaFin数据（中文）
    if alphafin_paths:
        print("处理AlphaFin数据...")
        for path in alphafin_paths:
            if Path(path).exists():
                chunks = process_alphafin_to_chunks(path)
                result["chinese"].extend(chunks)
                print(f"从 {path} 加载了 {len(chunks)} 个中文chunks")
            else:
                print(f"警告: AlphaFin文件不存在: {path}")
    
    print(f"\n处理完成:")
    print(f"  中文chunks: {len(result['chinese'])}")
    print(f"  英文chunks: {len(result['english'])}")
    
    return result

def save_processed_chunks(chunks: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """
    保存处理后的chunks到文件
    
    Args:
        chunks: 包含中文和英文chunks的字典
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存中文chunks
    if chunks["chinese"]:
        chinese_file = output_path / "chinese_chunks.jsonl"
        with open(chinese_file, "w", encoding="utf-8") as f:
            for chunk in chunks["chinese"]:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"中文chunks已保存到: {chinese_file}")
    
    # 保存英文chunks
    if chunks["english"]:
        english_file = output_path / "english_chunks.jsonl"
        with open(english_file, "w", encoding="utf-8") as f:
            for chunk in chunks["english"]:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        print(f"英文chunks已保存到: {english_file}")

if __name__ == "__main__":
    # 示例用法
    tatqa_paths = [
        "data/tatqa_dataset_raw/tatqa_dataset_train.json"
    ]
    
    alphafin_paths = [
        "data/alphafin/alphafin_rag_ready_generated_cleaned.json"
    ]
    
    # 处理数据
    chunks = process_unified_data(tatqa_paths, alphafin_paths)
    
    # 保存结果
    save_processed_chunks(chunks, "data/processed") 