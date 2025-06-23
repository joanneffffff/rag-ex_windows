import json
from pathlib import Path
from tqdm import tqdm
import re

def extract_unit_from_paragraph(paragraphs):
    """
    Attempts to extract numerical units from paragraphs, such as "in millions" or "in millions except per share amounts".
    """
    for para in paragraphs:
        text = para.get("text", "") if isinstance(para, dict) else para
        # Look for patterns like "dollars in millions" or "in millions"
        match = re.search(r'dollars in (millions|billions)|in (millions|billions)', text, re.IGNORECASE)
        if match:
            # Extract the matched unit word and standardize it
            unit = match.group(1) or match.group(2)
            if unit:
                return unit.lower().replace('s', '') + " USD" # Assuming USD for financial data
    return ""

def table_to_natural_text(table_dict, caption="", unit_info=""):
    """
    Converts a TatQA table into more conversational/descriptive natural language, suitable for LLMs.
    This function assumes the first sublist in table_dict['table'] is the header, and the rest are data rows.
    It attempts to organize more natural sentences, integrate unit information, and handle empty values and category header rows.
    """
    rows = table_dict.get("table", [])
    lines = []

    if caption:
        lines.append(f"Table Topic: {caption}.") # Added a period for sentence completion

    if not rows:
        return ""

    headers = rows[0]
    data_rows = rows[1:]

    for i, row in enumerate(data_rows):
        # Skip completely empty rows
        if not row or all(str(v).strip() == "" for v in row):
            continue

        # Identify and handle category header rows (e.g., "Current assets" where subsequent cells are empty)
        if len(row) > 1 and str(row[0]).strip() != "" and all(str(v).strip() == "" for v in row[1:]):
            lines.append(f"Table Category: {str(row[0]).strip()}.")
            continue

        # Process core data rows
        row_name = str(row[0]).strip().replace('.', '') # Clean up row name from trailing periods

        data_descriptions = []
        for h_idx, v in enumerate(row):
            if h_idx == 0: # Skip the first element as it's the row name
                continue
            
            header = headers[h_idx] if h_idx < len(headers) else f"Column {h_idx+1}" # Fallback for headers
            value = str(v).strip()

            if value: # Only process non-empty values
                # Attempt to add units and currency symbols
                # Improved regex to robustly match numbers (including negative, comma-separated, parenthetical negatives)
                if re.match(r'^-?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^\(\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)$', value): 
                    formatted_value = value.replace('$', '') # Remove $ for consistent re-adding
                    if unit_info:
                        # Handle parenthetical negative numbers before adding unit
                        if formatted_value.startswith('(') and formatted_value.endswith(')'):
                             formatted_value = f"(${formatted_value[1:-1]} {unit_info})"
                        else:
                             formatted_value = f"${formatted_value} {unit_info}"
                    else:
                        formatted_value = f"${formatted_value}" # If no unit info, keep currency symbol
                else:
                    formatted_value = value
                
                data_descriptions.append(f"{header} is {formatted_value}") # More natural phrasing

        # Combine row descriptions
        if row_name and data_descriptions:
            lines.append(f"Details for item {row_name}: {'; '.join(data_descriptions)}.")
        elif data_descriptions: # If row name is empty but there's data
            lines.append(f"Other data item: {'; '.join(data_descriptions)}.")
        elif row_name: # Only row name, no data (should be caught by empty row check usually)
            lines.append(f"Data item: {row_name}.")

    return "\n".join(lines)


def process_tatqa_to_qca(input_paths, output_path):
    """
    Processes TatQA dataset(s), integrating paragraph and table content into context,
    and generating Q-C-A formatted data. This function now accepts a list of input file paths.
    """
    all_data = []
    for input_path in input_paths:
        with open(input_path, "r", encoding="utf-8") as f:
            all_data.extend(json.load(f)) # Merge data from all input files

    with open(output_path, "w", encoding="utf-8") as fout:
        # Use the name of the output file for the tqdm description for clarity
        for item in tqdm(all_data, desc=f"Processing {Path(output_path).name}"):
            paragraphs = item.get("paragraphs", [])
            para_texts = []
            
            # Extract unit information, typically found in a descriptive paragraph
            unit_info = extract_unit_from_paragraph(paragraphs)

            for para in paragraphs:
                para_text = para.get("text", "") if isinstance(para, dict) else para
                if para_text:
                    para_texts.append(para_text)
            
            tables = item.get("tables", [])
            table_texts = []
            for table in tables:
                # Call the optimized table textualization function, passing unit info
                table_text = table_to_natural_text(table, table.get("caption", ""), unit_info)
                if table_text:
                    table_texts.append(table_text)
            
            # Combine context parts: paragraphs, then tables. Use double newlines for separation.
            context_parts = []
            if para_texts:
                context_parts.append("\n".join(para_texts))
            if table_texts:
                context_parts.append("\n\n".join(table_texts)) # Double newline between tables and after paragraphs
            
            context = "\n\n".join(context_parts).strip()
            
            qa_pairs = item.get("qa_pairs")
            if qa_pairs is None:
                qa_pairs = item.get("questions", []) # Fallback to "questions" key

            for qa in qa_pairs:
                question = qa.get("question", "").strip()
                answer = qa.get("answer", "")
                
                if isinstance(answer, list):
                    answer = "; ".join(str(a) for a in answer) # Use semicolon-space for better readability
                elif not isinstance(answer, str):
                    answer = str(answer) # Ensure all answers are strings
                
                if question and context and answer:
                    fout.write(json.dumps({
                        "question": question,
                        "context": context,
                        "answer": answer
                    }, ensure_ascii=False) + "\n")
    print(f"Generated Q-C-A data: {output_path}")

if __name__ == "__main__":
    # Define input and output paths
    base_raw_data_path = "data/tatqa_dataset_raw/"
    base_output_path = "evaluate_mrr/"

    # Merge train and dev datasets for training
    train_dev_inputs = [
        Path(base_raw_data_path) / "tatqa_dataset_train.json",
        Path(base_raw_data_path) / "tatqa_dataset_dev.json"
    ]
    process_tatqa_to_qca(
        input_paths=train_dev_inputs,
        output_path=Path(base_output_path) / "tatqa_train_qc.jsonl"
    )
    
    # Evaluation set remains unchanged
    eval_inputs = [
        Path(base_raw_data_path) / "tatqa_dataset_test_gold.json"
    ]
    process_tatqa_to_qca(
        input_paths=eval_inputs,
        output_path=Path(base_output_path) / "tatqa_eval.jsonl"
    )