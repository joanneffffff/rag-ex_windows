import json
from tqdm import tqdm
from pathlib import Path

def finalize_alphafin_data():
    """
    Creates a final, robust dataset for RAG.
    Each document will contain the full article content for embedding,
    and the human-written summary and a generic question in the metadata.
    """
    input_file = Path('data/alphafin/alphafin_cleaned.json')
    output_file = Path('data/alphafin/alphafin_final.json')

    print(f"Loading cleaned data from {input_file}...")
    if not input_file.exists():
        print(f"Fatal: Input file not found. Please run `process_alphafin.py` first.")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    final_data = []
    for record in tqdm(cleaned_data, desc="Finalizing Data"):
        context = record.get('input', '').strip()
        answer = record.get('output', '').strip()

        if not context:
            continue
        
        # The document passed to the retriever will just be the context.
        # The question and answer are stored as metadata for later use.
        final_data.append({
            'content': context,
            'metadata': {
                'question': '请总结这篇财经新闻并提取关键信息。',
                'answer': answer
            }
        })

    print(f"Saving final data to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)
        
    print(f"Processing complete. {len(final_data)} records saved.")

if __name__ == '__main__':
    finalize_alphafin_data() 