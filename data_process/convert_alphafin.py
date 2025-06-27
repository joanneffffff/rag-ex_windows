"""
Convert AlphaFin dataset to JSON format
"""

import json
from pathlib import Path
from datasets import load_dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_dataset_to_json(dataset_name: str, output_path: str):
    """Convert HuggingFace dataset to JSON format"""
    try:
        logger.info(f"Loading dataset {dataset_name}")
        dataset = load_dataset(dataset_name)
        
        # Convert to list of dictionaries
        records = []
        
        # Process train split
        if 'train' in dataset:
            logger.info("Processing train split...")
            for item in dataset['train']:
                record = {
                    'instruction': item.get('instruction', ''),
                    'input': item.get('input', ''),
                    'output': item.get('output', ''),
                    'split': 'train'
                }
                records.append(record)
            logger.info(f"Processed {len(dataset['train'])} training examples")
        
        # Process test split
        if 'test' in dataset:
            logger.info("Processing test split...")
            for item in dataset['test']:
                record = {
                    'instruction': item.get('instruction', ''),
                    'input': item.get('input', ''),
                    'output': item.get('output', ''),
                    'split': 'test'
                }
                records.append(record)
            logger.info(f"Processed {len(dataset['test'])} test examples")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        logger.info(f"Saving {len(records)} records to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        
        logger.info("Conversion completed successfully")
        return records
    
    except Exception as e:
        logger.error(f"Error converting dataset: {str(e)}")
        raise

def process_alphafin_data():
    """Process AlphaFin dataset"""
    try:
        # Define paths
        dataset_name = "AlphaFin/AlphaFin-dataset-v1"
        output_path = "data/alphafin/data.json"
        
        # Convert dataset
        records = convert_dataset_to_json(dataset_name, output_path)
        
        # Print summary
        logger.info("\nDataset Summary:")
        logger.info(f"Total records: {len(records)}")
        train_count = sum(1 for r in records if r['split'] == 'train')
        test_count = sum(1 for r in records if r['split'] == 'test')
        logger.info(f"Training examples: {train_count}")
        logger.info(f"Test examples: {test_count}")
        
        # Create a sample file with fewer examples
        sample_size = min(1000, len(records))
        sample_path = "data/alphafin/sample_data.json"
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(records[:sample_size], f, ensure_ascii=False, indent=2)
        logger.info(f"\nCreated sample file with {sample_size} examples: {sample_path}")
        
    except Exception as e:
        logger.error(f"Error processing AlphaFin data: {str(e)}")
        raise

if __name__ == "__main__":
    process_alphafin_data()

input_path = 'data/alphafin/sample_data.json'
output_path = 'data/alphafin/alphafin_qca.json'

qca_data = []

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    for item in data:
        question = item.get('instruction', '').strip()
        context = item.get('input', '').strip()
        answer = item.get('output', '').strip()
        if question and context and answer:
            qca_data.append({
                'question': question,
                'context': context,
                'answer': answer
            })

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(qca_data, f, ensure_ascii=False, indent=2)

print(f"已完成Q-C-A格式清洗，输出到: {output_path}") 