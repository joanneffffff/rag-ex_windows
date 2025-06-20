import json
import os
import numpy as np
import faiss
import pandas as pd
from xlm.utils.financial_data_loader import FinancialDataLoader
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

def test_financial_data():
    """Test processing financial data"""
    # Test data
    financial_data = [
        {
            'instruction': '我是一位股票分析师，我需要利用以下新闻信息来更好地完成金融分析，请你对下列新闻提取出可能对我有帮助的关键信息，形成更精简的新闻摘要。新闻具体内容如下：\n',
            'input': '2023-08-10 10:06:52_;莫斯科市长：俄罗斯的防空系统击落了两架朝莫斯科方向飞行的军用无人机。',
            'output': '莫斯科市长宣布，俄罗斯的防空系统成功击落了两架朝莫斯科方向飞行的军用无人机。'
        },
        {
            'instruction': ['', ''],
            'input': [
                "以下数据是海康威视002415.SZ时间为2018-04-24，你是一个股票分析师，我将给你提供一份每日指标数据，如下：{'TS股票代码': '002415.SZ', '交易日': '2018-04-24', '当日收盘价': 39.68, '市盈率（总市值/净利润， 亏损的PE为空）': 38.9059, '市盈率（TTM，亏损的PE为空）': 37.568, '市净率（总市值/净资产）': 11.36, '市销率': 8.7372, '市销率（TTM）': 8.2787, ' 股息率 （%）': 1.0082, '股息率（TTM）（%）': 1.0082, '总市值 （ 万元）': 36613809.2369}。【问题】：该股票的市盈率是多少？",
                "以下数据是福耀玻璃600660.SH时间为2023-05-24，你是一个股票分析师，我将给你提供一份每日指标，这是一份股票代码为600660.SH的关于换手率（自由流通股）的数据，如下：{'2023-04-24': 0.7026, '2023-04-25': 0.7476, '2023-04-26': 0.5744, '2023-04-27': 0.624, '2023-04-28': 0.5927, '2023-05-04': 0.49, '2023-05-05': 0.5479, '2023-05-08': 0.3729, '2023-05-09': 0.5345, '2023-05-10': 0.7582, '2023-05-11': 0.6437, '2023-05-12': 0.3559, '2023-05-15': 0.818, '2023-05-16': 0.5656, '2023-05-17': 0.6324, '2023-05-18': 0.5146, '2023-05-19': 0.3669, '2023-05-22': 0.4074, '2023-05-23': 0.4688, '2023-05-24': 0.3368}。【问题】：这段时间内换手率低于0.5的天数有多少天？"
            ],
            'output': ['【答案】：该股票的市盈率为38.9059。', '【答案】：这段时间内换手率低于0.5的天数有4天。']
        }
    ]
    
    # Save test data
    with open('test_financial_data.json', 'w', encoding='utf-8') as f:
        json.dump(financial_data, f, ensure_ascii=False, indent=2)
    
    # Process data
    loader = FinancialDataLoader()
    documents = loader.load_data('test_financial_data.json')
    
    # Print results
    print("\n=== Financial Data Processing Results ===")
    for doc in documents:
        print("\nDocument:")
        print(f"Source: {doc.metadata.source}")
        print(f"Created at: {doc.metadata.created_at}")
        print("Content:")
        print(doc.content)
        print("-" * 50)

class TatQAProcessor:
    """Processor for TatQA dataset with tables and paragraphs"""
    def __init__(self):
        self.cache_dir = "D:/AI/huggingface/tatqa"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize FAISS index
        self.dimension = 384  # default dimension for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # For storing document mappings
        self.doc_store = []
    
    def process_chunk(self, chunk_data: list) -> list[DocumentWithMetadata]:
        """Process a chunk of TatQA data"""
        documents = []
        
        for item in chunk_data:
            # Store original item for context
            item_docs = []
            
            # Process tables
            if 'table' in item:
                table_doc = self._process_table(item['table'])
                if table_doc:
                    item_docs.append(table_doc)
            
            # Process paragraphs
            if 'paragraphs' in item:
                for para in item['paragraphs']:
                    para_doc = self._process_paragraph(para)
                    if para_doc:
                        item_docs.append(para_doc)
            
            # Process QA pairs with context
            if 'qa_pairs' in item:
                for qa in item['qa_pairs']:
                    qa_doc = self._process_qa_pair(qa, item_docs)
                    if qa_doc:
                        documents.append(qa_doc)
            
            documents.extend(item_docs)
        
        return documents
    
    def _process_table(self, table_data: dict) -> DocumentWithMetadata:
        """处理表格数据为结构化中文格式"""
        try:
            df = pd.DataFrame(table_data)
            num_rows, num_cols = df.shape
            col_types = df.dtypes.to_dict()
            table_str = df.to_string()
            content = "[表格分析]\n"
            content += f"维度: {num_rows}行 × {num_cols}列\n"
            content += "列类型:\n"
            for col, dtype in col_types.items():
                content += f"- {col}: {dtype}\n"
            content += "\n表格内容:\n"
            content += table_str
            return DocumentWithMetadata(
                content=content,
                metadata=DocumentMetadata(
                    source="tatqa_table",
                    created_at="",
                    author=""
                )
            )
        except Exception as e:
            print(f"Error processing table: {e}")
            return None
    
    def _process_paragraph(self, para_data: str) -> DocumentWithMetadata:
        """处理段落数据为结构化中文格式"""
        try:
            sentences = para_data.split('.')
            word_count = len(para_data.split())
            content = "[段落分析]\n"
            content += f"词数: {word_count}\n"
            content += f"句数: {len(sentences)}\n"
            content += "\n段落内容:\n"
            content += para_data
            return DocumentWithMetadata(
                content=content,
                metadata=DocumentMetadata(
                    source="tatqa_paragraph",
                    created_at="",
                    author=""
                )
            )
        except Exception as e:
            print(f"Error processing paragraph: {e}")
            return None
    
    def _process_qa_pair(self, qa: dict, context_docs: list) -> DocumentWithMetadata:
        """处理单个问答对为结构化中文格式"""
        try:
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            answer_type = qa.get('answer_type', '')
            answer_scale = qa.get('scale', '')
            derivation = qa.get('derivation', [])
            content = "[问答对]\n"
            content += f"问题: {question}\n"
            content += f"答案: {answer}\n"
            content += f"答案类型: {answer_type}\n"
            if answer_scale:
                content += f"量纲: {answer_scale}\n"
            if derivation:
                content += "推导步骤:\n"
                for step in derivation:
                    content += f"- {step}\n"
            content += "\n上下文:\n"
            for doc in context_docs:
                content += f"\n{doc.content}\n"
                content += "-" * 40 + "\n"
            return DocumentWithMetadata(
                content=content,
                metadata=DocumentMetadata(
                    source="tatqa_qa_pair",
                    created_at="",
                    author=""
                )
            )
        except Exception as e:
            print(f"Error processing QA pair: {e}")
            return None

def test_tatqa_data():
    """Test processing TatQA data"""
    print("\n=== Testing TatQA Data Processing ===")
    
    # Load sample data
    try:
        with open('test/data/sample_tatqa.json', 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    except Exception as e:
        print(f"Error loading sample data: {e}")
        return
    
    # Initialize processor
    processor = TatQAProcessor()
    
    # Process data
    try:
        documents = processor.process_chunk(sample_data)
        
        # Print processing results
        print(f"\nProcessed {len(documents)} documents")
        
        # Count document types
        doc_types = {}
        for doc in documents:
            doc_type = doc.metadata.source
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print("\nDocument type distribution:")
        for doc_type, count in doc_types.items():
            print(f"- {doc_type}: {count}")
        
        # Print sample of each document type
        print("\nSample documents:")
        printed_types = set()
        for doc in documents:
            if doc.metadata.source not in printed_types:
                print(f"\n{doc.metadata.source}:")
                print("-" * 40)
                print(doc.content[:500] + "..." if len(doc.content) > 500 else doc.content)
                print("-" * 40)
                printed_types.add(doc.metadata.source)
        
        return documents
    except Exception as e:
        print(f"Error processing TatQA data: {e}")
        return None

if __name__ == "__main__":
    # Test financial data processing
    test_financial_data()
    
    # Test TatQA data processing
    test_tatqa_data() 