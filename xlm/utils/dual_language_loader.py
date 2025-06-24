"""
Dual language data loader for Chinese and English documents
"""

import json
from typing import List, Dict, Tuple
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm

from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class DualLanguageLoader:
    def __init__(self):
        """初始化双语言数据加载器"""
        pass
    
    def detect_language(self, text: str) -> str:
        """
        检测文本语言
        
        Args:
            text: 文本内容
            
        Returns:
            语言标识 ('chinese' 或 'english')
        """
        try:
            lang = detect(text)
            if lang.startswith('zh'):
                return 'chinese'
            else:
                return 'english'
        except LangDetectException:
            # 默认返回英文
            return 'english'
    
    def load_alphafin_data(self, file_path: str) -> List[DocumentWithMetadata]:
        """
        加载AlphaFin中文数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            中文文档列表
        """
        print(f"加载AlphaFin中文数据: {file_path}")
        
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for idx, item in enumerate(tqdm(data, desc="处理AlphaFin数据")):
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            context = item.get('context', '').strip()
            stock_name = item.get('stock_name', '公司')
            
            if question and answer and context:
                # 创建文档元数据
                metadata = DocumentMetadata(
                    doc_id=f"alphafin_{idx}",
                    source="alphafin",
                    language="chinese",
                    stock_name=stock_name,
                    question=question,
                    answer=answer
                )
                
                # 创建文档对象
                document = DocumentWithMetadata(
                    content=context,
                    metadata=metadata
                )
                
                documents.append(document)
        
        print(f"加载了 {len(documents)} 个中文文档")
        return documents
    
    def load_tatqa_data(self, file_path: str) -> List[DocumentWithMetadata]:
        """
        加载TatQA英文数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            英文文档列表
        """
        print(f"加载TatQA英文数据: {file_path}")
        
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for idx, item in enumerate(tqdm(data, desc="处理TatQA数据")):
            question = item.get('question', '').strip()
            answer = item.get('answer', '').strip()
            context = item.get('context', '').strip()
            
            if question and answer and context:
                # 创建文档元数据
                metadata = DocumentMetadata(
                    doc_id=f"tatqa_{idx}",
                    source="tatqa",
                    language="english",
                    question=question,
                    answer=answer
                )
                
                # 创建文档对象
                document = DocumentWithMetadata(
                    content=context,
                    metadata=metadata
                )
                
                documents.append(document)
        
        print(f"加载了 {len(documents)} 个英文文档")
        return documents
    
    def load_jsonl_data(self, file_path: str, language: str = None) -> List[DocumentWithMetadata]:
        """
        加载JSONL格式数据
        
        Args:
            file_path: 数据文件路径
            language: 指定语言，如果为None则自动检测
            
        Returns:
            文档列表
        """
        print(f"加载JSONL数据: {file_path}")
        
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, desc="处理JSONL数据")):
                try:
                    item = json.loads(line.strip())
                    query = item.get('query', '').strip()
                    context = item.get('context', '').strip()
                    answer = item.get('answer', '').strip()
                    
                    if query and context:
                        # 检测语言
                        if language is None:
                            detected_lang = self.detect_language(query)
                        else:
                            detected_lang = language
                        
                        # 创建文档元数据
                        metadata = DocumentMetadata(
                            doc_id=f"jsonl_{idx}",
                            source="jsonl",
                            language=detected_lang,
                            question=query,
                            answer=answer
                        )
                        
                        # 创建文档对象
                        document = DocumentWithMetadata(
                            content=context,
                            metadata=metadata
                        )
                        
                        documents.append(document)
                        
                except json.JSONDecodeError:
                    print(f"警告: 跳过无效的JSON行 {idx+1}")
                    continue
        
        print(f"加载了 {len(documents)} 个文档")
        return documents
    
    def separate_documents_by_language(self, documents: List[DocumentWithMetadata]) -> Tuple[List[DocumentWithMetadata], List[DocumentWithMetadata]]:
        """
        根据语言分离文档
        
        Args:
            documents: 文档列表
            
        Returns:
            (中文文档列表, 英文文档列表)
        """
        chinese_docs = []
        english_docs = []
        
        for doc in documents:
            if doc.metadata.language == 'chinese':
                chinese_docs.append(doc)
            else:
                english_docs.append(doc)
        
        print(f"分离结果: {len(chinese_docs)} 个中文文档, {len(english_docs)} 个英文文档")
        return chinese_docs, english_docs
    
    def load_dual_language_data(
        self,
        chinese_data_path: str = None,
        english_data_path: str = None,
        jsonl_data_path: str = None
    ) -> Tuple[List[DocumentWithMetadata], List[DocumentWithMetadata]]:
        """
        加载双语言数据
        
        Args:
            chinese_data_path: 中文数据路径
            english_data_path: 英文数据路径
            jsonl_data_path: JSONL数据路径
            
        Returns:
            (中文文档列表, 英文文档列表)
        """
        chinese_docs = []
        english_docs = []
        
        # 加载中文数据
        if chinese_data_path:
            if chinese_data_path.endswith('.json'):
                chinese_docs.extend(self.load_alphafin_data(chinese_data_path))
            elif chinese_data_path.endswith('.jsonl'):
                chinese_docs.extend(self.load_jsonl_data(chinese_data_path, 'chinese'))
        
        # 加载英文数据
        if english_data_path:
            if english_data_path.endswith('.json'):
                english_docs.extend(self.load_tatqa_data(english_data_path))
            elif english_data_path.endswith('.jsonl'):
                english_docs.extend(self.load_jsonl_data(english_data_path, 'english'))
        
        # 加载JSONL数据（自动检测语言）
        if jsonl_data_path:
            all_docs = self.load_jsonl_data(jsonl_data_path)
            chinese_temp, english_temp = self.separate_documents_by_language(all_docs)
            chinese_docs.extend(chinese_temp)
            english_docs.extend(english_temp)
        
        print(f"总计: {len(chinese_docs)} 个中文文档, {len(english_docs)} 个英文文档")
        return chinese_docs, english_docs 