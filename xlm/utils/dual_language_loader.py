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
        加载AlphaFin中文数据，字段映射如下：
        - question: generated_question
        - answer: original_answer
        - context: original_context
        - summary: summary（用于FAISS索引）
        """
        print(f"加载AlphaFin中文数据: {file_path}")
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for idx, item in enumerate(tqdm(data, desc="处理AlphaFin数据")):
                question = item.get('generated_question', '').strip()
                answer = item.get('original_answer', '').strip()
                context = item.get('original_context', '').strip()
                summary = item.get('summary', '').strip()
                company_name = item.get('company_name', '')
                stock_code = item.get('stock_code', '')
                report_date = item.get('report_date', '')
                # summary为空则跳过
                if not summary:
                    continue
                # 组装元数据
                metadata = DocumentMetadata(
                    source="alphafin",
                    language="chinese",
                    doc_id=f"alphafin_{idx}",
                    question=question,
                    answer=answer,
                    company_name=company_name,
                    stock_code=stock_code,
                    report_date=report_date,
                    summary=summary
                )
                # content字段为context，summary单独存入metadata
                document = DocumentWithMetadata(
                    content=context,
                    metadata=metadata
                )
                documents.append(document)
            print(f"加载了 {len(documents)} 个AlphaFin文档（summary不为空）")
            return documents
        except Exception as e:
            print(f"错误: 加载AlphaFin数据失败: {e}")
            return []

    def get_alphafin_summaries(self, documents: List[DocumentWithMetadata]) -> List[str]:
        """
        获取所有AlphaFin文档的summary字段列表，用于FAISS索引
        """
        return [doc.metadata.summary for doc in documents if hasattr(doc.metadata, 'summary') and doc.metadata.summary]
    
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
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for idx, item in enumerate(tqdm(data, desc="处理TatQA数据")):
                question = item.get('question', '').strip()
                answer = item.get('answer', '').strip()
                context = item.get('context', '').strip()
                
                if question and answer and context:
                    # 创建文档元数据
                    metadata = DocumentMetadata(
                        source="tatqa",
                        language="english",
                        doc_id=f"tatqa_{idx}",
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
            
        except Exception as e:
            print(f"错误: 加载TatQA数据失败: {e}")
            return []
    
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
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(tqdm(f, desc="处理JSONL数据")):
                    try:
                        item = json.loads(line.strip())
                        
                        # 提取必要字段
                        question = item.get('question', '').strip()
                        answer = item.get('answer', '').strip()
                        context = item.get('context', '').strip()
                        
                        if question and answer and context:
                            # 检测语言
                            if language is None:
                                detected_lang = self.detect_language(question)
                            else:
                                detected_lang = language
                            
                            # 创建文档元数据
                            metadata = DocumentMetadata(
                                source="jsonl",
                                language=detected_lang,
                                doc_id=f"jsonl_{idx}",
                                question=question,
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
            
            print(f"加载了 {len(documents)} 个JSONL文档")
            return documents
            
        except Exception as e:
            print(f"错误: 加载JSONL数据失败: {e}")
            return []
    
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
    
    def load_tatqa_context_only(self, file_path: str) -> List[DocumentWithMetadata]:
        """
        只加载TAT-QA英文数据的context字段（用于FAISS索引/检索库）
        """
        print(f"加载TAT-QA context-only数据: {file_path}")
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        context = item.get('context', '').strip()
                        if context:
                            metadata = DocumentMetadata(
                                source="tatqa",
                                created_at="",
                                author="",
                                language="english",
                                doc_id=f"tatqa_{idx}"
                            )
                            document = DocumentWithMetadata(
                                content=context,
                                metadata=metadata
                            )
                            documents.append(document)
                    except Exception as e:
                        print(f"跳过第{idx+1}行，原因: {e}")
            print(f"加载了 {len(documents)} 个TAT-QA context文档")
            return documents
        except Exception as e:
            print(f"错误: 加载TAT-QA context数据失败: {e}")
            return []

    def load_dual_language_data(
        self,
        chinese_data_path: str = None,
        english_data_path: str = None,
        jsonl_data_path: str = None
    ) -> Tuple[List[DocumentWithMetadata], List[DocumentWithMetadata]]:
        """
        加载双语言数据（英文优先用context-only方法）
        """
        chinese_docs = []
        english_docs = []
        # 加载中文数据
        if chinese_data_path:
            if chinese_data_path.endswith('.json'):
                chinese_docs.extend(self.load_alphafin_data(chinese_data_path))
            elif chinese_data_path.endswith('.jsonl'):
                chinese_docs.extend(self.load_jsonl_data(chinese_data_path, 'chinese'))
        # 加载英文数据（优先用context-only）
        if english_data_path:
            if english_data_path.endswith('.json'):
                english_docs.extend(self.load_tatqa_context_only(english_data_path))
            elif english_data_path.endswith('.jsonl'):
                english_docs.extend(self.load_tatqa_context_only(english_data_path))
        # 加载JSONL数据（自动检测语言）
        if jsonl_data_path:
            all_docs = self.load_jsonl_data(jsonl_data_path)
            chinese_temp, english_temp = self.separate_documents_by_language(all_docs)
            chinese_docs.extend(chinese_temp)
            english_docs.extend(english_temp)
        print(f"总计: {len(chinese_docs)} 个中文文档, {len(english_docs)} 个英文文档")
        return chinese_docs, english_docs
    
    def load_context_only_data(self, file_path: str, language: str = None) -> List[DocumentWithMetadata]:
        """
        只加载context字段的优化数据加载方法（用于RAG知识库）
        
        Args:
            file_path: 数据文件路径
            language: 指定语言，如果为None则自动检测
            
        Returns:
            文档列表（只包含context内容，元数据简化）
        """
        print(f"加载纯context数据: {file_path}")
        
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(tqdm(f, desc="处理context数据")):
                    try:
                        item = json.loads(line.strip())
                        
                        # 确保context字段存在且为字符串
                        context = item.get('context', '')
                        if isinstance(context, str):
                            context = context.strip()
                        else:
                            # 如果context不是字符串，尝试转换或跳过
                            print(f"警告: 第{idx}行的context不是字符串类型: {type(context)}")
                            if isinstance(context, dict):
                                # 如果是字典，尝试提取其中的context字段
                                context = context.get('context', str(context))
                            else:
                                context = str(context)
                        
                        if context:  # 只检查context是否存在且不为空
                            # 检测语言（使用context内容而不是query）
                            if language is None:
                                detected_lang = self.detect_language(context)
                            else:
                                detected_lang = language
                            
                            # 创建简化的文档元数据（只保留必要字段）
                            metadata = DocumentMetadata(
                                source="context_only",
                                language=detected_lang,
                                doc_id=f"context_{idx}"
                            )
                            
                            # 创建文档对象，确保content字段是字符串
                            doc = DocumentWithMetadata(
                                content=context,
                                metadata=metadata
                            )
                            documents.append(doc)
                        else:
                            print(f"警告: 第{idx}行的context为空，跳过")
                            
                    except json.JSONDecodeError as e:
                        print(f"警告: 第{idx}行JSON解析失败: {e}")
                        continue
                    except Exception as e:
                        print(f"警告: 第{idx}行处理失败: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"错误: 文件不存在: {file_path}")
            return []
        except Exception as e:
            print(f"错误: 读取文件失败: {e}")
            return []
        
        print(f"✅ 成功加载 {len(documents)} 个context文档")
        return documents 