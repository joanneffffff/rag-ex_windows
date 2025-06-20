import json
from datetime import datetime
from typing import List, Dict, Union, Optional
import pandas as pd
from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata

class FinancialDataProcessor:
    def __init__(self, cache_dir: str = "D:/AI/huggingface"):
        self.cache_dir = cache_dir
    
    def process_news(self, news_data: Dict) -> DocumentWithMetadata:
        """处理新闻数据为结构化文档格式"""
        timestamp = None
        if isinstance(news_data['input'], str) and '_' in news_data['input']:
            timestamp_str = news_data['input'].split('_')[0]
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                pass
        # 新模板格式
        content = "[金融新闻]\n"
        content += f"日期: {timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'N/A'}\n"
        content += f"标题: {news_data['input'].split('_', 1)[-1] if '_' in news_data['input'] else news_data['input']}\n"
        if news_data.get('output'):
            content += f"摘要: {news_data['output']}\n"
        return DocumentWithMetadata(
            content=content,
            metadata=DocumentMetadata(
                source="financial_news",
                created_at=str(timestamp) if timestamp else "",
                author=""
            )
        )
    
    def process_stock_data(self, stock_data: Dict) -> List[DocumentWithMetadata]:
        """处理股票问答数据为结构化文档格式"""
        documents = []
        if isinstance(stock_data.get('instruction'), list):
            for i, (inst, inp, out) in enumerate(zip(
                stock_data['instruction'],
                stock_data['input'],
                stock_data['output']
            )):
                stock_info = self._extract_stock_info(inp)
                content = "[股票问答]\n"
                content += f"股票代码: {stock_info['code'] if stock_info else '未知'}\n"
                content += f"日期: {stock_info['date'] if stock_info else '未知'}\n"
                # 提取问题
                question = inp.split('【问题】：')[-1] if '【问题】：' in inp else inp
                content += f"问题: {question}\n"
                # 提取答案
                answer = out.replace('【答案】：', '').strip() if isinstance(out, str) else out
                content += f"答案: {answer}\n"
                doc = DocumentWithMetadata(
                    content=content,
                    metadata=DocumentMetadata(
                        source=f"stock_data_{stock_info['code'] if stock_info else i}",
                        created_at=stock_info['date'] if stock_info else "",
                        author=""
                    )
                )
                documents.append(doc)
        return documents
    
    def _extract_stock_info(self, text: str) -> Optional[Dict[str, str]]:
        """Extract stock code and date from text"""
        import re
        
        # Try to find stock code (format: xxxxxx.SZ or xxxxxx.SH)
        code_match = re.search(r'(\d{6}\.(SZ|SH))', text)
        # Try to find date (format: YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', text)
        
        if code_match or date_match:
            return {
                'code': code_match.group(1) if code_match else None,
                'date': date_match.group(1) if date_match else None
            }
        return None
    
    def process_time_series(self, data: Dict[str, float], stock_code: str) -> DocumentWithMetadata:
        """处理时间序列数据为结构化文档格式"""
        series = pd.Series(data)
        stats = {
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'trend': '上升' if series.iloc[-1] > series.iloc[0] else '下降'
        }
        content = "[时间序列分析]\n"
        content += f"股票代码: {stock_code}\n"
        content += f"区间: {series.index[0]} 至 {series.index[-1]}\n"
        content += f"均值: {stats['mean']:.4f}\n"
        content += f"中位数: {stats['median']:.4f}\n"
        content += f"标准差: {stats['std']:.4f}\n"
        content += f"最小值: {stats['min']:.4f}\n"
        content += f"最大值: {stats['max']:.4f}\n"
        content += f"趋势: {stats['trend']}\n"
        content += "原始数据:\n"
        for date, value in data.items():
            content += f"  {date}: {value:.4f}\n"
        return DocumentWithMetadata(
            content=content,
            metadata=DocumentMetadata(
                source=f"time_series_{stock_code}",
                created_at=series.index[-1],
                author=""
            )
        ) 