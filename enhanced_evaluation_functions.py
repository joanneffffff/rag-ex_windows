import json
import hashlib
import numpy as np
from typing import List, Dict, Any
from xlm.dto.dto import DocumentWithMetadata

def calculate_content_hash(text: str) -> str:
    """计算文本内容的哈希值"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def find_correct_document_rank_enhanced(
    context: str, 
    retrieved_docs: List[DocumentWithMetadata], 
    sample: Dict[str, Any],
    encoder=None
) -> int:
    """
    增强版：使用多种策略查找正确答案的排名，支持relevant_doc_ids
    
    Args:
        context: 正确答案的context
        retrieved_docs: 检索到的文档列表
        sample: 评估样本（可能包含relevant_doc_ids字段）
        encoder: 编码器（用于相似度计算）
    
    Returns:
        找到的排名，0表示未找到
    """
    if not context or not retrieved_docs:
        return 0
    
    # 策略0: relevant_doc_ids匹配（最严格）- 适用于英文数据
    relevant_doc_ids = sample.get('relevant_doc_ids', [])
    if relevant_doc_ids:
        for rank, doc in enumerate(retrieved_docs, 1):
            # 尝试从文档内容中提取doc_id（如果是JSON格式）
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_id = doc_data.get('doc_id') or doc_data.get('id')
                    if doc_id in relevant_doc_ids:
                        return rank
            except:
                pass
            
            # 尝试从元数据中获取doc_id
            doc_id = getattr(doc, 'id', None) or getattr(doc.metadata, 'id', None) or getattr(doc.metadata, 'doc_id', None)
            if doc_id in relevant_doc_ids:
                return rank
    
    # 策略1: ID匹配（适用于中文数据）
    correct_doc_id = sample.get('doc_id') or sample.get('id') or sample.get('document_id')
    if correct_doc_id:
        for rank, doc in enumerate(retrieved_docs, 1):
            # 尝试从文档内容中提取doc_id（如果是JSON格式）
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    doc_id = doc_data.get('doc_id') or doc_data.get('id')
                    if doc_id == correct_doc_id:
                        return rank
            except:
                pass
            
            # 尝试从元数据中获取doc_id
            doc_id = getattr(doc, 'id', None) or getattr(doc.metadata, 'id', None) or getattr(doc.metadata, 'doc_id', None)
            if doc_id == correct_doc_id:
                return rank
    
    # 策略2: 内容哈希匹配
    context_hash = calculate_content_hash(context.strip())
    for rank, doc in enumerate(retrieved_docs, 1):
        # 处理JSON格式的文档内容
        doc_content = doc.content
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                # 提取context字段
                doc_context = doc_data.get('context', '')
                if doc_context:
                    doc_content = doc_context
        except:
            pass
        
        doc_hash = calculate_content_hash(doc_content.strip())
        if doc_hash == context_hash:
            return rank
    
    # 策略3: 精确文本匹配（改进版）
    context_clean = context.strip().lower()
    for rank, doc in enumerate(retrieved_docs, 1):
        # 处理JSON格式的文档内容
        doc_content = doc.content
        try:
            if doc.content.startswith('{'):
                doc_data = json.loads(doc.content)
                # 提取context字段
                doc_context = doc_data.get('context', '')
                if doc_context:
                    doc_content = doc_context
        except:
            pass
        
        doc_content_clean = doc_content.strip().lower()
        
        # 检查context是否包含在文档中，或文档是否包含在context中
        if (context_clean in doc_content_clean or 
            doc_content_clean in context_clean or
            context_clean == doc_content_clean):
            return rank
    
    # 策略4: 模糊文本匹配（使用关键词）
    context_words = set(context_clean.split())
    if len(context_words) > 3:  # 至少需要3个词
        for rank, doc in enumerate(retrieved_docs, 1):
            # 处理JSON格式的文档内容
            doc_content = doc.content
            try:
                if doc.content.startswith('{'):
                    doc_data = json.loads(doc.content)
                    # 提取context字段
                    doc_context = doc_data.get('context', '')
                    if doc_context:
                        doc_content = doc_context
            except:
                pass
            
            doc_content_clean = doc_content.strip().lower()
            doc_words = set(doc_content_clean.split())
            
            # 计算词汇重叠度
            overlap = len(context_words.intersection(doc_words))
            overlap_ratio = overlap / len(context_words)
            
            # 如果重叠度超过70%，认为匹配
            if overlap_ratio > 0.7:
                return rank
    
    # 策略5: 相似度匹配（如果有编码器）
    if encoder and len(context) > 10:  # 确保context足够长
        try:
            context_embedding = encoder.encode([context])
            
            # 准备文档内容用于编码
            doc_contents = []
            for doc in retrieved_docs:
                doc_content = doc.content
                try:
                    if doc.content.startswith('{'):
                        doc_data = json.loads(doc.content)
                        # 提取context字段
                        doc_context = doc_data.get('context', '')
                        if doc_context:
                            doc_content = doc_context
                except:
                    pass
                doc_contents.append(doc_content)
            
            doc_embeddings = encoder.encode(doc_contents)
            
            # 计算余弦相似度
            similarities = []
            for doc_emb in doc_embeddings:
                cos_sim = np.dot(context_embedding[0], doc_emb) / (
                    np.linalg.norm(context_embedding[0]) * np.linalg.norm(doc_emb)
                )
                similarities.append(cos_sim)
            
            # 找到最高相似度的文档
            max_sim_idx = int(np.argmax(similarities))
            max_similarity = similarities[max_sim_idx]
            
            # 如果相似度超过阈值，认为匹配
            if max_similarity > 0.8:
                return max_sim_idx + 1
                
        except Exception as e:
            print(f"相似度计算失败: {e}")
    
    return 0

def evaluate_with_enhanced_matching(
    eval_data: List[Dict[str, Any]],
    retriever,
    encoder,
    language: str = "chinese",
    dataset_name: str = "unknown",
    max_samples: int = None
) -> Dict[str, float]:
    """
    使用增强版匹配函数进行评估
    
    Args:
        eval_data: 评估数据列表
        retriever: 检索器
        encoder: 编码器
        language: 语言类型
        dataset_name: 数据集名称
        max_samples: 最大样本数
    
    Returns:
        评估结果字典
    """
    if max_samples:
        eval_data = eval_data[:max_samples]
    
    print(f"评估 {dataset_name} ({language}) - {len(eval_data)} 个样本")
    
    ranks = []
    total_queries = len(eval_data)
    
    for i, sample in enumerate(eval_data):
        if i % 100 == 0:
            print(f"  处理进度: {i}/{total_queries}")
        
        query = sample.get('query', '')
        context = sample.get('context', '')
        
        if not query or not context:
            continue
        
        try:
            # 检索文档
            retrieved_docs = retriever.retrieve(query, top_k=20)
            
            # 使用增强版函数查找排名
            found_rank = find_correct_document_rank_enhanced(
                context=context,
                retrieved_docs=retrieved_docs,
                sample=sample,
                encoder=encoder
            )
            
            ranks.append(found_rank)
            
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
            ranks.append(0)
    
    # 计算指标
    valid_ranks = [r for r in ranks if r > 0]
    mrr = np.mean([1.0 / r for r in valid_ranks]) if valid_ranks else 0.0
    hit_at_1 = sum(1 for r in ranks if r == 1) / len(ranks) if ranks else 0.0
    hit_at_5 = sum(1 for r in ranks if r <= 5) / len(ranks) if ranks else 0.0
    hit_at_10 = sum(1 for r in ranks if r <= 10) / len(ranks) if ranks else 0.0
    
    results = {
        'mrr': mrr,
        'hit_at_1': hit_at_1,
        'hit_at_5': hit_at_5,
        'hit_at_10': hit_at_10,
        'total_queries': len(ranks),
        'found_queries': len(valid_ranks)
    }
    
    print(f"  结果: MRR={mrr:.4f}, Hit@1={hit_at_1:.4f}, Hit@5={hit_at_5:.4f}, Hit@10={hit_at_10:.4f}")
    print(f"  找到正确答案: {len(valid_ranks)}/{len(ranks)}")
    
    return results
