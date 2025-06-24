from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """文档元数据"""
    source: str = ""
    created_at: str = ""
    author: str = ""
    language: str = ""


class DocumentWithMetadata(BaseModel):
    """带元数据的文档"""
    content: str
    metadata: DocumentMetadata = DocumentMetadata()


class FeatureImportance(BaseModel):
    """特征重要性"""
    feature: str
    score: float
    token_field: Optional[str] = None


class ExplanationDto(BaseModel):
    """解释数据传输对象"""
    explanations: List[FeatureImportance]
    input_text: str
    output_text: str


class ExplanationGranularity(str, Enum):
    WORD_LEVEL = "word_level_granularity"
    SENTENCE_LEVEL = "sentence_level_granularity"
    PARAGRAPH_LEVEL = "paragraph_level_granularity"
    PHRASE_LEVEL = "phrase_level_granularity"


class SimilarityMetric(Enum):
    COSINE = "cosine"


class RagOutput(BaseModel):
    """RAG系统输出"""
    retrieved_documents: List[DocumentWithMetadata]
    retriever_scores: List[float]
    prompt: str
    generated_responses: List[str]
    metadata: Dict[str, Any]
