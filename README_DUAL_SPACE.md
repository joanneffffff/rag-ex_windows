# 双空间双索引RAG系统

## 概述

本系统实现了**中英文分开的embedding空间和FAISS索引**，支持：
- 中文查询 → 中文编码器 + 中文FAISS索引
- 英文查询 → 英文编码器 + 英文FAISS索引
- 统一的重排序器（Qwen3-0.6B）
- 统一的生成式LLM

## 系统架构

```
查询输入 → 语言检测 → 选择编码器/索引 → FAISS检索 → Qwen重排序 → LLM生成答案
```

### 核心组件

1. **EnhancedRetriever**: 双空间双索引检索器
2. **QwenReranker**: 基于Qwen3-0.6B的重排序器
3. **DualLanguageLoader**: 双语言数据加载器
4. **配置系统**: 支持中英文编码器路径配置

## 文件结构

```
xlm/
├── components/
│   └── retriever/
│       ├── enhanced_retriever.py    # 双空间检索器
│       └── reranker.py              # Qwen重排序器
├── utils/
│   └── dual_language_loader.py      # 双语言数据加载器
└── registry/
    └── retriever.py                 # 检索器注册表

config/
└── parameters.py                    # 配置文件

run_enhanced_ui.py                   # 增强UI入口
test_dual_space_retriever.py         # 测试脚本
```

## 配置说明

### 编码器配置

在 `config/parameters.py` 中：

```python
@dataclass
class EncoderConfig:
    # 中文微调模型路径
    chinese_model_path: str = "models/finetuned_alphafin_zh"
    # 英文微调模型路径
    english_model_path: str = "models/finetuned_finbert_tatqa"
```

### 重排序器配置

```python
@dataclass
class RerankerConfig:
    model_name: str = "Qwen/Qwen3-Reranker-0.6B"
    use_quantization: bool = True
    quantization_type: str = "8bit"  # "8bit" 或 "4bit"
    enabled: bool = True
```

### 检索器配置

```python
@dataclass
class RetrieverConfig:
    use_faiss: bool = True
    retrieval_top_k: int = 100  # FAISS检索的top-k
    rerank_top_k: int = 10      # 重排序后的top-k
```

## 使用方法

### 1. 基本使用

```python
from config.parameters import Config
from xlm.registry.retriever import load_enhanced_retriever

# 创建配置
config = Config()
config.reranker.enabled = True
config.retriever.use_faiss = True

# 加载增强检索器
retriever = load_enhanced_retriever(
    config=config,
    chinese_data_path="data/alphafin/alphafin_rag_ready_generated_cleaned.json",
    english_data_path="data/tatqa_dataset_raw/tatqa_dataset_train.json"
)

# 检索文档
documents, scores = retriever.retrieve(
    text="安井食品主要生产什么产品？",
    top_k=5,
    return_scores=True
)
```

### 2. 运行UI

```bash
python run_enhanced_ui.py
```

### 3. 测试系统

```bash
python test_dual_space_retriever.py
```

## 数据格式

### 中文数据（AlphaFin）

```json
{
    "question": "安井食品主要生产什么产品？",
    "answer": "火锅料制品、面米制品和菜肴制品",
    "context": "安井食品是一家专注于速冻食品生产的企业...",
    "stock_name": "安井食品"
}
```

### 英文数据（TatQA）

```json
{
    "question": "What does Apple Inc. specialize in?",
    "answer": "consumer electronics, computer software, and online services",
    "context": "Apple Inc. is an American multinational technology company..."
}
```

## 系统特性

### 1. 自动语言检测

系统使用 `langdetect` 库自动检测查询语言：
- 中文查询 → 使用中文编码器和索引
- 英文查询 → 使用英文编码器和索引

### 2. 双FAISS索引

- `chinese_index`: 中文文档的FAISS索引
- `english_index`: 英文文档的FAISS索引
- 分别使用对应的编码器进行向量化

### 3. 统一重排序

无论中英文查询，都使用相同的Qwen3-0.6B重排序器对检索结果进行重排序。

### 4. 配置化

所有参数都可以通过配置文件调整，无需修改代码。

## 性能优化

### 1. 量化支持

重排序器支持8bit和4bit量化：
```python
config.reranker.quantization_type = "8bit"  # 或 "4bit"
```

### 2. Flash Attention

支持Flash Attention加速：
```python
reranker = QwenReranker(use_flash_attention=True)
```

### 3. GPU加速

FAISS支持GPU加速：
```python
config.retriever.use_gpu = True
```

## 故障排除

### 1. 模型加载失败

检查模型路径是否正确：
```python
# 确保以下路径存在
config.encoder.chinese_model_path = "models/finetuned_alphafin_zh"
config.encoder.english_model_path = "models/finetuned_finbert_tatqa"
```

### 2. 内存不足

启用量化：
```python
config.reranker.use_quantization = True
config.reranker.quantization_type = "4bit"  # 更节省内存
```

### 3. 检索效果不佳

调整参数：
```python
config.retriever.retrieval_top_k = 200  # 增加检索数量
config.retriever.rerank_top_k = 20      # 增加重排序数量
```

## 扩展开发

### 1. 添加新语言

1. 在 `EncoderConfig` 中添加新语言编码器路径
2. 修改 `EnhancedRetriever` 的语言检测逻辑
3. 添加对应的FAISS索引

### 2. 自定义重排序器

继承 `QwenReranker` 类并重写相关方法：
```python
class CustomReranker(QwenReranker):
    def format_instruction(self, instruction, query, document):
        # 自定义指令格式
        pass
```

### 3. 添加新的数据源

在 `DualLanguageLoader` 中添加新的数据加载方法：
```python
def load_custom_data(self, file_path: str) -> List[DocumentWithMetadata]:
    # 自定义数据加载逻辑
    pass
```

## 总结

双空间双索引系统实现了：
- ✅ 中英文完全分离的embedding空间
- ✅ 自动语言检测和编码器选择
- ✅ 统一的Qwen重排序器
- ✅ 配置化的参数管理
- ✅ 完整的UI界面
- ✅ 详细的测试和文档

系统设计完全符合您的需求：**中文查询用中文编码器+中文FAISS，英文查询用英文编码器+英文FAISS，统一用Qwen重排序器，最后用LLM生成答案**。 