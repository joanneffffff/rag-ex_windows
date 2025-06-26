# UI入口文件对比说明

## 文件对比

### 1. `run_optimized_ui.py` - 主服务器入口

**用途**: 生产环境的主要服务器入口
**特点**:
- 使用配置文件中的默认设置
- 可能使用缓存的embedding索引
- 标准的启动流程
- 适合日常使用

**关键配置**:
```python
ui = OptimizedRagUI(
    cache_dir=config.cache_dir,
    use_faiss=True,
    enable_reranker=True,
    # use_existing_embedding_index=None  # 使用config默认值
    # max_alphafin_chunks=None           # 使用config默认值
    window_title="Financial Explainable RAG System",
    title="Financial Explainable RAG System",
    examples=[
        ["德赛电池(000049)的下一季度收益预测如何？"],
        ["用友网络2019年的每股经营活动产生的现金流量净额是多少？"],
        ["下月股价能否上涨?"],
        ["How was internally developed software capitalised?"],
        ["Why did the Operating revenues decreased from 2018 to 2019?"],
        ["Why did the Operating costs decreased from 2018 to 2019?"]
    ]
)
```

### 2. `run_new_chunking_force.py` - 强制新chunking逻辑

**用途**: 测试和强制使用新的chunking逻辑
**特点**:
- 强制删除旧的embedding缓存
- 强制重新计算embedding
- 专门测试新chunking逻辑
- 适合开发和测试

**关键配置**:
```python
# 删除旧的embedding缓存
old_embeddings = glob.glob(os.path.join(cache_dir, "*finetuned_alphafin_zh*.npy"))
old_faiss = glob.glob(os.path.join(cache_dir, "*finetuned_alphafin_zh*.faiss"))

ui = OptimizedRagUI(
    cache_dir=config.cache_dir,
    use_faiss=True,
    enable_reranker=True,
    use_existing_embedding_index=False,  # 强制重新计算
    window_title="Financial RAG System (Force New Chunking)",
    title="Financial RAG System (Force New Chunking)",
    examples=[
        ["什么是市盈率？"],
        ["如何计算ROE？"],
        ["财务报表包括哪些内容？"],
        ["什么是资产负债表？"],
        ["现金流量表的作用是什么？"],
        ["How was internally developed software capitalised?"],
        ["Why did the Operating revenues decreased from 2018 to 2019?"]
    ]
)
```

## 主要差异

### 1. **缓存处理**
- **`run_optimized_ui.py`**: 可能使用现有缓存（取决于config设置）
- **`run_new_chunking_force.py`**: 强制删除旧缓存，重新计算

### 2. **Embedding索引**
- **`run_optimized_ui.py`**: `use_existing_embedding_index=None` (使用config默认值)
- **`run_new_chunking_force.py`**: `use_existing_embedding_index=False` (强制重新计算)

### 3. **示例问题**
- **`run_optimized_ui.py`**: 更多具体的金融问题
- **`run_new_chunking_force.py`**: 更多基础概念问题，适合测试新逻辑

### 4. **UI标题**
- **`run_optimized_ui.py`**: "Financial Explainable RAG System"
- **`run_new_chunking_force.py`**: "Financial RAG System (Force New Chunking)"

## 当前配置状态

### 配置文件设置 (`config/parameters.py`)
```python
@dataclass
class RetrieverConfig:
    use_existing_embedding_index: bool = False  # 强制重新计算
    rerank_top_k: int = 20                      # 20个上下文
```

### UI文件默认行为
```python
# 如果传入None，使用config默认值
self.use_existing_embedding_index = use_existing_embedding_index if use_existing_embedding_index is not None else config.retriever.use_existing_embedding_index
```

## 使用建议

### 生产环境使用
```bash
python run_optimized_ui.py
```
- 使用配置文件中的设置
- 如果config中`use_existing_embedding_index=False`，会自动重新计算
- 适合日常使用

### 开发和测试使用
```bash
python run_new_chunking_force.py
```
- 强制删除旧缓存
- 强制重新计算embedding
- 专门测试新chunking逻辑
- 适合验证新功能

### 当前状态分析

由于配置文件已经设置为：
- `use_existing_embedding_index: bool = False`
- `rerank_top_k: int = 20`

**两个文件现在实际上行为相同**，都会：
1. 强制重新计算embedding
2. 使用新的文档级别chunking
3. 显示20个相关上下文

## 建议

### 简化方案
由于配置已经优化，建议：

1. **保留 `run_optimized_ui.py`** 作为主入口
2. **删除 `run_new_chunking_force.py`** (功能重复)
3. **统一使用主入口** 进行所有操作

### 如果需要强制重新计算
可以在主入口中添加参数：
```python
ui = OptimizedRagUI(
    # ... 其他参数
    use_existing_embedding_index=False,  # 需要时强制重新计算
)
```

## 总结

- **`run_optimized_ui.py`**: 主服务器入口，适合生产环境
- **`run_new_chunking_force.py`**: 测试脚本，功能重复
- **当前配置**: 两个文件行为相同，都会使用新的chunking逻辑
- **建议**: 统一使用主入口，简化维护 