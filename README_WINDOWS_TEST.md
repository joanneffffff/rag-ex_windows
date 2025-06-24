# Windows环境RAG系统测试指南

## 系统概述

本系统实现了双空间双索引的RAG（检索增强生成）系统，支持中英文金融数据的统一处理：

- **中文编码器**: `models/finetuned_alphafin_zh`
- **英文编码器**: `models/finetuned_finbert_tatqa`
- **重排序器**: `Qwen/Qwen3-Reranker-0.6B`
- **生成器**: `Qwen/Qwen2-1.5B-Instruct`

## 系统架构

```
用户查询 → 语言检测 → 选择编码器 → FAISS检索 → Qwen重排序 → LLM生成答案
                ↓
        中文查询 → 中文编码器 + 中文FAISS索引
        英文查询 → 英文编码器 + 英文FAISS索引
```

## 数据源选择

### 选项1: 使用evaluate_mrr/jsonl文件
- **优点**: 已经处理好的Q-C-A格式，适合训练和评估
- **文件**: 
  - `evaluate_mrr/alphafin_train_qc.jsonl` (中文)
  - `evaluate_mrr/tatqa_train_qc.jsonl` (英文)

### 选项2: 使用原始cleaned数据
- **优点**: 需要进一步chunk处理，适合作为知识库
- **文件**:
  - `data/alphafin/alphafin_rag_ready_generated_cleaned.json` (中文)
  - `data/tatqa_dataset_raw/tatqa_dataset_train.json` (英文)

## Chunk逻辑整合

### TatQA数据处理 (finetune_encoder.py)
```python
def process_tatqa_to_qca_for_corpus(input_paths):
    # 将paragraphs和tables转换为自然语言chunks
    # 提取单位信息，格式化表格数据
```

### AlphaFin数据处理 (finetune_chinese_encoder.py)
```python
def convert_json_context_to_natural_language_chunks(json_str_context, company_name):
    # 将JSON格式的context转换为自然语言chunks
    # 处理财务报表数据、公司描述等
```

### 统一数据处理器 (xlm/utils/unified_chunk_processor.py)
```python
def process_unified_data(tatqa_paths, alphafin_paths):
    # 整合两种chunk逻辑
    # 统一处理TatQA和AlphaFin数据
```

## Windows环境测试

### 1. 基础测试
```bash
python test_windows_simple.py
```

### 2. 完整测试
```bash
python test_windows_gpu.py
```

### 3. 运行UI
```bash
# 增强UI (推荐)
python run_enhanced_ui.py

# 优化UI
python run_optimized_ui.py

# 本地测试
python run_local_test.py
```

## 测试内容

### GPU环境检查
- CUDA可用性检测
- GPU内存和型号信息
- 自动降级到CPU

### 模型文件检查
- 中文编码器路径验证
- 英文编码器路径验证
- 重排序器模型检查
- 生成器模型检查

### 数据文件检查
- 中文数据文件存在性
- 英文数据文件存在性
- 文件大小统计

### 功能测试
1. **基本检索测试**: 使用简单文档测试检索功能
2. **增强系统测试**: 测试双空间双索引系统
3. **生成器测试**: 测试LLM生成功能
4. **数据处理器测试**: 测试统一数据处理

## 配置说明

### 缓存目录配置
```python
# config/parameters.py
WINDOWS_CACHE_DIR = "M:/huggingface"  # Windows路径
LINUX_CACHE_DIR = "/users/sgjfei3/data/huggingface"  # Linux路径
```

### 设备配置
```python
# 自动检测GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## 常见问题

### 1. 模型文件不存在
**解决方案**: 
- 检查模型路径是否正确
- 确保模型已下载到指定目录
- 修改`config/parameters.py`中的路径

### 2. 数据文件不存在
**解决方案**:
- 检查数据文件路径
- 使用`python xlm/utils/unified_chunk_processor.py`处理原始数据
- 选择可用的数据文件

### 3. GPU内存不足
**解决方案**:
- 使用量化模型
- 减少batch_size
- 使用CPU模式

### 4. 依赖包缺失
**解决方案**:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 系统特性

### 双空间双索引
- 中文查询使用中文编码器和FAISS索引
- 英文查询使用英文编码器和FAISS索引
- 自动语言检测和切换

### 统一重排序
- 所有查询都使用Qwen重排序器
- 提高检索精度

### 兼容性
- 支持Windows和Linux环境
- 自动GPU/CPU切换
- 灵活的数据源选择

## 下一步

1. **运行测试**: `python test_windows_simple.py`
2. **启动UI**: `python run_enhanced_ui.py`
3. **自定义配置**: 修改`config/parameters.py`
4. **添加数据**: 使用统一数据处理器处理新数据

## 文件结构

```
rag-ex_windows/
├── config/parameters.py              # 配置文件
├── xlm/
│   ├── components/
│   │   ├── encoder/                  # 编码器组件
│   │   ├── retriever/                # 检索器组件
│   │   └── reranker/                 # 重排序器组件
│   ├── utils/
│   │   └── unified_chunk_processor.py # 统一数据处理器
│   └── registry/                     # 组件注册表
├── data/                             # 数据目录
├── models/                           # 模型目录
├── test_windows_simple.py           # Windows测试脚本
├── run_enhanced_ui.py               # 增强UI
└── README_WINDOWS_TEST.md           # 本文档
``` 