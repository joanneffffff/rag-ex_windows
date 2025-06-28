# RAG系统扰动策略实验指导

## 📋 概述

本文档提供了RAG系统扰动策略实验的完整指导，包括系统架构、实验设计、运行方法和结果分析。

## 🏗️ 系统架构

### 核心组件

```
RAG系统
├── 检索阶段 (Retrieval)
│   ├── 多阶段检索器 (EnhancedRetriever)
│   ├── FAISS向量检索
│   └── Qwen3-0.6B重排序器
├── 生成阶段 (Generation)
│   ├── 本地LLM生成器 (LocalLLMGenerator)
│   └── 微调金融模型 (SUFE-AIFLM-Lab/Fin-R1)
└── 扰动分析系统
    ├── 扰动器 (Perturbers)
    ├── 解释器 (Explainers)
    ├── 比较器 (Comparators)
    └── 分词器 (Tokenizers)
```

### 扰动位置

| 阶段 | 扰动位置 | 扰动内容 | 分析目标 |
|------|----------|----------|----------|
| **检索阶段** | `reference_text` | 文档内容 | 分析哪些文档信息对检索最重要 |
| **生成阶段** | `input_text` | 完整Prompt | 分析哪些Prompt元素对生成最重要 |

## 🔬 扰动策略

### 可用扰动器

1. **Leave-One-Out Perturber** (`leave_one_out`)
   - 移除单个特征，观察影响
   - 最基础和常用的扰动策略
   - 适用场景：识别关键特征

2. **Reorder Perturber** (`reorder`)
   - 改变文本顺序
   - 测试顺序敏感性
   - 适用场景：顺序敏感性分析

3. **Trend Perturber** (`trend`)
   - 趋势相关扰动
   - 适用于时间序列数据
   - 适用场景：时间序列分析

4. **Year Perturber** (`year`)
   - 年份相关扰动
   - 适用于时间敏感数据
   - 适用场景：时间敏感性分析

5. **Term Perturber** (`term`)
   - 金融术语替换扰动
   - 专业领域词汇替换
   - 适用场景：专业术语分析、领域特异性测试

## 🚀 实验脚本

### 1. 基础测试脚本

**文件**: `test_single_perturber.py`

**目的**: 快速验证扰动系统是否正常工作

**运行方法**:
```bash
python test_single_perturber.py
```

**输出内容**:
- 组件初始化状态
- 标准RAG运行结果
- Leave-One-Out扰动实验结果
- 扰动器类型测试结果

### 2. 全面扰动实验脚本

**文件**: `rag_perturbation_experiment.py`

**目的**: 测试所有扰动器在检索和生成阶段的效果

**运行方法**:
```bash
python rag_perturbation_experiment.py
```

**预期输出**:
```
🔍 检索阶段扰动实验...
--- 测试 leave_one_out ---
--- 测试 reorder ---
--- 测试 trend ---
--- 测试 year ---
--- 测试 term ---
...

🤖 生成阶段扰动实验...
📋 实验总结:
✅ 检索扰动: 5/5 成功
✅ 生成扰动: 5/5 成功
```

### 3. Prompt vs Context扰动对比脚本

**文件**: `prompt_vs_context_perturbation.py`

**目的**: 专门对比Context扰动和Prompt扰动的效果差异

**运行方法**:
```bash
python prompt_vs_context_perturbation.py
```

**输出内容**:
- Context扰动分析结果
- Prompt扰动分析结果
- 两种扰动的对比分析
- 特征重要性排序

## 📊 实验设计

### 测试问题集

```python
test_questions = [
    "首钢股份在2023年上半年的业绩表现如何？",
    "中国平安的财务状况怎么样？",
    "腾讯控股的游戏业务发展如何？"
]
```

### 实验流程

1. **系统初始化**
   - 加载配置
   - 初始化RAG组件
   - 初始化扰动系统组件

2. **标准RAG运行**
   - 运行正常RAG系统
   - 记录基准结果

3. **检索阶段扰动**
   - 对检索到的文档进行扰动
   - 分析文档内容的重要性

4. **生成阶段扰动**
   - 对完整Prompt进行扰动
   - 分析Prompt内容的重要性

5. **结果分析**
   - 特征重要性排序
   - 扰动效果对比
   - 系统鲁棒性评估

## 🔍 结果分析

### 关键指标

1. **特征重要性分数**
   - 范围：0-1
   - 越高表示特征越重要

2. **扰动成功率**
   - 成功运行的扰动器数量
   - 总扰动器数量

3. **Top特征分析**
   - 前5个最重要的特征
   - 特征类型和分布

### 分析维度

1. **检索阶段分析**
   - 哪些文档内容对检索最关键
   - 文档信息的敏感性
   - 检索系统的鲁棒性

2. **生成阶段分析**
   - 哪些Prompt元素对生成最关键
   - Prompt的敏感性
   - 生成系统的鲁棒性

3. **对比分析**
   - Context扰动 vs Prompt扰动
   - 不同扰动策略的效果差异
   - 系统整体的可解释性

## ⚙️ 配置说明

### 系统配置

**文件**: `config/parameters.py`

**关键参数**:
```python
# 生成器配置
max_new_tokens: int = 600  # 生成token数量
temperature: float = 0.2   # 生成温度
top_p: float = 0.8        # Top-p采样

# 检索器配置
retrieval_top_k: int = 100  # FAISS检索数量
rerank_top_k: int = 20      # 重排序数量

# 扰动配置
granularity: ExplanationGranularity.WORD_LEVEL  # 分析粒度
```

### 模型配置

- **生成器模型**: `SUFE-AIFLM-Lab/Fin-R1`
- **编码器模型**: `ProsusAI/finbert`
- **重排序器**: `Qwen/Qwen3-Reranker-0.6B`
- **量化类型**: 4bit (节省GPU内存)

## 🛠️ 故障排除

### 常见问题

1. **CUDA内存不足**
   - 解决方案：使用4bit量化
   - 检查：`config.generator.quantization_type = "4bit"`

2. **模型加载失败**
   - 检查：模型路径是否正确
   - 检查：缓存目录是否存在

3. **扰动器初始化失败**
   - 检查：依赖包是否安装
   - 检查：导入路径是否正确

4. **实验结果为空**
   - 检查：数据源是否正常
   - 检查：检索器是否正常工作

### 调试方法

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **分步测试**
   - 先测试单个组件
   - 再测试完整流程

3. **内存监控**
   ```python
   import torch
   print(f"GPU内存: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
   ```

## 📈 实验扩展

### 自定义扰动器

1. **继承Perturber基类**
   ```python
   from xlm.modules.perturber.perturber import Perturber
   
   class CustomPerturber(Perturber):
       def perturb(self, text: str, features: List[str]) -> List[str]:
           # 实现自定义扰动逻辑
           pass
   ```

2. **集成到实验系统**
   ```python
   self.perturbers['custom'] = CustomPerturber()
   ```

### 新的分析维度

1. **时间维度分析**
   - 不同时间段的扰动效果
   - 时间敏感性分析

2. **领域维度分析**
   - 不同金融领域的扰动效果
   - 领域特异性分析

3. **模型维度分析**
   - 不同模型的扰动效果
   - 模型鲁棒性对比

## 📝 实验报告模板

### 实验基本信息

- **实验日期**: YYYY-MM-DD
- **实验环境**: 硬件配置、软件版本
- **实验目的**: 具体的研究目标

### 实验结果

1. **标准RAG性能**
   - 检索准确率
   - 生成质量评分

2. **扰动分析结果**
   - 各扰动器的成功率
   - Top重要特征列表
   - 特征重要性分布

3. **对比分析**
   - Context vs Prompt扰动效果
   - 不同扰动策略的优劣

### 结论与建议

1. **主要发现**
   - 关键特征识别
   - 系统鲁棒性评估

2. **改进建议**
   - 系统优化方向
   - 进一步研究方向

## 🔗 相关资源

- **项目文档**: README.md
- **配置说明**: CONFIG_STATUS.md
- **测试报告**: linux_test_report.md
- **代码仓库**: 项目根目录

## 📞 技术支持

如遇到问题，请检查：
1. 系统配置是否正确
2. 依赖包是否完整安装
3. 模型文件是否存在
4. GPU内存是否充足

---

*最后更新: 2024年* 