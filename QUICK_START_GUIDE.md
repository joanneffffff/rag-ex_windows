# 扰动实验快速开始指南

## 🚀 快速开始

### 1. 环境准备

确保您的环境满足以下要求：
- Python 3.8+
- CUDA支持的GPU
- 已安装项目依赖包

### 2. 运行基础测试

首先运行基础测试验证系统是否正常工作：

```bash
python test_single_perturber.py
```

**预期输出**:
```
🧪 扰动器测试
✅ 组件初始化完成
✅ 标准RAG运行成功
✅ 扰动实验成功完成！
```

### 3. 运行Prompt vs Context对比实验

```bash
python prompt_vs_context_perturbation.py
```

**预期输出**:
```
🔍 Context扰动实验（检索阶段）
📊 Context扰动分析结果:
🏆 Top 5 重要文档特征:

🤖 Prompt扰动实验（生成阶段）
📊 Prompt扰动分析结果:
🏆 Top 5 重要Prompt特征:
```

### 4. 运行全面扰动实验

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

## 📊 结果解读

### 成功指标

- ✅ **组件初始化**: 所有组件正常加载
- ✅ **标准RAG**: 正常RAG系统运行成功
- ✅ **扰动实验**: 扰动器正常工作
- ✅ **结果分析**: 获得特征重要性分数

### 关键输出

1. **特征重要性分数** (0-1范围)
   - 1.0: 极其重要
   - 0.5: 中等重要
   - 0.0: 不重要

2. **Top特征列表**
   - 显示最重要的5个特征
   - 帮助理解系统决策依据

3. **扰动成功率**
   - 显示各扰动器的运行状态
   - 评估系统稳定性

## 🔧 常见问题

### Q: CUDA内存不足怎么办？
A: 检查配置文件中的量化设置：
```python
# config/parameters.py
use_quantization: bool = True
quantization_type: str = "4bit"
```

### Q: 模型加载失败怎么办？
A: 检查模型路径和缓存目录：
```python
# 确保模型文件存在
model_name: str = "SUFE-AIFLM-Lab/Fin-R1"
cache_dir: str = "/path/to/cache"
```

### Q: 扰动结果为空怎么办？
A: 检查数据源和检索器：
```python
# 确保数据文件存在
chinese_data_path: str = "evaluate_mrr/alphafin_train_qc.jsonl"
```

## 📈 下一步

1. **分析结果**: 查看特征重要性分数
2. **调整参数**: 修改配置参数
3. **扩展实验**: 添加新的扰动策略
4. **深入分析**: 进行更详细的可解释性研究

## 📞 获取帮助

- 查看完整文档: `PERTURBATION_EXPERIMENT_GUIDE.md`
- 检查配置文件: `config/parameters.py`
- 查看测试报告: `linux_test_report.md`

---

*快速开始指南 - 让您5分钟内开始扰动实验* 