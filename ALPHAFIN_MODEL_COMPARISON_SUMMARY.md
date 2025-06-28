# AlphaFin模型比较工具总结

## 🎯 工具概述

我们创建了一套完整的工具来使用AlphaFin数据集中的真实问题比较不同生成器模型的效果。

## 📁 创建的文件

### 1. 核心比较脚本
- **`compare_models_with_alphafin.py`**: 主要的模型比较脚本
  - 支持通过`--model_names`参数指定多个模型
  - 自动从AlphaFin数据集加载真实问题
  - 生成详细的比较报告

### 2. 辅助工具
- **`test_alphafin_data_loading.py`**: 测试AlphaFin数据加载功能
- **`run_alphafin_model_comparison.py`**: 交互式运行示例的工具
- **`ALPHAFIN_MODEL_COMPARISON_GUIDE.md`**: 详细使用指南

### 3. 配置文件
- **`config/parameters.py`**: 已更新为使用Qwen3-8B作为默认生成器

## 🚀 快速开始

### 1. 基本使用
```bash
# 使用默认设置比较Qwen3-8B和Fin-R1
python compare_models_with_alphafin.py
```

### 2. 指定模型
```bash
# 比较Qwen3-8B和Qwen2-7B
python compare_models_with_alphafin.py --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B
```

### 3. 使用评估数据集
```bash
# 使用alphafin_eval.jsonl中的问题
python compare_models_with_alphafin.py --data_path evaluate_mrr/alphafin_eval.jsonl
```

### 4. 交互式工具
```bash
# 运行交互式示例
python run_alphafin_model_comparison.py
```

## 📊 可用的AlphaFin问题

从测试中我们发现AlphaFin数据集包含真实的金融问题：

### 训练数据集问题示例
1. "德赛电池(000049)的下一季度收益预测如何？"
2. "用友网络2019年的每股经营活动产生的现金流量净额是多少？"
3. "下月股价能否上涨?"
4. "福耀玻璃（600660）的股东权益合计（不含少数股东权益）是多少？"
5. "旗滨集团（601636）的下月最终收益结果是:'跌',下跌概率:一般"

### 评估数据集问题示例
1. "安井食品（603345）何时发布年度报告?"
2. "兴业银行（601166）的兴银行三季报如何"
3. "北方华创（002371）的总市值是多少万元？"
4. "中国中免（601888）的市净率（总市值/净资产）是多少？"
5. "东鹏控股(003012)新产品会否带来盈利改善?"

## 🔧 支持的模型

### 推荐模型
- **Qwen/Qwen3-8B**: 主要推荐，内存效率高，生成质量好
- **Qwen/Qwen2-7B**: 较小版本，速度快
- **Qwen/Qwen2-1.5B**: 最小版本，最快

### 其他模型
- **SUFE-AIFLM-Lab/Fin-R1**: 金融专用，但内存需求高
- **Llama2-7B-chat-hf**: Llama2聊天版本
- **microsoft/DialoGPT-medium**: 微软对话模型

## 📈 输出结果

### 1. 单个模型结果
- JSON格式的详细结果
- 包含每个问题的回答、时间、token数等

### 2. 比较报告
- Markdown格式的综合报告
- 性能对比表格
- 示例回答展示

### 3. 性能指标
- 成功率
- 平均生成时间
- 平均Token数
- GPU内存使用

## 🎯 使用建议

### 1. 内存优化
```bash
# 对于内存受限的环境
python compare_models_with_alphafin.py --max_questions 3
```

### 2. 模型选择
```bash
# 推荐组合
python compare_models_with_alphafin.py --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B
```

### 3. 数据源选择
```bash
# 使用训练数据
python compare_models_with_alphafin.py --data_path evaluate_mrr/alphafin_train_qc.jsonl

# 使用评估数据
python compare_models_with_alphafin.py --data_path evaluate_mrr/alphafin_eval.jsonl
```

## 🔍 故障排除

### 常见问题
1. **内存不足**: 减少`--max_questions`数量
2. **模型加载失败**: 检查模型名称和网络连接
3. **数据文件不存在**: 检查文件路径

### 解决方案
```bash
# 使用CPU（较慢但稳定）
python compare_models_with_alphafin.py --device cpu

# 使用较小的模型
python compare_models_with_alphafin.py --model_names Qwen/Qwen2-1.5B
```

## 📝 扩展功能

### 1. 自定义问题
可以修改脚本加载自定义问题集

### 2. 添加评估指标
可以扩展脚本添加更多评估维度

### 3. 批量测试
可以创建脚本进行大规模模型测试

## 🎉 总结

这套工具提供了：

1. ✅ **完整的模型比较框架**
2. ✅ **真实的AlphaFin问题**
3. ✅ **灵活的配置选项**
4. ✅ **详细的性能分析**
5. ✅ **友好的用户界面**

通过这些工具，您可以：
- 科学地比较不同模型在金融问题上的表现
- 选择最适合的模型用于生产环境
- 持续监控和优化模型性能

**推荐使用Qwen3-8B作为主要生成器模型**，它在内存效率、生成质量和稳定性方面都表现优异！ 