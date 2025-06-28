# AlphaFin数据集模型比较指南

## 🎯 概述

这个脚本使用AlphaFin数据集中的真实问题来比较不同生成器模型的效果，支持通过命令行参数指定不同的模型。

## 🚀 快速开始

### 基本用法

```bash
# 使用默认模型比较（Qwen3-8B vs Fin-R1）
python compare_models_with_alphafin.py

# 指定特定模型
python compare_models_with_alphafin.py --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B

# 指定更多参数
python compare_models_with_alphafin.py \
    --model_names Qwen/Qwen3-8B SUFE-AIFLM-Lab/Fin-R1 \
    --max_questions 10 \
    --device cuda:1 \
    --output_dir my_comparison_results
```

### 支持的模型

```bash
# Qwen系列
python compare_models_with_alphafin.py --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B Qwen/Qwen2-1.5B

# 金融专用模型
python compare_models_with_alphafin.py --model_names SUFE-AIFLM-Lab/Fin-R1 Qwen/Qwen3-8B

# 混合比较
python compare_models_with_alphafin.py --model_names Qwen/Qwen3-8B Llama2-7B-chat-hf SUFE-AIFLM-Lab/Fin-R1
```

## 📋 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_names` | list | `["Qwen/Qwen3-8B", "SUFE-AIFLM-Lab/Fin-R1"]` | 要测试的模型名称列表 |
| `--data_path` | str | `"evaluate_mrr/alphafin_train_qc.jsonl"` | AlphaFin数据文件路径 |
| `--max_questions` | int | `5` | 最大测试问题数量 |
| `--device` | str | `"cuda:1"` | GPU设备 |
| `--output_dir` | str | `"model_comparison_results"` | 输出目录 |

## 📊 输出结果

### 1. 单个模型结果文件
- `Qwen_Qwen3_8B_alphafin_results.json`
- `SUFE_AIFLM_Lab_Fin_R1_alphafin_results.json`

### 2. 比较报告
- `model_comparison_report.md`

### 3. 结果格式

```json
{
  "model_name": "Qwen/Qwen3-8B",
  "device": "cuda:1",
  "success_count": 5,
  "total_time": 195.5,
  "avg_tokens": 31.6,
  "memory_usage": 5.98,
  "questions": [
    {
      "question": "什么是股票投资？",
      "response": "股票投资是指通过购买公司的股份...",
      "tokens": 38,
      "time": 39.42,
      "success": true
    }
  ]
}
```

## 🧪 使用示例

### 示例1：快速比较两个模型

```bash
python compare_models_with_alphafin.py \
    --model_names Qwen/Qwen3-8B SUFE-AIFLM-Lab/Fin-R1 \
    --max_questions 3
```

### 示例2：详细比较多个模型

```bash
python compare_models_with_alphafin.py \
    --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B Llama2-7B-chat-hf \
    --max_questions 10 \
    --output_dir detailed_comparison
```

### 示例3：使用不同GPU

```bash
python compare_models_with_alphafin.py \
    --model_names Qwen/Qwen3-8B \
    --device cuda:0 \
    --max_questions 5
```

## 📈 性能指标

脚本会计算以下性能指标：

1. **成功率**: 成功生成的问题数量 / 总问题数量
2. **平均生成时间**: 所有成功生成的平均时间
3. **平均Token数**: 生成回答的平均长度
4. **GPU内存使用**: 模型占用的GPU内存

## 🔧 技术特性

### 内存管理
- 自动清理GPU内存
- 4bit量化支持
- 分离模型测试避免内存冲突

### 错误处理
- 优雅处理模型加载失败
- 自动回退到默认问题
- 详细的错误日志

### 数据加载
- 自动从AlphaFin数据集加载问题
- 支持多种数据格式
- 可配置问题数量

## 🎯 最佳实践

### 1. 内存优化
```bash
# 对于内存受限的环境，使用较少的测试问题
python compare_models_with_alphafin.py --max_questions 3

# 使用4bit量化（默认启用）
# 脚本会自动应用4bit量化以节省内存
```

### 2. 模型选择
```bash
# 推荐组合：Qwen3-8B + 其他模型
python compare_models_with_alphafin.py \
    --model_names Qwen/Qwen3-8B Qwen/Qwen2-7B

# 避免同时测试多个大模型
# 建议一次只测试2-3个模型
```

### 3. 结果分析
```bash
# 查看生成的比较报告
cat model_comparison_results/model_comparison_report.md

# 分析单个模型结果
cat Qwen_Qwen3_8B_alphafin_results.json | jq '.success_count'
```

## 🚨 注意事项

1. **内存要求**: 确保GPU有足够内存运行选择的模型
2. **模型可用性**: 确保模型名称正确且可访问
3. **数据文件**: 确保AlphaFin数据文件存在
4. **网络连接**: 首次运行需要下载模型

## 🔍 故障排除

### 常见问题

1. **CUDA OOM错误**
   ```bash
   # 减少测试问题数量
   python compare_models_with_alphafin.py --max_questions 2
   
   # 使用CPU（较慢但稳定）
   python compare_models_with_alphafin.py --device cpu
   ```

2. **模型加载失败**
   ```bash
   # 检查模型名称是否正确
   # 确保网络连接正常
   # 尝试使用较小的模型
   ```

3. **数据文件不存在**
   ```bash
   # 检查文件路径
   ls evaluate_mrr/alphafin_train_qc.jsonl
   
   # 使用其他数据文件
   python compare_models_with_alphafin.py --data_path your_data.jsonl
   ```

## 📝 扩展功能

### 自定义问题
可以修改脚本中的`load_alphafin_questions`函数来加载自定义问题：

```python
def load_custom_questions():
    return [
        "你的自定义问题1",
        "你的自定义问题2",
        # ...
    ]
```

### 添加新的评估指标
可以在`test_model_with_alphafin_questions`函数中添加新的评估指标，如：
- 回答相关性评分
- 事实准确性检查
- 语言流畅度评估

## 🎉 总结

这个脚本提供了一个完整的框架来比较不同生成器模型在AlphaFin数据集上的表现，帮助您选择最适合的模型用于生产环境。 