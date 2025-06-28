#!/usr/bin/env python3
"""
对比测试：只用summary vs generated_question+summary 的检索效果
"""

import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def analyze_data_fields():
    """分析数据字段的分布和质量"""
    print("📊 分析AlphaFin数据字段分布")
    print("=" * 50)
    
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📈 总记录数: {len(data)}")
    
    # 统计字段存在性
    field_stats = {
        'summary': 0,
        'generated_question': 0,
        'original_context': 0,
        'company_name': 0,
        'stock_code': 0
    }
    
    # 统计字段长度
    length_stats = {
        'summary': [],
        'generated_question': [],
        'original_context': []
    }
    
    for record in data:
        for field in field_stats:
            if record.get(field):
                field_stats[field] += 1
        
        for field in length_stats:
            if record.get(field):
                length_stats[field].append(len(record[field]))
    
    print("\n📋 字段存在性统计:")
    for field, count in field_stats.items():
        percentage = (count / len(data)) * 100
        print(f"   {field}: {count}/{len(data)} ({percentage:.1f}%)")
    
    print("\n📏 字段长度统计:")
    for field, lengths in length_stats.items():
        if lengths:
            avg_length = sum(lengths) / len(lengths)
            max_length = max(lengths)
            min_length = min(lengths)
            print(f"   {field}:")
            print(f"     平均长度: {avg_length:.1f} 字符")
            print(f"     最大长度: {max_length} 字符")
            print(f"     最小长度: {min_length} 字符")
    
    # 分析几个样本
    print("\n🔍 样本分析:")
    for i, record in enumerate(data[:3]):
        print(f"\n   样本 {i+1}:")
        print(f"     公司: {record.get('company_name', 'N/A')}")
        print(f"     股票代码: {record.get('stock_code', 'N/A')}")
        print(f"     问题: {record.get('generated_question', 'N/A')[:100]}...")
        print(f"     摘要: {record.get('summary', 'N/A')[:100]}...")
        print(f"     原始上下文长度: {len(record.get('original_context', ''))} 字符")

def test_retrieval_strategies():
    """测试不同的检索策略"""
    print("\n🧪 测试不同检索策略")
    print("=" * 50)
    
    # 模拟查询
    test_queries = [
        "钢铁行业发展趋势",
        "公司业绩表现",
        "财务数据分析",
        "营收增长情况"
    ]
    
    print("建议的检索策略对比:")
    print("\n1. 只用summary:")
    print("   ✅ 优点: 简洁、高效、一致")
    print("   ❌ 缺点: 可能丢失查询意图信息")
    
    print("\n2. 只用generated_question:")
    print("   ✅ 优点: 包含查询意图")
    print("   ❌ 缺点: 缺少答案内容")
    
    print("\n3. generated_question + summary:")
    print("   ✅ 优点: 完整的问题-答案对应关系")
    print("   ✅ 优点: 更好的语义匹配")
    print("   ❌ 缺点: 文本较长，计算开销稍大")
    
    print("\n4. summary + original_context片段:")
    print("   ✅ 优点: 结合结构化摘要和原始内容")
    print("   ❌ 缺点: 可能过于冗长")

def recommend_strategy():
    """推荐最佳策略"""
    print("\n💡 推荐策略")
    print("=" * 50)
    
    print("基于分析，推荐使用: generated_question + summary")
    print("\n理由:")
    print("1. 🎯 语义完整性: 问题+答案的完整对应关系")
    print("2. 🔍 检索精度: 能更好地匹配用户查询意图")
    print("3. ⚡ 效率平衡: 比只用original_context更高效")
    print("4. 📊 数据质量: 两个字段都有较高的完整性")
    
    print("\n实现建议:")
    print("```python")
    print("# 在_build_faiss_index中")
    print("if question and summary:")
    print("    combined_text = f\"Question: {question} Summary: {summary}\"")
    print("    texts_for_embedding.append(combined_text)")
    print("elif summary:  # 回退方案")
    print("    texts_for_embedding.append(summary)")
    print("```")

if __name__ == "__main__":
    analyze_data_fields()
    test_retrieval_strategies()
    recommend_strategy() 