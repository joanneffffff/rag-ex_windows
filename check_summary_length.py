#!/usr/bin/env python3
"""
检查summary字段的长度分布
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def analyze_summary_lengths():
    """分析summary字段的长度分布"""
    print("📊 分析AlphaFin数据中summary字段的长度分布")
    print("=" * 60)
    
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    if not data_path.exists():
        print(f"❌ 数据文件不存在: {data_path}")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📈 总记录数: {len(data)}")
    
    # 统计summary长度
    summary_lengths = []
    long_summaries = []
    
    for i, record in enumerate(data):
        summary = record.get('summary', '')
        if summary:
            length = len(summary)
            summary_lengths.append(length)
            
            # 记录超长的summary
            if length > 8192:
                long_summaries.append({
                    'index': i,
                    'length': length,
                    'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                    'company': record.get('company_name', 'N/A')
                })
    
    if not summary_lengths:
        print("❌ 没有找到有效的summary字段")
        return
    
    # 基本统计
    avg_length = sum(summary_lengths) / len(summary_lengths)
    max_length = max(summary_lengths)
    min_length = min(summary_lengths)
    median_length = sorted(summary_lengths)[len(summary_lengths) // 2]
    
    print(f"\n📏 Summary长度统计:")
    print(f"   平均长度: {avg_length:.1f} 字符")
    print(f"   中位数长度: {median_length} 字符")
    print(f"   最小长度: {min_length} 字符")
    print(f"   最大长度: {max_length} 字符")
    
    # 长度分布
    print(f"\n📊 长度分布:")
    length_ranges = [
        (0, 100, "0-100字符"),
        (100, 500, "100-500字符"),
        (500, 1000, "500-1000字符"),
        (1000, 2000, "1000-2000字符"),
        (2000, 5000, "2000-5000字符"),
        (5000, 8192, "5000-8192字符"),
        (8192, float('inf'), "超过8192字符")
    ]
    
    for start, end, label in length_ranges:
        count = len([l for l in summary_lengths if start <= l < end])
        percentage = (count / len(summary_lengths)) * 100
        print(f"   {label}: {count} 条 ({percentage:.1f}%)")
    
    # 检查是否有超长的summary
    if long_summaries:
        print(f"\n⚠️  发现 {len(long_summaries)} 个超过8192字符的summary:")
        for item in long_summaries[:5]:  # 只显示前5个
            print(f"   索引 {item['index']}: {item['length']} 字符")
            print(f"   公司: {item['company']}")
            print(f"   内容预览: {item['summary']}")
            print()
        
        if len(long_summaries) > 5:
            print(f"   ... 还有 {len(long_summaries) - 5} 个超长summary")
    else:
        print(f"\n✅ 所有summary都在8192字符以内")
    
    # 分析长度分布
    print(f"\n🔍 长度分析:")
    if avg_length < 1000:
        print(f"   ✅ Summary平均长度较短 ({avg_length:.1f}字符)，适合FAISS索引")
    elif avg_length < 3000:
        print(f"   ⚠️  Summary平均长度中等 ({avg_length:.1f}字符)，需要关注")
    else:
        print(f"   ❌ Summary平均长度较长 ({avg_length:.1f}字符)，可能影响性能")
    
    if max_length > 8192:
        print(f"   ❌ 存在超过8192字符的summary，需要处理")
    else:
        print(f"   ✅ 所有summary都在8192字符以内，无需额外处理")
    
    # 建议
    print(f"\n💡 建议:")
    if max_length <= 8192:
        print(f"   ✅ 当前summary长度适合FAISS索引，无需额外chunking")
    else:
        print(f"   ⚠️  建议对超长summary进行截断或分割")
        print(f"   📝 可以考虑在_build_faiss_index中添加长度检查")

def check_faiss_index_impact():
    """检查对FAISS索引的影响"""
    print(f"\n🔍 对FAISS索引的影响分析:")
    print("=" * 60)
    
    data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    if not data_path.exists():
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计用于FAISS索引的文本长度
    faiss_text_lengths = []
    
    for record in data:
        summary = record.get('summary', '')
        if summary:
            faiss_text_lengths.append(len(summary))
    
    if not faiss_text_lengths:
        return
    
    avg_length = sum(faiss_text_lengths) / len(faiss_text_lengths)
    max_length = max(faiss_text_lengths)
    
    print(f"FAISS索引文本统计:")
    print(f"   平均长度: {avg_length:.1f} 字符")
    print(f"   最大长度: {max_length} 字符")
    print(f"   总文本数: {len(faiss_text_lengths)}")
    
    # 评估性能影响
    if avg_length < 500:
        print(f"   ✅ 平均长度较短，FAISS索引性能良好")
    elif avg_length < 1000:
        print(f"   ⚠️  平均长度中等，FAISS索引性能可接受")
    else:
        print(f"   ❌ 平均长度较长，可能影响FAISS索引性能")
    
    if max_length > 8192:
        print(f"   ❌ 存在超长文本，可能导致内存问题")
    else:
        print(f"   ✅ 所有文本都在合理长度范围内")

if __name__ == "__main__":
    analyze_summary_lengths()
    check_faiss_index_impact() 