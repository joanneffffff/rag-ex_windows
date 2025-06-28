import json
from pathlib import Path

def analyze_data_structure():
    """分析生成数据的结构"""
    file_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"总记录数：{len(data)}")
        
        if len(data) > 0:
            # 分析第一条记录的结构
            first_record = data[0]
            print("\n第一条记录的键：")
            for key in first_record.keys():
                print(f"  - {key}")
            
            print("\n第一条记录的完整内容：")
            print(json.dumps(first_record, ensure_ascii=False, indent=2))
            
            # 统计元数据字段的存在情况
            metadata_fields = ['company_name', 'stock_code', 'report_date']
            print(f"\n元数据字段统计：")
            for field in metadata_fields:
                count = sum(1 for record in data if field in record and record[field])
                print(f"  {field}: {count}/{len(data)} 条记录")
            
            # 检查核心字段
            core_fields = ['original_context', 'original_question', 'original_answer', 'generate_question', 'summary']
            print(f"\n核心字段统计：")
            for field in core_fields:
                count = sum(1 for record in data if field in record and record[field])
                print(f"  {field}: {count}/{len(data)} 条记录")
                
    except Exception as e:
        print(f"分析数据时发生错误: {e}")

if __name__ == '__main__':
    analyze_data_structure() 