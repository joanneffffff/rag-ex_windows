import json
import re
from pathlib import Path
from tqdm import tqdm
import argparse

# --- 清洗规则字典 (直接嵌入 Python 代码) ---
CLEANING_RULES_DICT = {
    "financial_keywords": [
        "股票", "证券", "基金", "债券", "财报", "盈利", "亏损", "营收", "利润", "市值", "投资", "金融", "银行", "交易所",
        "融资", "贷款", "利率", "汇率", "经济", "通胀", "通缩", "GDP", "CPI", "PPI", "财政", "税收", "预算", "赤字", "债务",
        "资产", "负债", "权益", "估值", "交易", "市场", "分析师", "评级", "研究报告", "监管", "合规", "风控",
        "并购", "重组", "IPO", "退市", "股息", "分红", "配股", "增发", "回购", "做多", "做空", "牛市", "熊市",
        "指数", "板块", "概念股", "蓝筹股", "成长股", "价值股", "小盘股", "大盘股", "创业板", "科创板",
        "开盘", "收盘", "涨停", "跌停", "交易量", "成交额", "换手率", "K线图", "均线", "技术指标", "基本面",
        "宏观经济", "行业分析", "公司研究", "风险管理", "资产配置", "投资组合", "对冲", "套利", "杠杆",
        "流动性", "信用风险", "市场风险", "操作风险", "政策风险", "系统性风险", "非系统性风险", "金融危机",
        "货币政策", "财政政策", "产业政策", "贸易政策", "进出口", "外汇储备", "国际收支", "资本流动",
        "直接投资", "证券投资", "衍生品", "金融创新", "金融科技", "数字货币", "区块链", "P2P", "众筹",
        "互联网金融", "普惠金融", "绿色金融", "可持续金融", "社会责任投资", "ESG", "央行", "银保监会",
        "证监会", "外管局", "金融机构", "商业银行", "投资银行", "证券公司", "基金公司", "保险公司",
        "信托公司", "资产管理公司", "评级机构", "清算所", "楼市", "房价", "房地产", "房贷", "公积金",
        "土地", "开发商", "物业", "业绩", "报表", "财报", "同比增长", "净利润率", "毛利率", "营业收入",
        "净资产", "总资产", "负债率", "现金流", "股本", "分红派息", "每股收益", "市净率", "市销率",
        "营业利润", "净利润"
    ],
    "negative_keywords": [
        "电影", "影片", "体育", "游戏", "音乐", "艺术", "食谱", "烹饪", "旅行", "度假", "天气", "时尚",
        "名人", "八卦", "健康小贴士", "锻炼", "政治", "选举", "犯罪", "小说", "故事", "诗歌", "历史",
        "地理", "科学项目", "教育", "学生", "老师", "学校", "大学", "娱乐", "新闻", "游乐场", "儿童",
        "小孩", "受伤", "事故", "报警", "医院", "治疗", "乐吧车", "安全带", "法律责任", "律师", "监护人",
        "消费者权益", "消费者", "摔伤", "坠落", "新闻报道", "记者", "受害者", "伤势", "投诉", "维权",
        "曝光", "娱乐中心", "公园", "摩天轮", "游乐设施"
    ],
    "patterns_to_exclude": [
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "\\b(足球|篮球|棒球|网球|排球)\\b",
        "\\b(导演|演员|制片人)\\b",
        "\\b(菜谱|配料|厨具)\\b",
        "\\b(章节|页|段落)\\b",
        "\\b(习近平|特朗普|拜登|普京|默克尔|马克龙)\\b",
        "\\b(海都网|福州新闻|新华社|中新社|人民日报|每日经济新闻|证券时报|XX日报|XX晚报|XX新闻网)\\b",
        "\\b(记者|报道|通讯员|特约评论员)\\b",
        "\\b(原标题|来源|责任编辑)\\b"
    ],
    "min_length_chars": 100,
    "min_financial_keyword_count": 4 # 降低阈值以确保捕捉更多相关金融数据，可根据需要调整
}

# --- 辅助函数：确保目录存在 ---
def ensure_directory_exists(file_path):
    """确保给定文件路径的父目录存在"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# --- 加载清洗规则 (直接从字典加载) ---
def load_cleaning_rules():
    """直接从 CLEANING_RULES_DICT 加载清洗规则并编译正则表达式"""
    rules = CLEANING_RULES_DICT.copy()
    compiled_patterns = []
    if "patterns_to_exclude" in rules:
        for pattern_str in rules["patterns_to_exclude"]:
            compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE))
    rules["compiled_patterns"] = compiled_patterns
    return rules

def prepare_financial_rag_data(input_raw_json_path: str, output_rag_ready_json_path: str):
    """
    整合数据准备流程：
    1. 根据关键词过滤原始数据中的金融相关记录。
    2. 将过滤后的记录解析并转换为 LLM 核心处理脚本所需的 query/context/answer 格式。
    """
    ensure_directory_exists(output_rag_ready_json_path)

    try:
        rules = load_cleaning_rules()
        financial_keywords = [k.lower() for k in rules.get("financial_keywords", [])]
        negative_keywords = [k.lower() for k in rules.get("negative_keywords", [])]
        compiled_patterns = rules.get("compiled_patterns", [])
        min_length_chars = rules.get("min_length_chars", 100)
        min_financial_keyword_count = rules.get("min_financial_keyword_count", 2)

        print(f"正在从 {input_raw_json_path} 读取原始数据进行过滤和格式转换...")
        
        raw_data = []
        try:
            with open(input_raw_json_path, 'r', encoding='utf-8') as f:
                # 尝试加载为JSON Lines (每行一个JSON对象)
                for line in f:
                    raw_data.append(json.loads(line))
        except json.JSONDecodeError:
            # 如果不是JSON Lines，尝试加载为单个JSON数组
            with open(input_raw_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        if not isinstance(raw_data, list):
            raise ValueError("输入文件内容不是有效的JSON数组或JSON Lines格式。")

        print(f"原始记录数量: {len(raw_data)}")

        processed_records = []
        
        # 正则表达式来解析 input 字符串中的 context 和 question
        # 寻找 【问题】：这个模式
        # 然后答案通常是 【答案】：
        input_pattern = re.compile(r'(.*?)【问题】：(.*)', re.DOTALL) # DOTALL让.匹配换行符
        output_pattern = re.compile(r'【答案】：(.*)', re.DOTALL)


        total_filtered_count = 0
        for i, record in tqdm(enumerate(raw_data), total=len(raw_data), desc="过滤与转换数据"):
            raw_input_str = record.get('input', '')
            raw_output_str = record.get('output', '')
            original_split = record.get('split', 'unknown') 
            original_instruction = record.get('instruction', '') 

            if not isinstance(raw_input_str, str) or not raw_input_str.strip():
                continue # 跳过非字符串或空内容

            # --- 阶段 1: 金融数据过滤 ---
            content_lower = raw_input_str.lower()

            if len(raw_input_str) < min_length_chars:
                continue
            if any(neg_kw in content_lower for neg_kw in negative_keywords):
                continue
            if any(pattern.search(raw_input_str) for pattern in compiled_patterns):
                continue
            financial_keyword_hits = sum(1 for kw in financial_keywords if kw in content_lower)
            if financial_keyword_hits < min_financial_keyword_count:
                continue
            
            # --- 阶段 2: 原始数据解析和格式转换 ---
            context_part = ""
            query_part = ""
            answer_part = ""

            # 解析 input 字段
            match_input = input_pattern.match(raw_input_str)
            if match_input:
                context_part = match_input.group(1).strip()
                query_part = match_input.group(2).strip()
            else:
                # 如果没有匹配到【问题】：，则整个 input 作为 context，query 使用默认值
                context_part = raw_input_str.strip()
                query_part = "请总结并回答问题。" 

            # 解析 output 字段
            match_output = output_pattern.match(raw_output_str)
            if match_output:
                answer_part = match_output.group(1).strip()
            else:
                # 如果没有匹配到【答案】：，则整个 output 作为 answer
                answer_part = raw_output_str.strip()

            # 确保解析出的关键部分非空，才加入最终结果
            if context_part and query_part and answer_part:
                processed_records.append({
                    'query': query_part,
                    'context': context_part,
                    'answer': answer_part,
                    'doc_id': f"raw_doc_{i}", # 生成一个简单的doc_id
                    'original_split': original_split, 
                    'original_instruction': original_instruction 
                })
                total_filtered_count += 1
            else:
                # 记录那些通过了金融过滤，但解析后缺少关键字段的记录
                # print(f"警告: 记录 {i} 过滤通过但解析失败 (缺少 context, query 或 answer)。")
                pass # 不打印过多警告，仅统计

        print(f"过滤并转换后，最终用于 LLM 处理的记录数量: {total_filtered_count}")

        # 保存结果
        print(f"保存处理后的数据到: {output_rag_ready_json_path}")
        with open(output_rag_ready_json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_records, f, ensure_ascii=False, indent=4)
        
        print("\n数据准备完成。前5条处理后的记录示例:")
        for i in range(min(5, len(processed_records))):
            record_summary = {k: processed_records[i].get(k, '') for k in ['query', 'context', 'answer']}
            print(record_summary)

    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"错误: 无法解析 JSON 文件 {input_raw_json_path} 或文件格式不正确 - {e}")
        print("请检查文件内容是否为有效的JSON数组或JSON Lines格式。")
    except Exception as e:
        print(f"发生未知错误: {e}")
        import traceback
        traceback.print_exc()

# --- 主执行部分 ---
if __name__ == "__main__":
    print("开始准备金融数据...")
    parser = argparse.ArgumentParser(
        description="整合数据准备流程：过滤金融数据并转换为LLM核心处理脚本所需格式。"
    )
    parser.add_argument("--input_raw_json_file", type=str, required=True, help="原始大文件的路径（包含instruction/input/output/split）。")
    parser.add_argument("--output_rag_ready_json_file", type=str, default="data/alphafin/alphafin_rag_ready_0627.json", help="处理后数据保存的路径，供LLM核心处理脚本使用。")
    
    args = parser.parse_args()
    
    prepare_financial_rag_data(args.input_raw_json_file, args.output_rag_ready_json_file)