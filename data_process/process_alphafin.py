import json
import re
from tqdm import tqdm
from pathlib import Path
import os

def process_alphafin_data():
    # --- 核心清洗逻辑 (从 alphafin_data_clean.py 提取) ---
    FINANCIAL_KEYWORDS = [
        '金融', '银行', '央行', '证券', '股票', '股市', '交易所', '期货', '期权', '基金', '债券',
        '投资', '融资', '贷款', '存款', '利率', '汇率', '货币', '经济', '通胀', '通缩', 'GDP',
        'CPI', 'PPI', '财政', '税收', '预算', '赤字', '债务', '财报', '盈利', '亏损', '营收',
        '成本', '利润', '收入', '支出', '资产', '负债', '权益', '市值', '估值', '交易', '市场',
        '分析师', '评级', '研究报告', '监管', '合规', '风控', '并购', '重组', 'IPO', '退市',
        '股息', '分红', '配股', '增发', '回购', '做多', '做空', '牛市', '熊市', '震荡', '波动',
        '指数', '板块', '概念股', '蓝筹股', '成长股', '价值股', '小盘股', '大盘股', '创业板',
        '科创板', '主板', '新三板', '开盘', '收盘', '涨停', '跌停', '交易量', '成交额', '换手率',
        'K线图', '均线', '技术指标', '基本面', '宏观经济', '行业分析', '公司研究', '风险管理',
        '资产配置', '投资组合', '对冲', '套利', '杠杆', '保证金', '爆仓', '流动性', '信用风险',
        '市场风险', '操作风险', '政策风险', '流动性风险', '系统性风险', '非系统性风险', '金融危机',
        '货币政策', '财政政策', '产业政策', '贸易政策', '国际贸易', '进出口', '外汇储备', '国际收支',
        '资本流动', '直接投资', '证券投资', '衍生品', '金融创新', '金融科技', '数字货币', '区块链',
        'P2P', '众筹', '互联网金融', '普惠金融', '绿色金融', '可持续金融', '社会责任投资', 'ESG',
        '央行', '银保监会', '证监会', '外管局', '金融机构', '商业银行', '投资银行', '证券公司',
        '基金公司', '保险公司', '信托公司', '资产管理公司', '评级机构', '交易所', '清算所',
        '楼市', '房价', '房地产', '房贷', '公积金', '土地', '开发商', '物业'
    ]
    NEGATIVE_KEYWORDS = [
        '电影', '影片', '体育', '游戏', '音乐', '艺术', '食谱', '烹饪',
        '旅行', '度假', '天气', '时尚', '名人', '八卦',
        '健康小贴士', '锻炼', '政治', '选举', '犯罪', '小说',
        '故事', '诗歌', '历史', '地理', '科学项目',
        '教育', '学生', '老师', '学校', '大学', '娱乐', '新闻',
        '游乐场', '儿童', '小孩', '受伤', '事故', '报警', '医院', '治疗',
        '乐吧车', '安全带', '法律责任', '律师', '监护人', '消费者权益', '消费者',
        '摔伤', '坠落', '新闻报道', '记者', '受害者', '伤势', '投诉', '维权', '曝光',
        '娱乐中心', '公园', '摩天轮', '游乐设施'
    ]
    PATTERNS_TO_EXCLUDE = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'\\b(习近平|特朗普|拜登|普京|默克尔|马克龙)\\b',
        r'\\b(海都网|福州新闻|新华社|中新社|人民日报|每日经济新闻|证券时报|XX日报|XX晚报|XX新闻网)\\b',
        r'\\b(记者|报道|通讯员|特约评论员)\\b',
        r'\\b(原标题|来源|责任编辑)\\b'
    ]
    COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in PATTERNS_TO_EXCLUDE]
    MIN_LENGTH_CHARS = 100
    MIN_FINANCIAL_KEYWORD_COUNT = 5

    # --- 文件路径 ---
    input_file = 'data/alphafin/data.json'
    rag_ready_file = 'data/alphafin/alphafin_rag_ready.json'

    # 确保输出目录存在
    output_dir = Path(rag_ready_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 清洗数据 ---
    try:
        print(f'Loading raw data from {input_file}...')
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        print(f'Raw record count: {len(raw_data)}')
        
        filtered_data = []
        for record in tqdm(raw_data, desc='Cleaning data'):
            content = record.get('input', '')
            if not isinstance(content, str) or not content.strip():
                continue

            content_lower = content.lower()
            
            if len(content) < MIN_LENGTH_CHARS:
                continue
            if any(neg_kw in content_lower for neg_kw in NEGATIVE_KEYWORDS):
                continue
            if any(pattern.search(content) for pattern in COMPILED_PATTERNS):
                continue
            
            financial_keyword_hits = sum(1 for kw in FINANCIAL_KEYWORDS if kw in content_lower)
            if financial_keyword_hits < MIN_FINANCIAL_KEYWORD_COUNT:
                continue
            
            filtered_data.append(record)
            
        print(f'Cleaned record count: {len(filtered_data)}')

        # --- 2. 转换为RAG格式 ---
        rag_data = []
        for record in tqdm(filtered_data, desc='Converting to RAG format'):
            context = record.get('input', '').strip()
            if context:
                rag_data.append({
                    'question': '请总结这篇财经新闻并提取关键信息。',
                    'context': context,
                    'answer': record.get('output', '') 
                })
        
        with open(rag_ready_file, 'w', encoding='utf-8') as f:
            json.dump(rag_data, f, ensure_ascii=False, indent=4)
        print(f'RAG-ready data saved to {rag_ready_file}')
        print(f'Total RAG-ready records: {len(rag_data)}')
        print('Data processing complete.')

    except FileNotFoundError:
        print(f'Error: Input file not found at {input_file}')
    except json.JSONDecodeError:
        print(f'Error: Could not decode JSON from {input_file}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')

def extract_info_from_context(context):
    company = None
    stock_code = None
    date = None
    # 1. "这是以XXX（YYYYYY）"或"这是以XXX（YYYYYY）："
    m = re.match(r"^这是以([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+)[（(]([0-9]{6})[)）]", context)
    if m:
        company = m.group(1).strip()
        stock_code = m.group(2)
    else:
        m2 = re.match(r"^以下数据是([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+)", context)
        if m2:
            company_full = m2.group(1).strip()
            m_code = re.search(r"([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)(\d{6}\.(SZ|SH))", company_full)
            if m_code:
                company = m_code.group(1).strip()
                stock_code = m_code.group(2)
            else:
                m_code2 = re.search(r"([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+?)(\d{6})", company_full)
                if m_code2:
                    company = m_code2.group(1).strip()
                    stock_code = m_code2.group(2)
                else:
                    company = re.split(r"的|时间为|:|：", company_full)[0].strip()
        # 股票代码兜底
        if not stock_code:
            m_code = re.search(r"(\d{6}\.(SZ|SH))", context)
            if m_code:
                stock_code = m_code.group(1)
            else:
                m_code2 = re.search(r"[（(](\d{6})[)）]", context)
                if m_code2:
                    stock_code = m_code2.group(1)
    # 股票代码补全后缀
    if stock_code and re.fullmatch(r"\d{6}", stock_code):
        # 在context中查找完整代码
        m_full = re.search(rf"{stock_code}\.(SZ|SH)", context)
        if m_full:
            stock_code = f"{stock_code}.{m_full.group(1)}"
    # 时间
    m_date = re.search(r"时间为(\d{4}-\d{2}-\d{2})", context)
    if m_date:
        date = m_date.group(1)
    else:
        m_date2 = re.search(r"(\d{4}-\d{2}-\d{2})", context)
        if m_date2:
            date = m_date2.group(1)
    return company, stock_code, date

def clean_instruction(text):
    # 删除instruction部分
    patterns = [
        r"你是一个股票分析师.*?数据如下：",
        r"请根据以下内容回答问题：",
        r"【问题】：.*?。",
        r"数据如下：",
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.DOTALL)
    return text.strip()

def kv_textify(context):
    # 尝试将dict或json字符串转为k-v文本
    if isinstance(context, dict):
        return '；'.join([f"{k}: {v}" for k, v in context.items()])
    try:
        ctx = json.loads(context)
        if isinstance(ctx, dict):
            return '；'.join([f"{k}: {v}" for k, v in ctx.items()])
    except Exception:
        pass
    return context

def replace_company_in_question(question, company, stock_code=None):
    if not company:
        return question
    if stock_code:
        company_full = f"{company}（{stock_code}）"
    else:
        company_full = company
    fuzzy_words = ["公司", "该股票", "该股", "股票", "这个股票", "这只股票", "该公司"]
    for word in fuzzy_words:
        # 允许句首或非中文/数字/字母前缀，确保原词被完整替换
        question = re.sub(
            rf'(^|[^\u4e00-\u9fa5A-Za-z0-9]){word}(?![\u4e00-\u9fa5A-Za-z0-9])',
            lambda m: (m.group(1) if m.group(1) else '') + company_full,
            question
        )
    return question

def process_alphafin_contexts(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = []
    for idx, item in enumerate(data):
        question = item.get('question', '')
        context = item.get('context', '')
        company, stock_code, date = extract_info_from_context(context)
        replaced_question = replace_company_in_question(question, company, stock_code)
        # 打印调试信息
        print(f"样例{idx+1}:")
        print('原始question:', question)
        print('提取公司名:', company)
        print('提取股票代码:', stock_code)
        print('替换后question:', replaced_question)
        print('-'*40)
        # 清理context
        context_clean = clean_instruction(context)
        context_clean = kv_textify(context_clean)
        new_data.append({
            'question': replaced_question,
            'context': context_clean,
            'answer': item.get('answer', '')
        })
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"处理完成，输出到{output_path}")

# 示例主函数
if __name__ == "__main__":
    process_alphafin_contexts(
        input_path='data/alphafin/alphafin_rag_ready_generated.json',
        output_path='data/alphafin/alphafin_rag_ready_generated_cleaned.json'
    ) 