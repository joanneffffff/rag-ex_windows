import json
import re
from tqdm import tqdm
from pathlib import Path
import os

# 辅助函数保持不变
def extract_info_from_context(context):
    company = None
    stock_code = None
    date = None

    # 尝试匹配 "这是以公司名（股票代码）" 或 "这是以公司名"
    # 新增对 "这是以..." 模式的优先匹配
    m = re.match(r"^这是以([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+)[（(](\d{6}(?:\.SZ|\.SH)?)[)）]", context)
    if m:
        company = m.group(1).strip()
        stock_code = m.group(2)
        if re.fullmatch(r"\d{6}", stock_code): # 如果只有六位数字，尝试在上下文找完整的代码
            m_full = re.search(rf"{stock_code}\.(SZ|SH)", context)
            if m_full:
                stock_code = f"{stock_code}.{m_full.group(1)}"
        return company, stock_code, date

    m = re.match(r"^这是以([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+)", context)
    if m:
        company = m.group(1).strip()
        # 即使只匹配到公司名，也尝试在剩余context中查找股票代码
        remaining_context = context[m.end():]
        m_code = re.search(r"(\d{6}\.(SZ|SH))", remaining_context)
        if m_code:
            stock_code = m_code.group(1)
        else:
            m_code2 = re.search(r"[（(](\d{6})[)）]", remaining_context)
            if m_code2:
                stock_code = m_code2.group(1)
            elif re.search(r"\d{6}", remaining_context): # 尝试匹配裸露的6位数字
                raw_code_match = re.search(r"(\d{6})", remaining_context)
                if raw_code_match:
                    stock_code = raw_code_match.group(1)
                    # 再次尝试补全 .SZ/.SH
                    if re.fullmatch(r"\d{6}", stock_code):
                        m_full = re.search(rf"{stock_code}\.(SZ|SH)", context) # 从整个context找，更保险
                        if m_full:
                            stock_code = f"{stock_code}.{m_full.group(1)}"
        return company, stock_code, date # 返回已提取的信息，不再继续后面的通用逻辑


    # 原始逻辑（作为 fallback，处理非 "这是以..." 开头的上下文）
    m = re.match(r"^以下数据是([\u4e00-\u9fa5A-Za-z0-9·（）()\-]+)", context)
    if m:
        company_full = m.group(1).strip()
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
    
    if not stock_code:
        m_code = re.search(r"(\d{6}\.(SZ|SH))", context)
        if m_code:
            stock_code = m_code.group(1)
        else:
            m_code2 = re.search(r"[（(](\d{6})[)）]", context)
            if m_code2:
                stock_code = m_code2.group(1)
            elif re.search(r"\d{6}", context): # 尝试匹配裸露的6位数字
                raw_code_match = re.search(r"(\d{6})", context)
                if raw_code_match:
                    stock_code = raw_code_match.group(1)
                    if re.fullmatch(r"\d{6}", stock_code): # 再次尝试补全 .SZ/.SH
                        m_full = re.search(rf"{stock_code}\.(SZ|SH)", context)
                        if m_full:
                            stock_code = f"{stock_code}.{m_full.group(1)}"

    m_date = re.search(r"时间为(\d{4}-\d{2}-\d{2})", context)
    if m_date:
        date = m_date.group(1)
    else:
        m_date2 = re.search(r"(\d{4}-\d{2}-\d{2})", context)
        if m_date2:
            date = m_date2.group(1)
            
    return company, stock_code, date


def clean_instruction(text):
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
    if isinstance(context, dict):
        return '；'.join([f"{k}: {v}" for k, v in context.items()])
    try:
        ctx = json.loads(context)
        if isinstance(ctx, dict):
            return '；'.join([f"{k}: {v}" for k, v in ctx.items()])
    except Exception:
        pass
    return context

# --- 改进后的 replace_company_in_question 函数 ---
def replace_company_in_question(question, company, stock_code=None):
    if not company:
        return question

    company_full = f"{company}（{stock_code}）" if stock_code else company
    original_question_lower = question.lower()
    current_question = question

    # 1. 检查问题是否已明确提及公司或股票代码 (优先级最高，直接返回)
    if company.lower() in original_question_lower or \
       (stock_code and stock_code.lower() in original_question_lower):
        return question

    # 2. 尝试替换模糊指代词，并避免重复（关键改进）
    # 模糊词列表，现在包含更多可能的模糊指代
    fuzzy_words_to_replace = [
        "该股票", "这个股票", "这只股票", "该股", "该公司",
        "公司", "股票" # "公司", "股票" 放后面，避免在“什么公司”中误替换
    ]
    # 排序，长的词优先匹配替换，避免“该公司”被“公司”先匹配
    fuzzy_words_to_replace.sort(key=len, reverse=True)

    replaced_in_this_step = False # 标记此步骤是否有替换发生

    for word in fuzzy_words_to_replace:
        # 使用负向先行/后行断言，确保替换的是独立的模糊词，而不是其他词的一部分
        # 允许模糊词后面直接是中文标点或空格
        # 例如：避免替换“上市公司”中的“公司”，但允许替换“这个股票的”
        pattern = rf'{re.escape(word)}(?=[，。！？,?!.\s]|$|\w)' # 匹配词语后是标点、空格、行尾或任意单词字符 (为了“股票的”这种)
        
        # 改进：如果模糊词是“公司”或“股票”，并且公司名本身包含这些词，
        # 并且原始问题中直接出现了模糊词，则替换
        if word in ["公司", "股票"] and word in company:
            # 如果公司名是“XX公司”，而原始问题是“该公司”，
            # 那么“该公司”会被替换成“XX公司”，而不是“XX公司公司”。
            # 这里是为了处理 "云南白药" -> "云南白药的股票" 这种，如果原始问题就是"这个股票"
            # 那么 "云南白药" 的替换就应该直接替换掉 "这个股票"
            # 为了确保精确替换，我们仍然使用 re.subn，并且只替换第一个匹配项
            pass # 沿用下面的通用替换逻辑
            
        new_question, num_subs = re.subn(pattern, company_full, current_question) # 替换所有匹配项
        if num_subs > 0:
            current_question = new_question
            replaced_in_this_step = True
    
    if replaced_in_this_step:
        # 额外清理：如果替换后出现“公司公司”、“股票股票”这种因为公司名本身带“公司”字样造成的重复，尝试去除
        # 例如：“XX公司（代码）公司” -> “XX公司（代码）”
        # 这里需要更精细，确保只移除紧跟在公司名后的冗余词
        # 注意：这里 company_full 可能包含括号，需要正确转义
        escaped_company_full = re.escape(company_full)
        current_question = re.sub(rf"({escaped_company_full})\s*(公司|股票)", r"\1", current_question)
        return current_question

    # 3. 如果问题是通用性质，且没有明确提及公司或模糊指代词，则智能地插入公司信息
    
    # 优先处理“如何计算X？”这类特定模式
    match_how_to_calculate = re.search(r"(如何计算)(.+?)(\?|？)", original_question_lower)
    if match_how_to_calculate:
        prefix = match_how_to_calculate.group(1)
        metric = match_how_to_calculate.group(2).strip()
        suffix = match_how_to_calculate.group(3)
        return f"{company_full}的{metric}{prefix}{suffix}"

    # 其次处理“X是多少？”这类模式
    match_is_what = re.search(r"(.+?)(是多少|为多少)(\?|？)", original_question_lower)
    if match_is_what:
        subject = match_is_what.group(1).strip()
        verb = match_is_what.group(2)
        suffix = match_is_what.group(3)
        return f"{company_full}的{subject}{verb}{suffix}"

    # 再次，处理“什么是X？”这类模式
    match_what_is = re.search(r"(什么是)(.+?)(\?|？)", original_question_lower)
    if match_what_is:
        prefix = match_what_is.group(1)
        concept = match_what_is.group(2).strip()
        suffix = match_what_is.group(3)
        return f"关于{company_full}，{prefix}{concept}{suffix}"
    
    # 处理你提供的 "2022年1月份股价预测如何?" 这种预测结果描述
    # 扩展的预测模式，能够匹配更通用的描述
    # 匹配 "X的[时间]预测结果是..." 或者 "X的[时间][指标]预测如何?"
    match_result_description = re.search(r"(的)(下月|下一个季度|本月|本季度|今年|明年|未来|20\d{2}年\d{1,2}月份)?(.+?)(最终收益结果是|表现是|走势是|预测是|预测如何|怎么样|表现如何)(.+)", original_question_lower)
    if match_result_description:
        # 如果原始问题中包含公司名或股票代码（已在最开始排除），则不进入此分支
        # 如果不包含，且匹配到这类描述性句式
        # 例如："这个股票的下月最终收益结果是:'跌',下跌概率:极大"
        # 目标是 "云南白药（代码）的下月最终收益结果是:'跌',下跌概率:极大"
        # 找到 "的" 字在原始问题中的位置，并在此之前插入公司名
        idx_of_de = original_question_lower.find('的')
        if idx_of_de != -1:
            # 确保 '的' 前面是模糊词，而不是其他名词
            pre_de_text = original_question_lower[:idx_of_de].strip()
            if pre_de_text in [w.lower() for w in fuzzy_words_to_replace]: # 如果"的"前面是模糊词
                # 替换模糊词，并保留“的”及后面的内容
                # 例如 "这个股票的" -> "明泰铝业（601677）的"
                q_remainder = question[idx_of_de:]
                return f"{company_full}{q_remainder}"
            else: # 如果前面不是模糊词，那么直接在句首添加
                 return f"{company_full}{question}"
        else: # 如果没有“的”字，直接句首添加
            return f"{company_full}{question}"

    # 其他通用问题，默认在句首添加（或根据上下文优化）
    general_question_patterns_fallback = [
        r"描述一下", r"解释一下", r"有哪些", r"分析一下", r"定义", r"的业务范围", r"的策略", r"如何" # "如何" 泛化处理
    ]
    if any(re.search(pattern, original_question_lower) for pattern in general_question_patterns_fallback):
        # 针对"描述一下XXX的业务范围" 这类问题进行优化
        match_describe_scope = re.search(r"(描述一下)(.*?)(的业务范围)(\?|？)", original_question_lower)
        if match_describe_scope:
            return f"{company_full}的业务范围如何描述？"
        
        # 默认情况下，在句首添加公司名
        return f"{company_full}的{question}"
            
    return question


# process_alphafin_data 函数（保持不变）
def process_alphafin_data():
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

    input_file = 'data/alphafin/data.json'
    rag_ready_file = 'data/alphafin/alphafin_rag_ready.json'

    output_dir = Path(rag_ready_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

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

# --- process_alphafin_contexts 函数 ---
def process_alphafin_contexts(input_path, output_path):
    print(f"开始处理文件：{input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    new_data = []
    
    for i, item in enumerate(tqdm(data, desc="处理问题并添加公司信息")):
        original_question = item.get('question', '')
        context = item.get('context', '')
        
        company, stock_code, date = extract_info_from_context(context)

        question_after_replacement = replace_company_in_question(original_question, company, stock_code)
        
        print(f"公司名: {company or '未提取'}")
        print(f"原始问题: {original_question}")
        print(f"替换后问题: {question_after_replacement}")
        print("---")
        
        context_clean = clean_instruction(context)
        context_clean = kv_textify(context_clean)
        
        new_data.append({
            'question': question_after_replacement,
            'context': context_clean,
            'answer': item.get('answer', '')
        })
    
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"\n处理完成，输出到：{output_path}")

# --- 主执行块 ---
if __name__ == "__main__":
    # 可以添加测试用例来验证你的新逻辑
    def run_specific_tests():
        test_cases = [
            ("这个股票的下月最终收益结果是:'跌',下跌概率:极大", "云南白药", None, "云南白药的下月最终收益结果是:'跌',下跌概率:极大"),
            ("这个股票的下一个季度的最终收益结果是:涨", "酒鬼酒", None, "酒鬼酒的下一个季度的最终收益结果是:涨"),
            ("该公司的财务状况如何？", "腾讯", None, "腾讯的财务状况如何？"), # 期望：避免“腾讯公司的财务状况如何”
            ("该公司2023年营收如何？", "格力电器", None, "格力电器2023年营收如何？"), # 期望：避免“格力电器公司2023年营收如何”
            ("明泰铝业（601677）2022年1月份股价预测如何?", "明泰铝业", "601677", "明泰铝业（601677）2022年1月份股价预测如何?"), # 已经包含，不应改变
            ("该公司最近表现如何？", "贵州茅台", "600519.SH", "贵州茅台（600519.SH）最近表现如何？")
        ]

        print("--- 特定问题场景测试开始 ---")
        for i, (q_raw, comp, code, expected_q) in enumerate(test_cases):
            replaced_q = replace_company_in_question(q_raw, comp, code)
            print(f"\n===== Test Case {i+1} =====")
            print(f"  公司名: {comp}")
            print(f"  原始问题: {q_raw}")
            print(f"  期望输出: {expected_q}")
            print(f"  实际输出: {replaced_q}")
            print(f"  匹配期望: {replaced_q == expected_q}")
            print("=========================")
    
    run_specific_tests()

    # 正式处理数据文件
    # process_alphafin_data() 
    process_alphafin_contexts(
        input_path='data/alphafin/alphafin_rag_ready_generated.json',
        output_path='data/alphafin/alphafin_rag_ready_generated_cleaned.json'
    )