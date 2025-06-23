import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os
import ast
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

# 定义一个简单的 CustomCrossEncoder 类，以便我们可以在compute_mrr_with_reranker中以类似的方式调用predict
class CustomCrossEncoder:
    def __init__(self, model_name, device, max_length=512, trust_remote_code=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # 加载模型配置
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        # 确保tokenizer的padding token设置
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"警告: 自定义Reranker的tokenizer没有定义pad_token，已使用eos_token作为pad_token。")
            elif self.tokenizer.unk_token is not None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
                print(f"警告: 自定义Reranker的tokenizer没有定义pad_token，已使用unk_token作为pad_token。")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"警告: 自定义Reranker的tokenizer没有定义pad_token，已添加新的pad_token '[PAD]'。")
        
        # 将tokenizer的pad_token_id同步到模型config中
        if self.tokenizer.pad_token_id is not None:
            config.pad_token_id = self.tokenizer.pad_token_id
            print(f"将tokenizer的pad_token_id ({self.tokenizer.pad_token_id}) 同步到模型config.pad_token_id。")
        else:
            print("警告: 无法获取tokenizer的pad_token_id，模型可能仍会遇到padding问题。")


        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config, trust_remote_code=trust_remote_code)
        self.device = device
        self.model.to(device)
        self.model.eval() # 设置为评估模式
        self.max_length = max_length

        # 确保tokenizer的padding方向是正确的（通常是右侧填充）
        if self.tokenizer.padding_side != "right":
            self.tokenizer.padding_side = "right"
            print(f"警告: tokenizer的padding_side不是'right'，已将其设置为'right'。")


    @torch.no_grad() # 预测时不需要计算梯度
    def predict(self, sentences, batch_size=32, show_progress_bar=False, convert_to_tensor=True):
        all_scores = []
        iterator = range(0, len(sentences), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Reranker predicting")

        for i in iterator:
            batch_sentences = sentences[i:i + batch_size]
            
            features = self.tokenizer(
                batch_sentences, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**features)
            
            # 关键修改：从 logits 中提取分数
            # outputs.logits 的形状通常是 (batch_size, num_labels)
            # 对于二分类，num_labels通常为2。我们取索引为1的logit作为相关性分数。
            
            # 确保 outputs.logits 至少有一个维度
            if outputs.logits.dim() > 1 and outputs.logits.shape[1] > 1:
                # 尝试选择第二个 logit（通常代表正类别，即相关性）
                scores = outputs.logits[:, 1]
            else:
                # 如果只有一个维度，或者只有一个输出标签，直接使用该输出
                scores = outputs.logits
            
            # 确保 scores 是一个一维张量，方便后续处理
            scores = scores.squeeze()

            if convert_to_tensor:
                all_scores.append(scores)
            else:
                all_scores.extend(scores.cpu().numpy().tolist())

        if convert_to_tensor:
            return torch.cat(all_scores)
        else:
            return all_scores


def get_question_or_query(item_dict):
    if "question" in item_dict:
        return item_dict["question"]
    elif "query" in item_dict:
        return item_dict["query"]
    return None

def convert_json_context_to_natural_language_chunks(json_str_context, company_name="公司"):
    chunks = []
    
    if not json_str_context or not json_str_context.strip():
        return chunks

    # 步骤1：预清理一些常见的前缀和标点，以便更准确地提取字典和纯文本
    cleaned_for_processing = json_str_context.replace("【问题】:", "").replace("【答案】:", "").strip()
    cleaned_for_processing = cleaned_for_processing.replace('，', ',')
    cleaned_for_processing = cleaned_for_processing.replace('：', ':')
    cleaned_for_processing = cleaned_for_processing.replace('。', ',')
    cleaned_for_processing = cleaned_for_processing.replace('【', '')
    cleaned_for_processing = cleaned_for_processing.replace('】', '')
    cleaned_for_processing = cleaned_for_processing.replace('\u3000', ' ')
    cleaned_for_processing = cleaned_for_processing.replace('\xa0', ' ').strip() # 处理 &nbsp; 这样的特殊空格

    # --- NEW: 尝试解析研报格式 (优化后的正则表达式) ---
    # 匹配 '这是以...为题目,在...日期发布的研究报告。研报内容如下: 研报题目是:...'
    # 调整研报内容捕获，使其能更鲁棒地处理后面的内容，包括重复的“研报题目是”
    # 将第三个捕获组从 `(.+)` 调整为更具体地从 `研报题目是:` 之后开始捕获
    report_match = re.match(
        r"这是以(.+?)为题目,在(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?)日期发布的研究报告。研报内容如下: (.+)", 
        cleaned_for_processing, 
        re.DOTALL
    )
    
    if report_match:
        report_title_full = report_match.group(1).strip()
        report_date = report_match.group(2).strip()
        report_raw_content = report_match.group(3).strip() # 捕获到原始的“研报题目是”部分

        # 从 report_raw_content 中再次提取实际的“研报题目是”后面的内容
        content_after_second_title_match = re.match(r"研报题目是:(.+)", report_raw_content, re.DOTALL)
        if content_after_second_title_match:
            report_content_preview = content_after_second_title_match.group(1).strip()
        else:
            report_content_preview = report_raw_content # 如果没有第二个“研报题目是:”，则使用捕获到的全部内容

        # 进一步从 report_title_full 中提取公司名称和股票代码，如果存在的话
        company_stock_match = re.search(r"(.+?)（(\d{6}\.\w{2})）", report_title_full)
        company_info = ""
        if company_stock_match:
            report_company_name = company_stock_match.group(1).strip()
            report_stock_code = company_stock_match.group(2).strip()
            company_info = f"，公司名称：{report_company_name}，股票代码：{report_stock_code}"
            # 移除标题中的公司信息，只保留主题
            report_title_main = re.sub(r"（\d{6}\.\w{2}）", "", report_title_full).strip()
        else:
            report_title_main = report_title_full

        chunk_text = f"一份发布日期为 {report_date} 的研究报告，其标题是：“{report_title_main}”{company_info}。报告摘要内容：{report_content_preview.rstrip('...') if report_content_preview.endswith('...') else report_content_preview}。"
        chunks.append(chunk_text)
        return chunks # 如果匹配到研报格式，直接返回

    # --- 优先尝试字典解析 (与之前的逻辑相同) ---
    extracted_dict_str = None
    parsed_data = None 

    temp_dict_search_str = re.sub(r"Timestamp\(['\"](.*?)['\"]\)", r"'\1'", cleaned_for_processing)
    all_dict_matches = re.findall(r"(\{.*?\})", temp_dict_search_str, re.DOTALL) 

    for potential_dict_str in all_dict_matches:
        cleaned_potential_dict_str = potential_dict_str.strip()
        
        # 尝试 json.loads
        json_compatible_str_temp = cleaned_potential_dict_str.replace("'", '"')
        try:
            parsed_data_temp = json.loads(json_compatible_str_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break 
        except json.JSONDecodeError:
            pass 

        # 尝试 ast.literal_eval
        fixed_for_ast_eval_temp = re.sub(
            r"(?<!['\"\w.])\b(0[1-9]\d*)\b(?![\d.]|['\"\w.])", 
            r"'\1'", 
            cleaned_potential_dict_str
        )
        try:
            parsed_data_temp = ast.literal_eval(fixed_for_ast_eval_temp)
            if isinstance(parsed_data_temp, dict):
                extracted_dict_str = cleaned_potential_dict_str
                parsed_data = parsed_data_temp
                break 
        except (ValueError, SyntaxError):
            pass 

    if extracted_dict_str is not None and isinstance(parsed_data, dict):
        # 如果成功解析出字典，则按字典方式处理
        for metric_name, time_series_data in parsed_data.items():
            if not isinstance(metric_name, str):
                metric_name = str(metric_name)

            cleaned_metric_name = re.sub(r'（.*?）', '', metric_name).strip()
            
            if not isinstance(time_series_data, dict):
                if time_series_data is not None and str(time_series_data).strip():
                    chunks.append(f"{company_name}的{cleaned_metric_name}数据为：{time_series_data}。")
                continue
            if not time_series_data:
                continue
            
            try:
                sorted_dates = sorted(time_series_data.keys(), key=str)
            except TypeError: 
                sorted_dates = [str(k) for k in time_series_data.keys()]
                
            description_parts = []
            for date in sorted_dates:
                value = time_series_data[date]
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}".rstrip('0').rstrip('.') if isinstance(value, float) else str(value)
                else:
                    formatted_value = str(value)
                description_parts.append(f"在{date}为{formatted_value}")
            
            if description_parts:
                if len(description_parts) <= 3:
                    full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                else:
                    first_part = "，".join(description_parts[:3])
                    last_part = "，".join(description_parts[-3:])
                    if len(sorted_dates) > 6:
                        full_description = f"{company_name}的{cleaned_metric_name}数据从{sorted_dates[0]}到{sorted_dates[-1]}，主要变化为：{first_part}，...，{last_part}。"
                    else:
                         full_description = f"{company_name}的{cleaned_metric_name}数据: " + "，".join(description_parts) + "。"
                chunks.append(full_description)
    else:
        # 如果没有成功解析出字典，也没有匹配到研报格式，则假定是普通纯文本内容
        # 移除可能存在的日期前缀 (例如 "2022-04-27_;" 或 "2023-08-14 16:48:44_;")
        pure_text = re.sub(r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?[_;]?", "", cleaned_for_processing, 1).strip()
        
        # 进一步清理可能存在的像 "华尔街见闻/严婷2015年05月25日23:19:02" 这种更复杂的前缀
        # 匹配开头是中文、斜杠、日期、时间、中文、逗号的模式
        pure_text = re.sub(r"^[\u4e00-\u9fa5]+(?:/[\u4e00-\u9fa5]+)?\d{4}年\d{2}月\d{2}日\d{2}:\d{2}:\d{2}(?:据[\u4e00-\u9fa5]+?,)?\d{1,2}月\d{1,2}日,?", "", pure_text).strip()

        # 处理像 "市场资金进出截至周一收盘," 这样的头部
        pure_text = re.sub(r"^(?:市场资金进出)?截至周[一二三四五六日]收盘,?", "", pure_text).strip()

        # 处理像 "渤海活塞中期净利预减60%-70%渤海活塞7月24日晚间公告," 这样的头部
        pure_text = re.sub(r"^[\u4e00-\u9fa5]+?中期净利预减\d+%-?\d*%(?:[\u4e00-\u9fa5]+?\d{1,2}月\d{1,2}日晚间公告,)?", "", pure_text).strip()


        if pure_text: 
            chunks.append(pure_text)
        else:
            print(f"警告：未能在 context 字符串中找到有效结构 (字典、研报或纯文本)。原始字符串（前100字符）：{json_str_context[:100]}...")
            chunks.append(f"{company_name}的相关数据（原始格式，无有效结构）：{json_str_context.strip()}")

    return chunks

def build_corpus_from_alphafin_contexts(alphafin_qca_file_path):
    all_chunks = []
    seen_contexts = set()
    
    try:
        with open(alphafin_qca_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：文件 '{alphafin_qca_file_path}' 不是有效的 JSON 格式。")
        return []
    except Exception as e:
        print(f"读取文件时发生未知错误：{e}")
        return []

    if not isinstance(data, list):
        data = [data]

    chunk_counter = 0
    for item in tqdm(data, desc=f"从 '{Path(alphafin_qca_file_path).name}' 的 context 字段构建语料库"):
        if not isinstance(item, dict) or "context" not in item:
            print(f"警告：文件 {Path(alphafin_qca_file_path).name} 中发现无效项，跳过。项内容：{item}")
            continue

        original_json_context_str = item.get('context', '')
        company_name = item.get('stock_name', '公司')
        
        natural_language_chunks = convert_json_context_to_natural_language_chunks(
            original_json_context_str, company_name=company_name
        )
        
        is_parse_failed_chunk = False
        if not natural_language_chunks: 
            is_parse_failed_chunk = True
        elif len(natural_language_chunks) == 1 and ("原始格式，无有效字典" in natural_language_chunks[0] or "原始格式，解析失败" in natural_language_chunks[0] or "原始格式，无有效结构" in natural_language_chunks[0]):
            is_parse_failed_chunk = True

        if not is_parse_failed_chunk:
            for nl_chunk_text in natural_language_chunks:
                if nl_chunk_text and nl_chunk_text not in seen_contexts:
                    seen_contexts.add(nl_chunk_text)
                    all_chunks.append({
                        "doc_id": f"doc_{chunk_counter}",
                        "chunk_id": f"chunk_{chunk_counter}",
                        "text": nl_chunk_text,
                        "source_type": "alphafin_context_nl"
                    })
                    chunk_counter += 1
        else:
            pass 


    return all_chunks


def compute_mrr_with_reranker(encoder_model, reranker_model, eval_data, corpus_chunks, top_k_retrieval=100, top_k_rerank=10):
    query_texts = [get_question_or_query(item) for item in eval_data]
    correct_chunk_contents = [item["context"] for item in eval_data]

    processed_correct_chunk_contents = []
    for i, original_context_str in enumerate(correct_chunk_contents):
        company_name = eval_data[i].get('stock_name', '公司')
        nl_chunks = convert_json_context_to_natural_language_chunks(original_context_str, company_name)
        processed_correct_chunk_contents.append("\n\n".join(nl_chunks))


    print("编码评估查询 (Queries) with Encoder...")
    query_embeddings = encoder_model.encode(query_texts, show_progress_bar=True, convert_to_tensor=True)
    
    print("编码检索库上下文 (Chunks) with Encoder...")
    corpus_texts = [chunk["text"] for chunk in corpus_chunks]
    corpus_embeddings = encoder_model.encode(corpus_texts, show_progress_bar=True, convert_to_tensor=True)

    mrr_scores = []
    
    print(f"Evaluating MRR with Reranker (Retrieval Top-{top_k_retrieval}, Rerank Top-{top_k_rerank}):")
    for i, query_emb in tqdm(enumerate(query_embeddings), total=len(query_embeddings)):
        current_query_text = query_texts[i]
        expected_retrieved_chunk_text = processed_correct_chunk_contents[i] 

        if not expected_retrieved_chunk_text or "原始格式，无有效字典" in expected_retrieved_chunk_text or "原始格式，解析失败" in expected_retrieved_chunk_text or "原始格式，无有效结构" in expected_retrieved_chunk_text:
            mrr_scores.append(0) 
            continue

        cos_scores = util.cos_sim(query_emb, corpus_embeddings)[0]
        top_retrieved_indices = torch.topk(cos_scores, k=min(top_k_retrieval, len(corpus_chunks)))[1].tolist()
        
        reranker_input_pairs = []
        retrieved_chunks_for_rerank = [] 
        for idx in top_retrieved_indices:
            chunk = corpus_chunks[idx]
            reranker_input_pairs.append([current_query_text, chunk["text"]])
            retrieved_chunks_for_rerank.append(chunk)

        if not reranker_input_pairs:
            mrr_scores.append(0)
            continue

        # 使用我们自定义的 CustomCrossEncoder 实例进行预测
        reranker_scores = reranker_model.predict(
            reranker_input_pairs, 
            show_progress_bar=False, 
            convert_to_tensor=True,
        )


        scored_retrieved_chunks_with_scores = []
        for j, score in enumerate(reranker_scores):
            scored_retrieved_chunks_with_scores.append({"chunk_text": retrieved_chunks_for_rerank[j]["text"], "score": score.item()})

        scored_retrieved_chunks_with_scores.sort(key=lambda x: x["score"], reverse=True)

        found_rank = -1
        for rank, item in enumerate(scored_retrieved_chunks_with_scores[:top_k_rerank]):
            if item["chunk_text"] == expected_retrieved_chunk_text:
                found_rank = rank + 1
                break
        
        if found_rank != -1:
            mrr_scores.append(1 / found_rank)
        else:
            mrr_scores.append(0)

    return np.mean(mrr_scores)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_model_name", type=str, required=True, help="Path to the fine-tuned Encoder model.")
    parser.add_argument("--reranker_model_name", type=str, default="cross-encoder/ms-marco-MiniLM-L-12-v2", help="Path to the Reranker model or Hugging Face ID.")
    parser.add_argument("--eval_jsonl", type=str, required=True, help="Path to the evaluation JSONL file.")
    parser.add_argument("--base_raw_data_path", type=str, default="data/alphafin/alphafin_rag_ready_generated_cleaned.json", help="Path to the JSON file containing all contexts for Chinese corpus building.")
    parser.add_argument("--top_k_retrieval", type=int, default=100, help="Number of top documents to retrieve from Encoder before reranking.")
    parser.add_argument("--top_k_rerank", type=int, default=10, help="Number of top documents to consider after reranking for MRR calculation.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    print(f"加载 Encoder 模型: {args.encoder_model_name}")
    try:
        encoder_model = SentenceTransformer(args.encoder_model_name) 
        encoder_model.to(device)
        print("Encoder 模型加载成功。")
    except Exception as e:
        print(f"加载 Encoder 模型失败。错误信息: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"加载 Reranker 模型: {args.reranker_model_name}")
    try:
        # 使用我们自定义的 CustomCrossEncoder 类
        reranker_model = CustomCrossEncoder(
            args.reranker_model_name, 
            device=device,
            max_length=512, # 建议值，可以根据你的数据和模型最大输入长度调整
            trust_remote_code=True # 对于某些模型如Qwen系列、自定义模型可能需要
        )
        print("Reranker 模型加载成功。")

    except Exception as e:
        print(f"加载 Reranker 模型失败。错误信息: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"加载评估数据：{args.eval_jsonl}")
    raw_eval_data = []
    with open(args.eval_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            raw_eval_data.append(json.loads(line))
    
    eval_data = []
    for item in raw_eval_data:
        is_valid_item = isinstance(item, dict) and "context" in item and \
                        (get_question_or_query(item) is not None)
        
        if is_valid_item:
            eval_data.append(item) 
        else:
            print(f"警告：评估数据中发现无效样本，跳过。样本内容：{item}")
    print(f"加载了 {len(eval_data)} 个有效评估样本。 (共 {len(raw_eval_data)} 个原始样本)")

    if not eval_data:
        print("错误：没有找到任何有效的评估样本，无法计算 MRR。")
        return

    print(f"从中文 AlphaFin QCA 文件构建检索语料库：{args.base_raw_data_path}")
    corpus_chunks = build_corpus_from_alphafin_contexts(args.base_raw_data_path)
    print(f"构建了 {len(corpus_chunks)} 个 Chunk 作为检索库。")

    if not corpus_chunks:
        print("错误：构建的检索语料库为空，无法进行评估。")
        return

    mrr = compute_mrr_with_reranker(encoder_model, reranker_model, eval_data, corpus_chunks, args.top_k_retrieval, args.top_k_rerank)
    print(f"\nMean Reciprocal Rank (MRR) with Reranker: {mrr:.4f}")

if __name__ == "__main__":
    main()