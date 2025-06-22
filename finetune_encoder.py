import json
import argparse
import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers.evaluation import SentenceEvaluator

# --- 数据加载函数 ---
def load_qc_pairs(jsonl_path, max_samples=None):
    """加载训练用的 Q-C 对"""
    pairs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                q_raw = item['query'] if 'query' in item else item.get('question', '')
                q = str(q_raw).strip() 
                
                c_raw = item.get('context', '')
                c = str(c_raw).strip()
                
                if q and c:
                    pairs.append(InputExample(texts=[q, c], label=1.0))
                else:
                    print(f"警告：训练数据行 {i+1}，因为 'query'/'question' 或 'context' 为空或无效。原始行: {line.strip()}")
            except json.JSONDecodeError:
                print(f"错误：跳过训练数据行 {i+1}，非法的 JSON 格式: {line.strip()}")
            except Exception as e:
                print(f"处理训练数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
    return pairs

def load_qca_for_eval(jsonl_path, max_samples=None):
    """加载评估用的 Q-C-A 数据，供 MRREvaluator 使用"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line)
                q_raw = item['query'] if 'query' in item else item.get('question', '')
                q = str(q_raw).strip() 
                
                c_raw = item.get('context', '')
                c = str(c_raw).strip()
                
                a_raw = item.get('answer', '')
                a = str(a_raw).strip() 

                if q and c: # 确保问题和上下文都非空
                    data.append({'query': q, 'context': c, 'answer': a})
                else:
                    print(f"警告：评估数据行 {i+1}，因为 'query'/'question' 或 'context' 为空或无效。原始行: {line.strip()}")
            except json.JSONDecodeError:
                print(f"错误：跳过评估数据行 {i+1}，非法的 JSON 格式: {line.strip()}")
            except Exception as e:
                print(f"处理评估数据行 {i+1} 时发生未知错误: {e} - 原始行: {line.strip()}")
    return data

# --- MRR 评估器类 ---
class MRREvaluator(SentenceEvaluator):
    """
    对给定的 (query, context, answer) 数据集计算 Mean Reciprocal Rank (MRR)。
    假设每个 item['context'] 是其 item['query'] 的正确上下文。
    """
    def __init__(self, dataset, name='', show_progress_bar=False, write_csv=True):
        self.dataset = dataset
        self.name = name
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv

        self.queries = [item['query'] for item in dataset]
        self.contexts = [item['context'] for item in dataset]
        self.answers = [item['answer'] for item in dataset] 

        self.csv_file: str = None
        self.csv_headers = ["epoch", "steps", "MRR"]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if self.write_csv:
                self.csv_file = os.path.join(output_path, self.name + "_mrr_evaluation_results.csv")
                if not os.path.isfile(self.csv_file) or epoch == 0:
                    with open(self.csv_file, newline="", mode="w", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(self.csv_headers)
                        
        print(f"\n--- 开始 MRR 评估 (Epoch: {epoch}, Steps: {steps}) ---")

        if not self.dataset:
            print("警告：评估数据集为空，MRR为0。")
            mrr = 0.0
        else:
            print(f"编码 {len(self.contexts)} 个评估上下文...")
            context_embeddings = model.encode(self.contexts, batch_size=64, convert_to_tensor=True,
                                              show_progress_bar=self.show_progress_bar)

            mrrs = []
            iterator = tqdm(self.dataset, desc='评估 MRR', disable=not self.show_progress_bar)
            for i, item in enumerate(iterator):
                query_emb = model.encode(item['query'], convert_to_tensor=True)
                scores = util.cos_sim(query_emb, context_embeddings)[0].cpu().numpy()

                target_context_idx = i 

                sorted_indices = np.argsort(scores)[::-1]
                
                rank = -1
                for r, idx in enumerate(sorted_indices):
                    if idx == target_context_idx:
                        rank = r + 1
                        break
                
                if rank != -1:
                    mrr_score = 1.0 / rank
                    mrrs.append(mrr_score)
                else:
                    mrrs.append(0.0) 
            
            mrr = np.mean(mrrs) if mrrs else 0.0 

        print(f"MRR (Epoch: {epoch}, Steps: {steps}): {mrr:.4f}")
        print(f"--- MRR 评估结束 ---")

        if output_path is not None and self.write_csv:
            with open(self.csv_file, newline="", mode="a", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, steps, round(mrr, 4)])

        return mrr

# --- 主微调函数 ---
def finetune_encoder(model_name, train_jsonl, eval_jsonl, output_dir, batch_size=32, epochs=1, max_samples=None, eval_steps=500):
    print(f"正在加载训练数据：{train_jsonl}")
    train_examples = load_qc_pairs(train_jsonl, max_samples)
    if not train_examples:
        print("警告：没有加载到有效的训练样本，请检查数据文件和键名。")
        return

    print(f"正在加载评估数据：{eval_jsonl}")
    eval_data = load_qca_for_eval(eval_jsonl, max_samples) 
    if not eval_data:
        print("警告：没有加载到有效的评估样本，无法进行MRR评估。")
        evaluator = None
    else:
        print(f"加载了 {len(eval_data)} 个评估样本。")
        evaluator = MRREvaluator(dataset=eval_data, name='mrr_eval', show_progress_bar=True)


    print(f"加载模型：{model_name}")
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"错误：无法加载 SentenceTransformer 模型 '{model_name}'。请确保模型路径正确或模型ID有效。")
        print(f"详细错误信息: {e}")
        return

    print(f"训练样本数量：{len(train_examples)}")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model) 
    
    print(f"开始模型微调，共 {epochs} 个 epoch...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], 
        epochs=epochs,
        warmup_steps=100,
        evaluator=evaluator,          
        evaluation_steps=eval_steps,  
        output_path=output_dir,       
        show_progress_bar='batches'   
    )
    
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"模型已保存到：{output_dir}")

# --- 主执行块 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='预训练模型名或路径')
    parser.add_argument('--train_jsonl', type=str, required=True, help='训练数据jsonl文件路径')
    parser.add_argument('--eval_jsonl', type=str, required=True, help='评估数据jsonl文件路径') # <--- 这一行必须存在
    parser.add_argument('--output_dir', type=str, required=True, help='微调模型输出目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_samples', type=int, default=None, help='训练和评估的最大样本数')
    parser.add_argument('--eval_steps', type=int, default=0, help='每隔多少训练步评估一次 (0表示只在每个epoch结束时评估)') # <--- 这一行也必须存在
    
    args = parser.parse_args()

    # 执行模型微调
    finetune_encoder(
        model_name=args.model_name,
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_samples=args.max_samples,
        eval_steps=args.eval_steps
    )