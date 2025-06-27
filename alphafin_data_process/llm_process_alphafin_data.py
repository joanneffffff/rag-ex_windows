import json
import torch
from tqdm import tqdm
from pathlib import Path
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import textwrap
import sys

def load_model_and_tokenizer(model_name: str, device: str):
    """
    Loads the specified model and tokenizer with 4-bit quantization if on GPU.
    """
    print(f"Loading model: {model_name}...")
    
    bnb_config = None
    if device == "cuda":
        print("CUDA device detected. Applying 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Model {model_name} loaded successfully on device: {model.device}")
    return model, tokenizer

def generate_questions(
    input_path: Path,
    output_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    limit: int,
    save_interval: int,
):
    """
    Main function to generate questions for the AlphaFin dataset.
    """
    # --- 1. Load Model ---
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, device)
    except Exception as e:
        print(f"Fatal: Could not load the model. Error: {e}")
        return

    # --- 2. Load Data ---
    print(f"Loading cleaned data from {input_path}...")
    if not input_path.exists():
        print(f"Fatal: Input file not found. Please create '{input_path.name}' first.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)

    if limit:
        print(f"Limiting processing to {limit} records.")
        cleaned_data = cleaned_data[:limit]

    # --- 3. System Prompt Definition ---
    system_prompt = textwrap.dedent("""
        You are a professional financial analyst. Your task is to create a high-quality Question-Answer pair based on the provided Chinese financial news article. I will provide you with the full article (Context) and a human-written summary (Answer). Your job is to generate one single, specific question that is directly and completely answered by the given 'Answer', using only information found in the 'Context'. The question should be something a financial analyst would ask. Do not answer the question. Only provide the question itself.
    """).strip()

    final_rag_data = []
    
    # --- 4. Generate Questions in Batches ---
    print(f"Generating questions for {len(cleaned_data)} articles in batches of {batch_size}...")
    
    # We use enumerate to get a batch index for periodic saving
    for batch_idx, i in enumerate(tqdm(range(0, len(cleaned_data), batch_size), desc="Generating Questions")):
        batch_records = cleaned_data[i:i+batch_size]
        
        messages_batch = []
        valid_records_in_batch = []
        for record in batch_records:
            context = record.get('context', '').strip()
            answer = record.get('answer', '').strip()
            if context and answer:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"}
                ]
                messages_batch.append(messages)
                valid_records_in_batch.append(record)
        
        if not messages_batch:
            continue

        is_qwen_model = "qwen" in model_name.lower()
        template_args = {"tokenize": False, "add_generation_prompt": True}
        if is_qwen_model:
            template_args["enable_thinking"] = False

        prompts = [
            tokenizer.apply_chat_template(
                conversation=m, **template_args
            ) for m in messages_batch
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": 60, "do_sample": True, "pad_token_id": tokenizer.pad_token_id
        }
        if is_qwen_model:
            gen_kwargs["top_p"] = 0.95; gen_kwargs["temperature"] = 0.6
        else:
            gen_kwargs["top_p"] = 0.9; gen_kwargs["temperature"] = 0.7

        outputs = model.generate(**inputs, **gen_kwargs)

        input_ids_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[:, input_ids_len:]
        decoded_questions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        for idx, generated_text in enumerate(decoded_questions):
            # Robust fix: The model sometimes generates the next turn's prompt.
            # We split by the end-of-turn token and take only the first part,
            # which is the actual generated content.
            if '<|im_end|>' in generated_text:
                question = generated_text.split('<|im_end|>')[0]
            else:
                question = generated_text

            # Also remove any other special tokens that might have been missed
            question = question.replace(tokenizer.eos_token, '').strip()

            if question:
                record = valid_records_in_batch[idx]
                final_rag_data.append({
                    'question': question,
                    'context': record.get('context', '').strip(),
                    'answer': record.get('answer', '').strip()
                })
        
        # --- Periodic Saving ---
        # Save after every `save_interval` batches
        if (batch_idx + 1) % save_interval == 0 and final_rag_data:
            print(f"\n--- Periodic Save: Saving {len(final_rag_data)} records at batch {batch_idx+1} ---")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_rag_data, f, ensure_ascii=False, indent=4)

    # --- 5. Final Save ---
    print(f"\nFinal Save: Saving all {len(final_rag_data)} new RAG-ready records to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_rag_data, f, ensure_ascii=False, indent=4)
        
    print("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate specific questions for AlphaFin articles using an LLM.",
        epilog="Use a small model like Phi-3-mini for testing on CPU."
    )
    
    parser.add_argument("--input_file", type=str, default="data/alphafin/alphafin_rag_ready.json", help="Path to the input cleaned data file.")
    parser.add_argument("--output_file", type=str, default="data/alphafin/alphafin_rag_ready_generated.json", help="Path to the output RAG-ready file.")
    parser.add_argument("--model_name", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="Name of the Hugging Face model to use.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (use a smaller value for CPU).")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to process for testing.")
    parser.add_argument("--save_interval", type=int, default=50, help="Save progress every N batches.")
    
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA specified but not available. Falling back to CPU.")
        args.device = "cpu"

    if args.device == "cpu":
        print("Running on CPU. Consider using a smaller batch size (e.g., --batch_size 1) and a limit (e.g., --limit 10).")

    generate_questions(
        input_path=Path(args.input_file),
        output_path=Path(args.output_file),
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        save_interval=args.save_interval
    ) 