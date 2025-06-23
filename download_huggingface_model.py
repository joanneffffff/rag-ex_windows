from huggingface_hub import snapshot_download
import os

def download_hf_model(repo_id: str, local_dir: str, allow_patterns=None, ignore_patterns=None):
    """
    从 Hugging Face Hub 下载指定模型的所有文件。

    Args:
        repo_id (str): 模型在 Hugging Face Hub 上的 ID，例如 "ProsusAI/finbert"。
        local_dir (str): 模型文件将被下载到的本地目录路径。
        allow_patterns (list, optional): 只下载匹配这些模式的文件。默认为 None (下载所有文件)。
        ignore_patterns (list, optional): 忽略匹配这些模式的文件。默认为 None (不忽略任何文件)。
    """
    print(f"正在尝试下载模型 '{repo_id}' 到 '{local_dir}'...")
    
    # 确保本地目录存在
    os.makedirs(local_dir, exist_ok=True)

    try:
        # 使用 snapshot_download 函数下载模型
        # 该函数会下载模型的所有版本控制文件，并返回本地路径
        # local_dir_use_symlinks=False 确保下载的是实际文件而非符号链接，在某些文件系统可能更稳健
        # force_download=True 强制重新下载，即使文件已存在于缓存中
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir_use_symlinks=False,
            # force_download=True # 如果需要强制重新下载，可以取消注释此行
        )
        print(f"模型 '{repo_id}' 已成功下载到: {downloaded_path}")
        return downloaded_path
    except Exception as e:
        print(f"下载模型 '{repo_id}' 失败。错误信息: {e}")
        return None

if __name__ == "__main__":
    # --- 配置您要下载的模型和目标目录 ---
    model_to_download = "ProsusAI/finbert" 
    # 建议使用一个全新的、您有写入权限的目录，避免与旧的Hugging Face缓存冲突
    target_local_directory = "/users/sgjfei3/data/manually_downloaded_models/finbert" 
    
    # --- 调用下载函数 ---
    downloaded_model_path = download_hf_model(model_to_download, target_local_directory)

    if downloaded_model_path:
        print("\n下载完成。现在您可以将您的评估和微调脚本中的 --model_name 指向此路径:")
        print(f"例如：python evaluate_encoder_mrr.py --model_name {downloaded_model_path} --eval_jsonl ...")
        print(f"例如：python finetune_encoder.py --model_name {downloaded_model_path} --train_jsonl ...")
    else:
        print("\n模型下载失败，请检查网络连接、目标目录权限或模型ID是否正确。")