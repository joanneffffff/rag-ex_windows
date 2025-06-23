from transformers import AutoModel, AutoTokenizer
import os

# 这是您手动下载模型到的路径，请确保与您实际下载的路径一致
model_local_path = "/mnt/data1/users/sgjfei3/manually_downloaded_models/finbert" 

print(f"尝试从本地路径加载模型和分词器: {model_local_path}")

try:
    # 尝试加载模型
    model = AutoModel.from_pretrained(model_local_path)
    print("模型加载成功！")

    # 尝试加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_local_path)
    print("分词器加载成功！")

    print("本地模型和分词器均可被 'transformers' 库识别。")

except Exception as e:
    print(f"从本地路径加载模型失败。错误信息: {e}")
    import traceback
    traceback.print_exc()

print("测试完成。")