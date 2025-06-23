import os
from sentence_transformers import SentenceTransformer

# 尝试设置 Hugging Face 缓存目录 (可选，如果遇到权限问题或想指定位置)
# 请替换为您有写入权限的真实路径，例如 '/tmp/hf_cache' (Linux/macOS) 或 'C:\\temp\\hf_cache' (Windows)
# os.environ['HF_HOME'] = '/path/to/your/custom/hf_cache' 

print("--- 诊断脚本开始执行 ---")

try:
    print("步骤 1: 尝试加载 ProsusAI/finbert 模型...")
    # 这一行是下载模型的关键点，可能在这里卡住
    model = SentenceTransformer('ProsusAI/finbert')
    print("步骤 2: 模型加载成功！")

    # 进一步测试：尝试编码一些文本
    print("步骤 3: 尝试编码示例文本...")
    sentences = ["这是一个测试句子。", "另一个测试句子。"]
    embeddings = model.encode(sentences) # 这一行可能会卡住或抛出错误
    print(f"步骤 4: 编码成功，生成的嵌入形状: {embeddings.shape}")
    print("--- 脚本执行完毕 ---")

except Exception as e:
    print(f"--- 捕获到异常 ---")
    print(f"模型加载或编码失败，异常类型: {type(e).__name__}")
    print(f"错误信息: {e}")
    import traceback
    traceback.print_exc() # 打印完整的堆栈跟踪，帮助诊断
except KeyboardInterrupt:
    print("\n--- 脚本被用户中断 (Ctrl+C) ---")

print("--- 脚本退出 ---")