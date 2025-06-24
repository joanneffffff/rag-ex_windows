#!/usr/bin/env python3
"""
Enhanced RAG UI for Linux - Console version with dual space dual index
支持双空间双索引和Qwen重排序器 - Linux GPU环境测试版本
"""

import os
import sys
import traceback
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.parameters import Config
from xlm.registry.retriever import load_enhanced_retriever
from xlm.registry.generator import load_generator

class EnhancedRagConsole:
    def __init__(
        self,
        chinese_data_path: str = "",
        english_data_path: str = "",
        cache_dir: str = "",
        use_faiss: bool = True,
        enable_reranker: bool = True
    ):
        """
        初始化增强RAG控制台系统
        
        Args:
            chinese_data_path: 中文数据路径
            english_data_path: 英文数据路径
            cache_dir: 缓存目录
            use_faiss: 是否使用FAISS
            enable_reranker: 是否启用重排序器
        """
        self.chinese_data_path = chinese_data_path
        self.english_data_path = english_data_path
        self.cache_dir = cache_dir
        self.use_faiss = use_faiss
        self.enable_reranker = enable_reranker
        
        # 设置环境变量
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
            os.environ['HF_HOME'] = cache_dir
            os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
        
        # 初始化系统组件
        self._init_components()
    
    def _init_components(self):
        """初始化系统组件"""
        print("\n=== 初始化增强RAG系统 ===")
        
        # 创建配置
        self.config = Config()
        self.config.retriever.use_faiss = self.use_faiss
        self.config.reranker.enabled = self.enable_reranker
        
        if self.cache_dir:
            self.config.cache_dir = self.cache_dir
        
        print(f"配置信息:")
        print(f"- FAISS: {self.config.retriever.use_faiss}")
        print(f"- 重排序器: {self.config.reranker.enabled}")
        print(f"- 中文编码器: {self.config.encoder.chinese_model_path}")
        print(f"- 英文编码器: {self.config.encoder.english_model_path}")
        print(f"- 重排序器: {self.config.reranker.model_name}")
        
        # 加载增强检索器
        print("\n1. 加载增强检索器...")
        try:
            self.retriever = load_enhanced_retriever(
                config=self.config,
                chinese_data_path=self.chinese_data_path if self.chinese_data_path else None,
                english_data_path=self.english_data_path if self.english_data_path else None
            )
            print("✅ 增强检索器加载成功")
        except Exception as e:
            print(f"❌ 增强检索器加载失败: {e}")
            print("尝试使用备用检索器...")
            self._init_fallback_retriever()
        
        # 加载生成器
        print("\n2. 加载生成器...")
        try:
            self.generator = load_generator(
                generator_model_name=self.config.generator.model_name,
                use_local_llm=True
            )
            print("✅ 生成器加载成功")
        except Exception as e:
            print(f"❌ 生成器加载失败: {e}")
            self.generator = None
        
        print("=== 系统初始化完成 ===")
    
    def _init_fallback_retriever(self):
        """初始化备用检索器"""
        try:
            from xlm.components.retriever.sbert_retriever import SBERTRetriever
            from xlm.components.encoder.encoder import Encoder
            from xlm.dto.dto import DocumentWithMetadata, DocumentMetadata
            
            print("使用备用SBERT检索器...")
            
            # 创建简单的测试文档
            test_docs = [
                DocumentWithMetadata(
                    content="这是一个测试文档，用于验证系统功能。",
                    metadata=DocumentMetadata(
                        doc_id="test_1",
                        source="test",
                        language="chinese"
                    )
                )
            ]
            
            encoder = Encoder(
                model_name=self.config.encoder.model_name,
                device=self.config.encoder.device,
                cache_dir=self.config.encoder.cache_dir
            )
            
            self.retriever = SBERTRetriever(
                encoder=encoder,
                corpus_documents=test_docs,
                use_faiss=self.use_faiss
            )
            print("✅ 备用检索器加载成功")
            
        except Exception as e:
            print(f"❌ 备用检索器也加载失败: {e}")
            self.retriever = None
    
    def process_query(self, query: str, top_k: int = 5):
        """
        处理查询
        
        Args:
            query: 查询文本
            top_k: 检索文档数量
            
        Returns:
            (答案, 检索结果, 错误信息)
        """
        if not query.strip():
            return "", "", "查询不能为空"
        
        try:
            print(f"\n🔍 处理查询: {query}")
            print(f"📊 参数: top_k={top_k}, reranker={self.enable_reranker}")
            
            if self.retriever:
                # 使用检索器
                retrieved_documents, retriever_scores = self.retriever.retrieve(
                    text=query,
                    top_k=top_k,
                    return_scores=True
                )
                
                if not retrieved_documents:
                    return "未找到相关文档", "", ""
                
                # 格式化检索结果
                docs_text = ""
                for i, (doc, score) in enumerate(zip(retrieved_documents, retriever_scores)):
                    docs_text += f"文档 {i+1} (分数: {score:.4f}):\n"
                    docs_text += f"{doc.content}\n"
                    if hasattr(doc, 'metadata') and doc.metadata:
                        docs_text += f"元数据: {doc.metadata.doc_id}, {getattr(doc.metadata, 'language', 'unknown')}\n"
                    docs_text += "-" * 50 + "\n"
                
                # 如果有生成器，尝试生成答案
                if self.generator:
                    try:
                        context = "\n".join([doc.content for doc in retrieved_documents[:3]])
                        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
                        answer = self.generator.generate(prompt)
                    except Exception as e:
                        answer = f"生成答案时出错: {e}"
                else:
                    answer = "仅检索模式，未生成答案"
                
                return answer, docs_text, ""
                
            else:
                return "", "", "检索器未初始化"
                
        except Exception as e:
            error_msg = f"处理查询时发生错误: {str(e)}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            return "", "", error_msg
    
    def get_system_info(self):
        """获取系统信息"""
        info = {
            "FAISS": "启用" if self.use_faiss else "禁用",
            "重排序器": "启用" if self.enable_reranker else "禁用",
            "检索器": "正常" if self.retriever else "异常",
            "生成器": "正常" if self.generator else "异常"
        }
        
        # 尝试获取文档数量信息
        if self.retriever:
            try:
                if hasattr(self.retriever, 'get_corpus_size'):
                    corpus_sizes = self.retriever.get_corpus_size()
                    info["中文文档"] = str(corpus_sizes.get('chinese', 0))
                    info["英文文档"] = str(corpus_sizes.get('english', 0))
                elif hasattr(self.retriever, 'corpus_documents'):
                    info["文档数量"] = str(len(self.retriever.corpus_documents))
            except:
                info["文档数量"] = "未知"
        
        return info
    
    def run_interactive(self):
        """运行交互式控制台"""
        print("\n🚀 增强RAG系统启动成功！")
        print("=" * 60)
        
        # 显示系统信息
        info = self.get_system_info()
        print("📊 系统状态:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\n💡 使用说明:")
        print("  - 输入问题（支持中英文）")
        print("  - 输入 'info' 查看系统信息")
        print("  - 输入 'quit' 或 'exit' 退出")
        print("  - 输入 'help' 查看帮助")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n❓ 请输入您的问题: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'info':
                    info = self.get_system_info()
                    print("\n📊 系统信息:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.lower() == 'help':
                    print("\n💡 帮助信息:")
                    print("  - 支持中英文查询，系统会自动检测语言")
                    print("  - 中文查询使用中文编码器和索引")
                    print("  - 英文查询使用英文编码器和索引")
                    print("  - 所有查询都使用统一的Qwen重排序器")
                    print("  - 命令: info, help, quit/exit/q")
                    continue
                elif not user_input:
                    continue
                
                # 处理查询
                answer, context, error = self.process_query(user_input)
                
                if error:
                    print(f"\n❌ 错误: {error}")
                else:
                    print(f"\n💡 答案:")
                    print(answer)
                    
                    if context:
                        print(f"\n📄 检索到的文档:")
                        print(context)
                
            except KeyboardInterrupt:
                print("\n\n👋 用户中断，再见！")
                break
            except Exception as e:
                print(f"\n❌ 系统错误: {e}")
                traceback.print_exc()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="增强RAG系统 - Linux控制台版本")
    parser.add_argument("--chinese_data", type=str, 
                       default="data/alphafin/alphafin_rag_ready_generated_cleaned.json",
                       help="中文数据路径")
    parser.add_argument("--english_data", type=str,
                       default="data/tatqa_dataset_raw/tatqa_dataset_train.json", 
                       help="英文数据路径")
    parser.add_argument("--cache_dir", type=str, default="/tmp/huggingface",
                       help="模型缓存目录")
    parser.add_argument("--no_faiss", action="store_true",
                       help="禁用FAISS")
    parser.add_argument("--no_reranker", action="store_true",
                       help="禁用重排序器")
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if args.chinese_data and not os.path.exists(args.chinese_data):
        print(f"⚠️  警告: 中文数据文件不存在: {args.chinese_data}")
        args.chinese_data = ""
    
    if args.english_data and not os.path.exists(args.english_data):
        print(f"⚠️  警告: 英文数据文件不存在: {args.english_data}")
        args.english_data = ""
    
    # 创建UI实例
    ui = EnhancedRagConsole(
        chinese_data_path=args.chinese_data,
        english_data_path=args.english_data,
        cache_dir=args.cache_dir,
        use_faiss=not args.no_faiss,
        enable_reranker=not args.no_reranker
    )
    
    # 运行交互式控制台
    ui.run_interactive()

if __name__ == "__main__":
    main() 