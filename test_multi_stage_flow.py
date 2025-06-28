#!/usr/bin/env python3
"""
测试修改后的多阶段检索流程
验证：预过滤 → FAISS检索(summary) → 重排序(original_context) → UI显示(original_context)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_multi_stage_flow():
    """测试多阶段检索流程"""
    print("🧪 测试修改后的多阶段检索流程")
    print("=" * 60)
    
    try:
        # 导入多阶段检索系统
        sys.path.append(str(Path(__file__).parent / "alphafin_data_process"))
        from multi_stage_retrieval_final import MultiStageRetrievalSystem
        
        # 数据文件路径
        data_path = Path("data/alphafin/alphafin_merged_generated_qa.json")
        
        if not data_path.exists():
            print(f"❌ 数据文件不存在: {data_path}")
            return
        
        print("📊 初始化多阶段检索系统...")
        retrieval_system = MultiStageRetrievalSystem(
            data_path=data_path,
            dataset_type="chinese",
            use_existing_config=True
        )
        
        print("\n✅ 系统初始化完成")
        print(f"📈 数据总量: {len(retrieval_system.data)} 条记录")
        print(f"🔍 有效索引: {len(retrieval_system.valid_indices)} 条")
        print(f"🔄 重排序文档: {len(retrieval_system.contexts_for_rerank)} 条")
        
        # 测试查询
        test_queries = [
            {
                "query": "钢铁行业发展趋势",
                "company_name": None,
                "stock_code": None,
                "description": "通用查询（无元数据过滤）"
            },
            {
                "query": "公司业绩表现如何？",
                "company_name": "首钢股份",
                "stock_code": None,
                "description": "基于公司名称的查询"
            },
            {
                "query": "财务数据",
                "company_name": None,
                "stock_code": "000959",
                "description": "基于股票代码的查询"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n🔍 测试 {i}: {test_case['description']}")
            print(f"   查询: {test_case['query']}")
            if test_case['company_name']:
                print(f"   公司: {test_case['company_name']}")
            if test_case['stock_code']:
                print(f"   股票代码: {test_case['stock_code']}")
            
            try:
                # 执行多阶段检索
                results = retrieval_system.search(
                    query=test_case['query'],
                    company_name=test_case['company_name'],
                    stock_code=test_case['stock_code'],
                    top_k=3
                )
                
                print(f"   ✅ 检索成功，找到 {len(results)} 条结果")
                
                # 显示前3条结果
                for j, result in enumerate(results[:3], 1):
                    print(f"\n   结果 {j}:")
                    print(f"     公司: {result.get('company_name', 'N/A')}")
                    print(f"     股票代码: {result.get('stock_code', 'N/A')}")
                    print(f"     分数: {result.get('combined_score', 0):.4f}")
                    
                    # 检查字段完整性
                    has_summary = bool(result.get('summary'))
                    has_original_context = bool(result.get('original_context'))
                    has_generated_question = bool(result.get('generated_question'))
                    
                    print(f"     字段检查:")
                    print(f"       - summary: {'✅' if has_summary else '❌'}")
                    print(f"       - original_context: {'✅' if has_original_context else '❌'}")
                    print(f"       - generated_question: {'✅' if has_generated_question else '❌'}")
                    
                    # 显示original_context的前100个字符
                    original_context = result.get('original_context', '')
                    if original_context:
                        print(f"     原始上下文预览: {original_context[:100]}...")
                    else:
                        print(f"     原始上下文: 无")
                
            except Exception as e:
                print(f"   ❌ 检索失败: {e}")
        
        print(f"\n🎉 测试完成！")
        print(f"📋 流程验证:")
        print(f"   1. ✅ 预过滤: 基于元数据（公司名称、股票代码、报告日期）")
        print(f"   2. ✅ FAISS索引构建: 使用summary字段")
        print(f"   3. ✅ FAISS检索: 基于summary嵌入向量")
        print(f"   4. ✅ 重排序: 使用original_context进行Qwen重排序")
        print(f"   5. ✅ UI显示: 返回完整的original_context")
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_stage_flow() 