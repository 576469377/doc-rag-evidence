#!/usr/bin/env python3
"""
测试ColPali检索性能（优化前后对比）
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core.schemas import AppConfig, QueryInput
from impl.index_colpali import ColPaliRetriever

def test_colpali_speed():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    # Load ColPali index
    index_dir = Path("data/indices/colpali_default")
    if not index_dir.exists():
        print("❌ ColPali索引不存在")
        return
    
    print("=" * 70)
    print("ColPali检索性能测试")
    print("=" * 70)
    
    retriever = ColPaliRetriever.load(
        index_dir,
        model_name=config.colpali["model"],
        device="cuda:2"
    )
    
    print(f"\n✅ 已加载索引: {len(retriever.store.page_ids)} 页")
    
    # Test queries
    test_queries = [
        "磷酸氢钙的主要原料有哪些？",
        "食品添加剂的技术要求是什么？",
        "产品的感官要求有哪些规定？"
    ]
    
    print(f"\n测试查询数量: {len(test_queries)}")
    print(f"Top-K: {config.top_k_retrieve}")
    print(f"Coarse-K: {retriever.max_global_pool_pages}")
    
    # Warm up
    print("\n⏳ 预热中...")
    retriever.retrieve(test_queries[0], config=config)
    
    # Benchmark
    print("\n" + "─" * 70)
    print("性能测试开始")
    print("─" * 70)
    
    total_time = 0
    for i, query in enumerate(test_queries, 1):
        start = time.time()
        result = retriever.retrieve(
            QueryInput(query_id=f"test_{i}", question=query),
            config=config
        )
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"\n查询 {i}: {query}")
        print(f"  ⏱️  耗时: {elapsed:.3f}秒 ({result.elapsed_ms}ms)")
        print(f"  📄 结果数: {len(result.hits)}")
        if result.hits:
            print(f"  🏆 Top-1: {result.hits[0].doc_id} page {result.hits[0].page_id} (score: {result.hits[0].score:.4f})")
    
    avg_time = total_time / len(test_queries)
    
    print("\n" + "=" * 70)
    print("测试结果")
    print("=" * 70)
    print(f"总耗时: {total_time:.3f}秒")
    print(f"平均每次查询: {avg_time:.3f}秒")
    print(f"查询速率: {1/avg_time:.2f} queries/sec")
    
    # Performance breakdown
    print(f"\n性能分析:")
    print(f"  • 索引页面数: {len(retriever.store.page_ids)}")
    print(f"  • Coarse检索候选数: {retriever.max_global_pool_pages}")
    print(f"  • 并行worker数: 8 (ThreadPoolExecutor)")
    print(f"  • 加速效果: 约 {retriever.max_global_pool_pages / 8:.1f}x (理论上)")

if __name__ == "__main__":
    test_colpali_speed()
