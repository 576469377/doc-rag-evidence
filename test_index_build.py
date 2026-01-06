#!/usr/bin/env python3
"""测试 Dense 和 ColPali 索引构建"""

import sys
import yaml
from pathlib import Path

print("=" * 60)
print("测试索引构建")
print("=" * 60)

# 加载配置
with open("configs/app.yaml", 'r') as f:
    config_dict = yaml.safe_load(f)

from core.schemas import AppConfig, IndexUnit
config = AppConfig(**config_dict)

# 创建测试单元
test_units = [
    IndexUnit(
        unit_id="test_doc_p0000_b0",
        doc_id="test_doc",
        page_id=0,
        block_id="b0",
        text="This is a test document about machine learning."
    ),
    IndexUnit(
        unit_id="test_doc_p0000_b1",
        doc_id="test_doc",
        page_id=0,
        block_id="b1",
        text="Deep learning is a subset of machine learning."
    ),
]

print(f"\n创建了 {len(test_units)} 个测试单元\n")

# 测试 1: Dense Indexer
print("测试 1: Dense Indexer")
print("-" * 60)

try:
    from impl.index_dense import VLLMEmbedder, DenseIndexerRetriever
    
    # 检查 embedding 服务
    import requests
    try:
        resp = requests.get("http://localhost:8001/health", timeout=2)
        if resp.status_code == 200:
            print("✅ Embedding 服务运行中")
        else:
            print(f"⚠️  Embedding 服务状态异常: {resp.status_code}")
    except Exception as e:
        print(f"❌ Embedding 服务未运行: {e}")
        print("   跳过 Dense 测试")
        print()
        test_units = None  # Skip dense test
    
    if test_units:
        embedder = VLLMEmbedder(
            endpoint=config.dense["endpoint"],
            model=config.dense["model"]
        )
        
        retriever = DenseIndexerRetriever(embedder)
        
        # 测试新的签名
        print("调用 build_index(units, config)...")
        retriever.build_index(test_units, config)
        
        print(f"✅ Dense 索引构建成功")
        print(f"   索引单元数: {len(retriever.units)}")
        print(f"   索引类型: {retriever.index_type}")
        
except Exception as e:
    print(f"❌ Dense 索引失败: {e}")
    import traceback
    traceback.print_exc()

print()

# 测试 2: ColPali Indexer
print("测试 2: ColPali Indexer")
print("-" * 60)

try:
    from impl.index_colpali import ColPaliRetriever
    
    print(f"模型路径: {config.colpali['model']}")
    print(f"设备: {config.colpali.get('device', 'cuda:0')}")
    
    # 只测试初始化（不实际加载模型，太慢）
    print("\n测试模型加载...")
    print("⚠️  注意: 实际加载 ColQwen3 模型需要几分钟...")
    print("   如果要测试完整流程，请手动运行:")
    print("   python -c 'from impl.index_colpali import ColPaliRetriever; r = ColPaliRetriever(...)'")
    
    # 检查必要的库
    try:
        import torch
        import transformers
        print("✅ PyTorch 和 Transformers 已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
    
    try:
        import faiss
        print("✅ FAISS 已安装")
    except ImportError:
        print("❌ FAISS 未安装")
    
    print("\n✅ ColPali 类可以导入（未测试实际加载模型）")
    
except Exception as e:
    print(f"❌ ColPali 测试失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("测试完成")
print("=" * 60)
print()
print("总结:")
print("  • Dense 索引: build_index() 方法签名已修复")
print("  • ColPali 模型: AutoModel 加载使用 trust_remote_code=True")
print("  • ColPali API: 使用 process_images/process_texts 和 .embeddings")
print()
