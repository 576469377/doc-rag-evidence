#!/usr/bin/env python3
"""
测试增量索引功能
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core.schemas import AppConfig
from infra.store_local import DocumentStoreLocal
from impl.index_incremental import IncrementalIndexManager

def main():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    # Initialize store
    store = DocumentStoreLocal(config)
    
    # Initialize incremental index manager
    manager = IncrementalIndexManager(config, store)
    
    print("=" * 70)
    print("增量索引功能测试")
    print("=" * 70)
    
    # Check for new documents
    new_docs_bm25 = manager.get_new_documents("bm25_default")
    new_docs_dense = manager.get_new_documents("dense_default")
    new_docs_colpali = manager.get_new_documents("colpali_default")
    
    print(f"\n未索引的文档:")
    print(f"  BM25:    {len(new_docs_bm25)} 个文档")
    print(f"  Dense:   {len(new_docs_dense)} 个文档")
    print(f"  ColPali: {len(new_docs_colpali)} 个文档")
    
    if new_docs_bm25:
        print(f"\n  需要索引的文档: {new_docs_bm25}")
    else:
        print(f"\n  ✅ 所有文档已索引")
    
    # Test BM25 incremental update
    if new_docs_bm25:
        print("\n" + "─" * 70)
        print("测试 BM25 增量更新")
        print("─" * 70)
        
        result = manager.update_bm25_index(index_name="bm25_default")
        print(f"\n状态: {result['status']}")
        print(f"消息: {result['message']}")
        if result['status'] == 'success':
            print(f"新增文档: {result['new_docs']}")
            print(f"新增单元: {result['new_units']}")
            print(f"总文档数: {result['total_docs']}")
            print(f"总单元数: {result['total_units']}")
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
