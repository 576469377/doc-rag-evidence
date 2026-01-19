#!/usr/bin/env python3
"""
检查索引状态的辅助脚本
"""

from pathlib import Path
from infra.store_local import DocumentStoreLocal
from core.schemas import AppConfig
from impl.index_tracker import IndexTracker
import yaml
import sys

def check_index_status(index_name: str = "dense_vl_default"):
    """检查指定索引的完成状态"""
    
    # Load config
    with open('configs/app.yaml') as f:
        config = AppConfig(**yaml.safe_load(f))
    
    store = DocumentStoreLocal(config)
    index_dir = Path(config.indices_dir) / index_name
    
    if not index_dir.exists():
        print(f"❌ 索引目录不存在: {index_dir}")
        return
    
    tracker = IndexTracker(index_dir)
    
    # Get all documents
    all_docs = {doc.doc_id: doc for doc in store.list_documents()}
    
    # Statistics
    total_docs = len(all_docs)
    indexed_docs = len(tracker.indexed_docs)
    missing_docs = total_docs - indexed_docs
    
    print(f"📊 索引状态统计 - {index_name}")
    print(f"{'='*60}")
    print(f"总文档数:     {total_docs}")
    print(f"已索引文档数: {indexed_docs} (✓)")
    print(f"未索引文档数: {missing_docs} (✗)")
    print(f"完成比例:     {indexed_docs/total_docs*100:.1f}%")
    print()
    
    # Count pages
    total_pages = sum(doc.page_count for doc in all_docs.values())
    indexed_pages = sum(info.get('page_count', 0) for info in tracker.indexed_docs.values())
    
    print(f"总页数:       {total_pages}")
    print(f"已索引页数:   {indexed_pages}")
    print(f"页面完成比例: {indexed_pages/total_pages*100:.1f}%")
    print()
    
    # Show some completed docs
    print(f"✅ 已完成文档示例（前10个）:")
    completed = []
    for doc_id, doc in all_docs.items():
        if doc_id in tracker.indexed_docs:
            indexed_info = tracker.indexed_docs[doc_id]
            if indexed_info.get('page_count', 0) == doc.page_count:
                completed.append(doc_id)
    
    for i, doc_id in enumerate(completed[:10], 1):
        doc = all_docs[doc_id]
        print(f"  {i:2d}. {doc_id} ({doc.page_count} pages)")
    
    if len(completed) > 10:
        print(f"  ... 还有 {len(completed) - 10} 个文档")
    
    print()
    
    # Show some missing docs
    missing = [doc_id for doc_id in all_docs if doc_id not in tracker.indexed_docs]
    if missing:
        print(f"❌ 未索引文档示例（前10个）:")
        for i, doc_id in enumerate(missing[:10], 1):
            doc = all_docs[doc_id]
            print(f"  {i:2d}. {doc_id} ({doc.page_count} pages)")
        
        if len(missing) > 10:
            print(f"  ... 还有 {len(missing) - 10} 个文档")
    
    print()
    print(f"{'='*60}")
    
    # Check for incomplete (pages mismatch)
    incomplete = []
    for doc_id, doc in all_docs.items():
        if doc_id in tracker.indexed_docs:
            indexed_info = tracker.indexed_docs[doc_id]
            indexed_pages = indexed_info.get('page_count', 0)
            if indexed_pages != doc.page_count:
                incomplete.append((doc_id, indexed_pages, doc.page_count))
    
    if incomplete:
        print(f"⚠️  页数不匹配的文档:")
        for doc_id, indexed, total in incomplete[:10]:
            print(f"  {doc_id}: {indexed}/{total} pages")
        print()

if __name__ == "__main__":
    index_name = sys.argv[1] if len(sys.argv) > 1 else "dense_vl_default"
    check_index_status(index_name)
