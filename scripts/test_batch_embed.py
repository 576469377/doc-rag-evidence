#!/usr/bin/env python3
"""
测试批量图像embedding性能
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from core.schemas import AppConfig
from impl.index_colpali import ColPaliRetriever

def test_batch_embedding():
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    # Get all page images
    docs_dir = Path("data/docs")
    page_images = []
    
    for doc_dir in docs_dir.iterdir():
        if not doc_dir.is_dir():
            continue
        pages_dir = doc_dir / "pages"
        if not pages_dir.exists():
            continue
        
        for page_dir in sorted(pages_dir.iterdir()):
            if not page_dir.is_dir():
                continue
            image_path = page_dir / "page.png"
            if image_path.exists():
                page_images.append(str(image_path))
    
    if not page_images:
        print("❌ No page images found")
        return
    
    print("=" * 70)
    print(f"批量Embedding性能测试")
    print("=" * 70)
    print(f"图像数量: {len(page_images)}")
    
    # Load model
    print(f"\n⏳ 加载ColPali模型...")
    retriever = ColPaliRetriever(
        model_name=config.colpali["model"],
        device="cuda:2",
        cache_dir=Path("data/cache/colpali_embeddings")
    )
    print(f"✅ 模型加载完成")
    
    # Test batch sizes
    test_images = page_images[:8]  # Test with 8 images
    
    for batch_size in [4, 8]:
        print(f"\n{'─' * 70}")
        print(f"Batch Size = {batch_size}")
        print(f"{'─' * 70}")
        
        # Clear cache to ensure fresh run
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start = time.time()
        results = retriever._embed_images_batch(test_images, batch_size=batch_size)
        elapsed = time.time() - start
        
        print(f"  总耗时: {elapsed:.2f}秒")
        print(f"  平均每张: {elapsed/len(test_images)*1000:.1f}ms")
        print(f"  吞吐量: {len(test_images)/elapsed:.2f} images/sec")
        print(f"  结果数: {len(results)}")
        
        if batch_size == 1:
            baseline = elapsed
        else:
            speedup = baseline / elapsed
            print(f"  加速比: {speedup:.2f}x vs batch_size=1")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_batch_embedding()
