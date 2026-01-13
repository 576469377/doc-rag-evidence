#!/usr/bin/env python3
"""
实时监控ColPali索引构建的GPU利用率
"""
import sys
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def monitor_gpu():
    """获取GPU 2的使用情况"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", 
             "--format=csv,noheader,nounits", "-i", "2"],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            mem_used = int(parts[0].strip())
            mem_total = int(parts[1].strip())
            gpu_util = int(parts[2].strip())
            return mem_used, mem_total, gpu_util
    except:
        pass
    return None, None, None

def test_build_with_monitor():
    """测试索引构建并监控GPU"""
    import yaml
    from core.schemas import AppConfig
    from infra.store_local import DocumentStoreLocal
    from impl.index_colpali import ColPaliRetriever
    
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = AppConfig(**config_dict)
    
    store = DocumentStoreLocal(config)
    
    # Get test pages from docs directory
    docs_dir = Path(config.docs_dir)
    all_pages = []
    
    for doc_dir in sorted(docs_dir.iterdir()):
        if not doc_dir.is_dir():
            continue
        doc_id = doc_dir.name
        pages_dir = doc_dir / "pages"
        if not pages_dir.exists():
            continue
        
        for page_dir in sorted(pages_dir.iterdir()):
            if not page_dir.is_dir():
                continue
            try:
                page_id = int(page_dir.name)
                all_pages.append((doc_id, page_id))
            except ValueError:
                continue
    
    if not all_pages:
        print("❌ No pages found")
        return
    
    test_pages = all_pages[:16]  # Test with 16 pages to see batch effect
    
    print("=" * 70)
    print(f"ColPali批量处理性能测试")
    print("=" * 70)
    print(f"测试页面: {len(test_pages)}")
    print(f"Batch size: 16")
    print(f"GPU: cuda:2")
    print()
    
    # Load model
    print("⏳ 加载ColPali模型...")
    retriever = ColPaliRetriever(
        model_name=config.colpali["model"],
        device="cuda:2",
        cache_dir=Path("data/cache/colpali_embeddings")
    )
    
    # Monitor baseline
    mem_used, mem_total, gpu_util = monitor_gpu()
    if mem_used:
        print(f"📊 模型加载后: GPU显存 {mem_used}/{mem_total} MB ({mem_used/mem_total*100:.1f}%), 利用率 {gpu_util}%")
    
    print()
    print("⏳ 开始构建索引...")
    print("   (观察GPU利用率应该升高到 60-90%)")
    print()
    
    # Start building
    import threading
    building = [True]
    max_util = [0]
    max_mem = [0]
    
    def monitor_thread():
        while building[0]:
            mem_used, mem_total, gpu_util = monitor_gpu()
            if mem_used:
                max_util[0] = max(max_util[0], gpu_util)
                max_mem[0] = max(max_mem[0], mem_used)
                print(f"\r📊 实时: GPU显存 {mem_used}/{mem_total} MB ({mem_used/mem_total*100:.1f}%), 利用率 {gpu_util}%", end='', flush=True)
            time.sleep(0.5)
    
    monitor = threading.Thread(target=monitor_thread, daemon=True)
    monitor.start()
    
    start = time.time()
    retriever.build_index(test_pages, config=config)
    elapsed = time.time() - start
    
    building[0] = False
    time.sleep(0.6)
    print()
    
    print()
    print("=" * 70)
    print("结果")
    print("=" * 70)
    print(f"✅ 构建完成")
    print(f"   总耗时: {elapsed:.2f}秒")
    print(f"   平均/页: {elapsed/len(test_pages):.2f}秒")
    print(f"   吞吐量: {len(test_pages)/elapsed:.2f} pages/sec")
    print()
    print(f"📊 GPU峰值:")
    print(f"   最大显存: {max_mem[0]} MB ({max_mem[0]/mem_total*100:.1f}%)")
    print(f"   最大利用率: {max_util[0]}%")
    print()
    
    if max_util[0] < 50:
        print("⚠️  GPU利用率偏低，可能原因：")
        print("   1. 图像已缓存，跳过了实际计算")
        print("   2. batch_size太小，可以增大到24或32")
        print("   3. CPU预处理成为瓶颈（图像解码）")
    elif max_util[0] < 80:
        print("✅ GPU利用率正常（50-80%是合理范围）")
    else:
        print("🚀 GPU利用率很高（>80%），批量处理效果显著！")

if __name__ == "__main__":
    test_build_with_monitor()
