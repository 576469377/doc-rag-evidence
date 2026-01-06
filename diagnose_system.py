#!/usr/bin/env python3
"""
诊断OCR和文档导入问题
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_ocr_service():
    """检查OCR服务状态"""
    import requests
    import yaml
    from core.schemas import AppConfig
    
    print("=" * 60)
    print("🔍 检查OCR服务")
    print("=" * 60)
    
    with open("configs/app.yaml") as f:
        config = AppConfig(**yaml.safe_load(f))
    
    endpoint = config.ocr.get('endpoint', 'http://localhost:8000')
    
    print(f"\n配置的OCR端点: {endpoint}")
    print(f"模型: {config.ocr.get('model', 'N/A')}")
    print(f"模型路径: {config.ocr.get('model_path', 'N/A')}")
    
    try:
        print(f"\n尝试连接 {endpoint}/health ...")
        response = requests.get(f"{endpoint}/health", timeout=5)
        
        if response.status_code == 200:
            print("✅ OCR服务正在运行")
            return True
        else:
            print(f"⚠️  OCR服务响应异常 (状态码: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到OCR服务")
        print("\n可能的原因:")
        print("  1. vLLM服务未启动")
        print("  2. 端口不正确")
        print("  3. 服务正在启动中（首次启动需要1-2分钟）")
        print("\n解决方案:")
        print("  # 启动OCR服务")
        print("  ./scripts/start_ocr_vllm.sh")
        print("\n  # 或查看日志")
        print("  tail -f logs/vllm_ocr.log")
        return False
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        return False

def check_documents():
    """检查已导入的文档"""
    import yaml
    from core.schemas import AppConfig
    from infra.store_local import DocumentStoreLocal
    
    print("\n" + "=" * 60)
    print("📚 检查已导入文档")
    print("=" * 60)
    
    with open("configs/app.yaml") as f:
        config = AppConfig(**yaml.safe_load(f))
    
    store = DocumentStoreLocal(config)
    docs = store.list_documents()
    
    if not docs:
        print("\n⚠️  没有找到任何文档")
        print("请先在UI中上传PDF文档")
        return
    
    print(f"\n找到 {len(docs)} 个文档:\n")
    
    for doc in docs:
        print(f"📄 {doc.doc_id}")
        print(f"   标题: {doc.title}")
        print(f"   页数: {doc.page_count}")
        print(f"   创建时间: {doc.created_at}")
        
        # Check pages
        pages = store.list_pages(doc.doc_id)
        text_pages = sum(1 for p in pages if p.has_text)
        image_pages = sum(1 for p in pages if p.has_image)
        
        print(f"   有文本页面: {text_pages}/{len(pages)}")
        print(f"   有图片页面: {image_pages}/{len(pages)}")
        
        # Check first page content
        if pages:
            page = pages[0]
            artifact = store.load_page_artifact(doc.doc_id, page.page_id)
            
            if artifact and artifact.text:
                text_len = len(artifact.text.text.strip())
                blocks_count = len(artifact.blocks) if artifact.blocks else 0
                
                if text_len == 0:
                    print(f"   ⚠️  第一页文本为空 (OCR可能失败)")
                else:
                    print(f"   ✅ 第一页有文本 ({text_len} 字符)")
                
                if blocks_count == 0:
                    print(f"   ⚠️  没有blocks (需要重新导入)")
                else:
                    print(f"   ✅ 有 {blocks_count} 个blocks")
            else:
                print(f"   ❌ 无法加载页面内容")
        
        print()

def check_index_status():
    """检查索引状态"""
    from pathlib import Path
    
    print("=" * 60)
    print("🔧 检查索引状态")
    print("=" * 60)
    
    indices_dir = Path("data/indices")
    
    if not indices_dir.exists():
        print("\n⚠️  索引目录不存在")
        return
    
    index_dirs = list(indices_dir.iterdir())
    
    if not index_dirs:
        print("\n⚠️  没有找到任何索引")
        print("请在UI中构建索引: Document Management → Build Indices")
        return
    
    print(f"\n找到 {len(index_dirs)} 个索引:\n")
    
    for index_dir in sorted(index_dirs):
        if index_dir.is_dir():
            files = list(index_dir.glob("*"))
            print(f"📁 {index_dir.name}")
            print(f"   文件数: {len(files)}")
            print(f"   文件: {[f.name for f in files[:5]]}")
            print()

def main():
    """运行所有诊断"""
    print("\n" + "=" * 60)
    print("🏥 Doc-RAG-Evidence 系统诊断")
    print("=" * 60)
    
    # 1. Check OCR service
    ocr_ok = check_ocr_service()
    
    # 2. Check documents
    check_documents()
    
    # 3. Check indices
    check_index_status()
    
    # Summary
    print("=" * 60)
    print("📊 诊断总结")
    print("=" * 60)
    
    if not ocr_ok:
        print("\n⚠️  问题: OCR服务未运行")
        print("\n建议:")
        print("1. 启动OCR服务:")
        print("   ./scripts/start_ocr_vllm.sh")
        print("\n2. 等待1-2分钟让服务完全启动")
        print("\n3. 重新运行此诊断脚本")
        print("   python diagnose_system.py")
        print("\n4. 如果需要，重新导入PDF（勾选Use OCR）")
    else:
        print("\n✅ OCR服务正常")
        print("\n如果文档文本为空，请:")
        print("1. 在UI中删除旧文档")
        print("2. 重新上传PDF并勾选'Use OCR'")
        print("3. 构建索引: Document Management → Build Indices")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
