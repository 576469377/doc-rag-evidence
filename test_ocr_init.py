#!/usr/bin/env python3
"""测试OCR客户端初始化"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from core.schemas import AppConfig
from infra.store_local import DocumentStoreLocal
from impl.ingest_pdf_v1 import PDFIngestorV1

print('🔍 测试OCR客户端初始化...\n')

with open('configs/app.yaml') as f:
    config = AppConfig(**yaml.safe_load(f))

store = DocumentStoreLocal(config)

print(f'配置信息:')
print(f'  OCR provider: {config.ocr.get("provider")}')
print(f'  OCR endpoint: {config.ocr.get("endpoint")}')
print(f'  OCR model: {config.ocr.get("model")}\n')

# Test with use_ocr=True
print('初始化 PDFIngestorV1 (use_ocr=True)...')
ingestor = PDFIngestorV1(config=config, store=store, use_ocr=True)

print(f'\nOCR客户端信息:')
print(f'  类型: {type(ingestor.ocr_client).__name__}')
print(f'  use_ocr: {ingestor.use_ocr}')

if hasattr(ingestor.ocr_client, 'endpoint'):
    print(f'  端点: {ingestor.ocr_client.endpoint}')
    print(f'  模型: {ingestor.ocr_client.model}')
    print(f'  超时: {ingestor.ocr_client.timeout}秒')
    print(f'\n✅ OCR客户端已正确初始化 (SGLangOcrClient)')
else:
    print(f'\n❌ OCR客户端未正确初始化 (MockOcrClient - 会返回空文本!)')
