#!/usr/bin/env python3
"""
Simple test: verify generator can be initialized and fallback works.
"""
import sys
import yaml

from core.schemas import AppConfig, GenerationRequest, EvidenceItem

# Test configuration
config_dict = {
    "data_root": "data",
    "docs_dir": "data/docs",
    "indices_dir": "data/indices",
    "runs_dir": "data/runs",
    "reports_dir": "data/reports",
    "chunk_level": "block",
    "top_k_retrieve": 20,
    "top_k_rerank": 10,
    "top_k_evidence": 5,
    "citation_level": "block",
    "bbox_mode": "none",
    "embedder_name": "vllm-embedder",
    "reranker_name": "none",
    "llm_name": "qwen-llm",
    "max_context_chars": 12000,
    "require_citations": True,
    "generator": {
        "type": "qwen3_vl"
    },
    "llm": {
        "backend": "vllm",
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "endpoint": "http://localhost:8002",
        "max_new_tokens": 1024,
        "temperature": 0.1,
        "top_p": 0.9,
        "citation_policy": "strict"
    },
    "ocr": {},
    "dense": {},
    "colpali": {}
}

config = AppConfig(**config_dict)

print("=" * 60)
print("Testing Generator Initialization")
print("=" * 60)

# Try to create generator
try:
    from impl.generator_qwen_llm import QwenLLMGenerator
    generator = QwenLLMGenerator(config)
    print(f"✅ Generator initialized (backend={generator.backend})")
except Exception as e:
    print(f"❌ Failed to initialize: {e}")
    sys.exit(1)

# Create test evidence
evidence = [
    EvidenceItem(
        rank=1,
        unit_id="test_001",
        doc_id="test_doc",
        page_id=0,
        block_id="block_1",
        snippet="磷酸氢钙的主要原料包括氢氧化钙、碳酸钙和氧化钙。",
        score=0.95
    ),
    EvidenceItem(
        rank=2,
        unit_id="test_002",
        doc_id="test_doc",
        page_id=1,
        block_id="block_2",
        snippet="生产过程中还需要使用食品添加剂磷酸。",
        score=0.88
    )
]

# Create request
request = GenerationRequest(
    query_id="simple_test_001",
    question="磷酸氢钙的原料有哪些？",
    evidence=evidence,
    config=config
)

print(f"\n📝 Query: {request.question}")
print(f"   Evidence: {len(evidence)} items")

# Test generation
print(f"\n⏳ Generating...")
try:
    result = generator.generate(request)
    
    print(f"\n✅ Success!")
    print(f"   Time: {result.elapsed_ms}ms")
    print(f"\n📄 Answer ({len(result.output.answer)} chars):")
    print(result.output.answer)
    print(f"\n🔗 Citations: {result.output.cited_units}")
    
except Exception as e:
    print(f"\n❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ Test passed!")
