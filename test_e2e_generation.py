#!/usr/bin/env python3
"""
End-to-end test: Retrieval + LLM Generation
Test the complete pipeline with real Qwen3-VL model.
"""
import sys
sys.path.insert(0, "/workspace/doc-rag-evidence")

from app.ui.main_v1 import DocRAGUIV1

print("=" * 70)
print("End-to-End Test: Complete Pipeline with Qwen3-VL Generation")
print("=" * 70)

# Initialize app (will load config, retrievers, generator, etc.)
app = DocRAGUIV1(config_path="configs/app.yaml")

# Override to use Qwen3-VL
app.config.generator["type"] = "qwen3_vl"
from impl.generator_qwen_llm import QwenLLMGenerator
app.generator = QwenLLMGenerator(app.config)
app.pipeline.generator = app.generator  # Update pipeline's generator
print("\n✅ Switched to QwenLLMGenerator")

# Test questions
test_cases = [
    {
        "mode": "bm25",
        "question": "磷酸氢钙的原料有哪些？",
        "doc_id": "gb_1886.3-2021"
    },
    {
        "mode": "dense",
        "question": "食品添加剂的质量要求是什么？",
        "doc_id": "gb_1886.3-2021"
    },
    {
        "mode": "colpali",
        "question": "产品标准中对包装有什么要求？",
        "doc_id": "gb_1886.3-2021"
    }
]

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"Test Case {i}: {case['mode'].upper()} Retrieval + LLM Generation")
    print(f"{'='*70}")
    print(f"📝 Question: {case['question']}")
    print(f"📄 Document: {case['doc_id']}")
    
    # Create request
    from core.schemas import QueryInput
    request = QueryInput(
        query_id=f"e2e_llm_{i:02d}",
        question=case['question'],
        doc_ids=[case['doc_id']],
        retrieval_mode=case['mode']
    )
    
    # Switch retriever if needed
    if case['mode'] != app.config.retrieval_mode:
        app.pipeline.retriever = app.retrievers[case['mode']]
    
    # Run pipeline
    print(f"\n⏳ Running pipeline...")
    try:
        result = app.pipeline.answer(request, app.config)
        
        print(f"\n✅ Success!")
        print(f"   Hits: {len(result.retrieval.hits)} → Evidence: {len(result.evidence.evidence)}")
        
        # Show top evidence
        print(f"\n📚 Top 3 Evidence:")
        for j, ev in enumerate(result.evidence.evidence[:3], 1):
            snippet = (ev.snippet[:60] + "...") if len(ev.snippet) > 60 else ev.snippet
            print(f"   [{j}] {snippet}")
        
        # Show generated answer
        answer = result.generation.output.answer
        answer_display = (answer[:250] + "...") if len(answer) > 250 else answer
        print(f"\n💬 Answer ({len(answer)} chars):")
        print(f"   {answer_display}")
        
        # Show citations
        cited = result.generation.output.cited_units
        print(f"\n🔗 Citations: {len(cited)} references")
        if cited:
            print(f"   ✅ LLM cited evidence in answer")
        else:
            print(f"   ⚠️  No citations detected")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*70}")
print("✅ End-to-End Test Completed!")
print(f"{'='*70}")
