#!/usr/bin/env python3
"""
Test Qwen LLM generator with evidence-based generation.
Tests both API (vLLM) and transformers backends.
"""
import sys
import yaml
from pathlib import Path

from core.schemas import AppConfig, GenerationRequest, EvidenceItem


def test_generator(backend="vllm"):
    """Test generator with specified backend."""
    # Load config
    with open("configs/app.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Override for testing
    config_dict["generator"]["type"] = "qwen3_vl"
    config_dict["llm"]["backend"] = backend
    
    config = AppConfig(**config_dict)
    
    print("=" * 60)
    print(f"Testing QwenLLMGenerator (backend={backend})")
    print("=" * 60)
    
    # Create generator
    from impl.generator_qwen_llm import QwenLLMGenerator
    
    try:
        generator = QwenLLMGenerator(config)
    except Exception as e:
        print(f"\n❌ Failed to initialize generator: {e}")
        return False
    
    # Create test evidence
    evidence = [
        EvidenceItem(
            rank=1,
            unit_id="gb_1886.3-2021_p0002_b000",
            doc_id="gb_1886.3-2021",
            page_id=2,
            block_id="gb_1886.3-2021__page0002__blk000",
            snippet="本标准适用于以氢氧化钙(或碳酸钙、氧化钙)和食品添加剂磷酸(含湿法磷酸)为原料生产的食品添加剂磷酸氢钙。",
            score=0.95
        ),
        EvidenceItem(
            rank=2,
            unit_id="gb_1886.3-2021_p0001_b002",
            doc_id="gb_1886.3-2021",
            page_id=1,
            block_id="gb_1886.3-2021__page0001__blk002",
            snippet="本标准规定了食品添加剂磷酸氢钙的要求、检验方法、检验规则及标签、包装、运输和贮存。",
            score=0.88
        ),
        EvidenceItem(
            rank=3,
            unit_id="gb_1886.3-2021_p0000_b001",
            doc_id="gb_1886.3-2021",
            page_id=0,
            block_id="gb_1886.3-2021__page0000__blk001",
            snippet="GB 1886.3—2021 食品安全国家标准 食品添加剂 磷酸氢钙",
            score=0.75
        )
    ]
    
    # Create request
    request = GenerationRequest(
        query_id="test_gen_001",
        question="磷酸氢钙的原料是什么？",
        evidence=evidence,
        config=config
    )
    
    print(f"\n📝 Test Query: {request.question}")
    print(f"   Evidence items: {len(evidence)}")
    
    # Generate
    print(f"\n⏳ Generating answer...")
    try:
        result = generator.generate(request)
        
        print(f"\n✅ Generation succeeded!")
        print(f"   Elapsed: {result.elapsed_ms}ms")
        print(f"\n📄 Answer:")
        print(f"   {result.output.answer}")
        print(f"\n🔗 Citations:")
        print(f"   Cited units: {len(result.output.cited_units)}")
        for unit_id in result.output.cited_units:
            print(f"   - {unit_id}")
        
        # Validate
        has_citations = len(result.output.cited_units) > 0
        answer_not_empty = len(result.output.answer) > 10
        
        print(f"\n📋 Validation:")
        print(f"   {'✅' if answer_not_empty else '❌'} Answer not empty ({len(result.output.answer)} chars)")
        print(f"   {'✅' if has_citations else '❌'} Has citations ({len(result.output.cited_units)} units)")
        
        return answer_not_empty and has_citations
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    # Test different backends
    results = {}
    
    # Test vLLM (if available)
    print("\n\n" + "=" * 60)
    print("Test 1: vLLM Backend")
    print("=" * 60)
    try:
        results["vllm"] = test_generator("vllm")
    except Exception as e:
        print(f"vLLM test skipped: {e}")
        results["vllm"] = None
    
    # Test transformers (always available)
    print("\n\n" + "=" * 60)
    print("Test 2: Transformers Backend")
    print("=" * 60)
    try:
        results["transformers"] = test_generator("transformers")
    except Exception as e:
        print(f"Transformers test failed: {e}")
        results["transformers"] = False
    
    # Summary
    print("\n\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for backend, result in results.items():
        if result is None:
            status = "⏭️  SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {backend.upper():15} {status}")
    
    # Exit code
    any_passed = any(r for r in results.values() if r is True)
    sys.exit(0 if any_passed else 1)


if __name__ == "__main__":
    main()
