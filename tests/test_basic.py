#!/usr/bin/env python3
# tests/test_basic.py
"""
Basic smoke tests for core components.
Not comprehensive, but ensures basic functionality works.

Usage:
    python tests/test_basic.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import (
    AppConfig, DocumentMeta, PageArtifact, PageText, Block,
    IndexUnit, QueryInput, RetrieveHit, EvidenceItem
)
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator


def test_schemas():
    """Test basic schema creation."""
    print("Testing schemas...")
    
    # AppConfig
    config = AppConfig()
    assert config.data_root == "data"
    assert config.chunk_level == "block"
    
    # DocumentMeta
    meta = DocumentMeta(
        doc_id="test_doc",
        title="Test Document",
        source_path="/tmp/test.pdf",
        created_at="2026-01-06T00:00:00Z",
        page_count=10
    )
    assert meta.doc_id == "test_doc"
    
    print("✅ Schemas OK")


def test_store(tmp_path):
    """Test document store."""
    print("Testing document store...")
    
    config = AppConfig(docs_dir=str(tmp_path / "docs"))
    store = DocumentStoreLocal(config)
    
    # Save document
    meta = DocumentMeta(
        doc_id="test_doc",
        title="Test",
        source_path="/tmp/test.pdf",
        created_at="2026-01-06T00:00:00Z",
        page_count=1
    )
    store.save_document(meta)
    
    # Load document
    loaded = store.get_document("test_doc")
    assert loaded is not None
    assert loaded.doc_id == "test_doc"
    
    # List documents
    docs = store.list_documents()
    assert len(docs) == 1
    
    print("✅ Store OK")


def test_selector():
    """Test evidence selector."""
    print("Testing evidence selector...")
    
    from core.schemas import RetrievalResult
    
    selector = TopKEvidenceSelector(snippet_length=100)
    
    # Create mock retrieval result
    query = QueryInput(query_id="test", question="test question")
    hits = [
        RetrieveHit(
            unit_id=f"unit_{i}",
            doc_id="doc1",
            page_id=i,
            text="This is a test text " * 20,
            score=1.0 / (i + 1),
            source="bm25"
        )
        for i in range(10)
    ]
    retrieval = RetrievalResult(query_id="test", hits=hits, elapsed_ms=100)
    
    config = AppConfig(top_k_evidence=5)
    result = selector.select(query, retrieval, config)
    
    assert len(result.evidence) == 5
    assert result.evidence[0].rank == 0
    assert len(result.evidence[0].snippet) <= 103  # 100 + "..."
    
    print("✅ Selector OK")


def test_generator():
    """Test template generator."""
    print("Testing generator...")
    
    from core.schemas import GenerationRequest
    
    generator = TemplateGenerator(mode="summary")
    
    evidence = [
        EvidenceItem(
            rank=i,
            unit_id=f"unit_{i}",
            doc_id="doc1",
            page_id=i,
            snippet=f"Evidence text {i}",
            score=1.0,
            rationale="test"
        )
        for i in range(3)
    ]
    
    req = GenerationRequest(
        query_id="test",
        question="What is the topic?",
        evidence=evidence,
        config=AppConfig()
    )
    
    result = generator.generate(req)
    assert result.output.answer != ""
    assert len(result.output.cited_units) == 3
    
    print("✅ Generator OK")


def main():
    """Run all tests."""
    import tempfile
    
    print("=" * 60)
    print("Running Basic Smoke Tests")
    print("=" * 60 + "\n")
    
    try:
        test_schemas()
        
        with tempfile.TemporaryDirectory() as tmp:
            test_store(Path(tmp))
        
        test_selector()
        test_generator()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
