#!/usr/bin/env python3
"""
Document ingestion script with V1 ingestor (OCR support).
"""
import argparse
from pathlib import Path
import yaml
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.schemas import AppConfig
from infra.store_local import DocumentStoreLocal
from impl.ingest_pdf_v1 import PDFIngestorV1


def load_config(config_path: str) -> AppConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return AppConfig(**config_dict)


def ingest_document(
    pdf_path: Path,
    config: AppConfig,
    store: DocumentStoreLocal,
    use_ocr: bool = False,
    doc_id: str = None
):
    """
    Ingest a single PDF document.
    
    Args:
        pdf_path: Path to PDF file
        config: Application config
        store: Document store
        use_ocr: Whether to use OCR
        doc_id: Optional document ID
    """
    print(f"\nIngesting: {pdf_path}")
    print(f"OCR: {'enabled' if use_ocr else 'disabled'}")
    
    ingestor = PDFIngestorV1(
        config=config,
        store=store,
        use_ocr=use_ocr
    )
    
    meta = ingestor.ingest(str(pdf_path), doc_id=doc_id)
    
    print(f"✅ Ingested {meta.doc_id}:")
    print(f"   - Pages: {meta.page_count}")
    print(f"   - SHA256: {meta.sha256[:16]}...")
    
    # Count blocks
    total_blocks = 0
    for page_id in range(meta.page_count):
        artifact = store.load_page_artifact(meta.doc_id, page_id)
        if artifact:
            total_blocks += len(artifact.blocks)
    
    print(f"   - Blocks: {total_blocks}")


def ingest_directory(
    pdf_dir: Path,
    config: AppConfig,
    store: DocumentStoreLocal,
    use_ocr: bool = False,
    pattern: str = "*.pdf"
):
    """
    Ingest all PDFs in a directory.
    
    Args:
        pdf_dir: Directory containing PDFs
        config: Application config
        store: Document store
        use_ocr: Whether to use OCR
        pattern: File pattern to match
    """
    pdf_files = sorted(pdf_dir.glob(pattern))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir} matching {pattern}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_path.name}")
        
        try:
            ingest_document(
                pdf_path=pdf_path,
                config=config,
                store=store,
                use_ocr=use_ocr
            )
        except Exception as e:
            print(f"❌ Error ingesting {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF documents with V1 ingestor")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/app.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to single PDF file to ingest"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        help="Path to directory containing PDFs"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR (slower but better for scanned PDFs)"
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        help="Custom document ID (only for single PDF)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pdf",
        help="File pattern for directory ingestion (default: *.pdf)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.pdf and not args.pdf_dir:
        parser.error("Either --pdf or --pdf-dir must be specified")
    
    if args.pdf and args.pdf_dir:
        parser.error("Cannot specify both --pdf and --pdf-dir")
    
    if args.doc_id and args.pdf_dir:
        parser.error("--doc-id can only be used with --pdf")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize store
    store = DocumentStoreLocal(config)
    
    try:
        if args.pdf:
            # Ingest single PDF
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
                sys.exit(1)
            
            ingest_document(
                pdf_path=pdf_path,
                config=config,
                store=store,
                use_ocr=args.use_ocr,
                doc_id=args.doc_id
            )
        
        elif args.pdf_dir:
            # Ingest directory
            pdf_dir = Path(args.pdf_dir)
            if not pdf_dir.is_dir():
                print(f"Error: Not a directory: {pdf_dir}", file=sys.stderr)
                sys.exit(1)
            
            ingest_directory(
                pdf_dir=pdf_dir,
                config=config,
                store=store,
                use_ocr=args.use_ocr,
                pattern=args.pattern
            )
        
        print("\n=== Ingestion complete ===")
        print("Next steps:")
        print("  1. Build indices: python scripts/build_indices_v1.py --all")
        print("  2. Launch UI: python app/ui/main_v1.py")
        
    except Exception as e:
        print(f"\nError during ingestion: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
