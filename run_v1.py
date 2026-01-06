#!/usr/bin/env python3
"""
Quick launcher for V0.1+ system.
Provides convenient shortcuts for common workflows.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import os

# Add project root to Python path for subprocess calls
project_root = Path(__file__).parent.absolute()
os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')


def run_command(cmd: list, description: str):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with code {result.returncode}")
        sys.exit(result.returncode)
    
    print(f"\n✅ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Quick launcher for doc-rag-evidence V0.1+",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest PDFs with OCR and build all indices
  python run_v1.py --ingest-dir data/raw_pdfs --use-ocr --build-all
  
  # Just build indices (after manual ingestion)
  python run_v1.py --build-all
  
  # Launch UI only
  python run_v1.py --ui
  
  # Full pipeline: ingest → index → UI
  python run_v1.py --ingest-dir data/raw_pdfs --build-all --ui
        """
    )
    
    # Ingestion options
    parser.add_argument(
        "--ingest-pdf",
        type=str,
        help="Ingest a single PDF file"
    )
    parser.add_argument(
        "--ingest-dir",
        type=str,
        help="Ingest all PDFs in directory"
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR during ingestion"
    )
    
    # Indexing options
    parser.add_argument(
        "--build-bm25",
        action="store_true",
        help="Build BM25 index"
    )
    parser.add_argument(
        "--build-dense",
        action="store_true",
        help="Build dense embedding index"
    )
    parser.add_argument(
        "--build-colpali",
        action="store_true",
        help="Build ColPali vision index"
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        help="Build all enabled indices"
    )
    
    # UI option
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Gradio UI"
    )
    parser.add_argument(
        "--ui-share",
        action="store_true",
        help="Create shareable Gradio link"
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/app.yaml",
        help="Path to config file (default: configs/app.yaml)"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not any([
        args.ingest_pdf,
        args.ingest_dir,
        args.build_bm25,
        args.build_dense,
        args.build_colpali,
        args.build_all,
        args.ui
    ]):
        parser.print_help()
        sys.exit(1)
    
    # Step 1: Ingestion
    if args.ingest_pdf or args.ingest_dir:
        cmd = ["python", "scripts/ingest_docs_v1.py", "--config", args.config]
        
        if args.ingest_pdf:
            cmd.extend(["--pdf", args.ingest_pdf])
        elif args.ingest_dir:
            cmd.extend(["--pdf-dir", args.ingest_dir])
        
        if args.use_ocr:
            cmd.append("--use-ocr")
        
        run_command(cmd, "Document Ingestion")
    
    # Step 2: Indexing
    if any([args.build_bm25, args.build_dense, args.build_colpali, args.build_all]):
        cmd = ["python", "scripts/build_indices_v1.py", "--config", args.config]
        
        if args.build_all:
            cmd.append("--all")
        else:
            if args.build_bm25:
                cmd.append("--bm25")
            if args.build_dense:
                cmd.append("--dense")
            if args.build_colpali:
                cmd.append("--colpali")
        
        run_command(cmd, "Index Building")
    
    # Step 3: UI
    if args.ui:
        cmd = ["python", "app/ui/main_v1.py", "--config", args.config]
        
        if args.ui_share:
            cmd.append("--share")
        
        print(f"\n{'='*60}")
        print("Launching UI...")
        print(f"Command: {' '.join(cmd)}")
        print("="*60)
        print("\n🚀 UI will be available at http://localhost:7860")
        print("Press Ctrl+C to stop\n")
        
        # Run UI (blocks until interrupted)
        subprocess.run(cmd)
    
    print("\n" + "="*60)
    print("✅ All tasks completed")
    print("="*60)


if __name__ == "__main__":
    main()
