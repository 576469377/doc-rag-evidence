# impl/ingest_pdf.py
"""
PDF Ingestor V0 implementation using pdfplumber.
Extracts text at page level and optionally at block level.

V0 focuses on:
  - Text extraction (no OCR yet)
  - Page-level metadata
  - Basic block segmentation (paragraph-like chunks)
  - No bbox extraction (can add later)
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from core.schemas import (
    AppConfig, DocumentMeta, PageArtifact, PageText, Block
)
from infra.store_local import DocumentStoreLocal


class PDFIngestorV0:
    """PDF ingestor using pdfplumber for text extraction."""

    def __init__(self, store: DocumentStoreLocal):
        self.store = store
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber is required for PDF ingestion. "
                "Install with: pip install pdfplumber"
            )
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required for page rendering. "
                "Install with: pip install pymupdf"
            )

    def ingest(self, source_path: str, config: AppConfig) -> DocumentMeta:
        """
        Ingest a PDF file and create document metadata.
        
        Args:
            source_path: Path to PDF file
            config: App configuration
            
        Returns:
            DocumentMeta with basic info
        """
        source_path = Path(source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF not found: {source_path}")

        # Generate doc_id and compute hash
        doc_id = self._generate_doc_id(source_path)
        sha256 = self._compute_sha256(source_path)

        # Get page count
        with pdfplumber.open(source_path) as pdf:
            page_count = len(pdf.pages)

        # Create metadata
        meta = DocumentMeta(
            doc_id=doc_id,
            title=source_path.stem,
            source_path=str(source_path),
            sha256=sha256,
            created_at=datetime.now(timezone.utc).isoformat(),
            page_count=page_count,
            extra={}
        )

        # Save metadata
        self.store.save_document(meta)
        return meta

    def build_page_artifacts(self, doc_id: str, config: AppConfig) -> List[PageArtifact]:
        """
        Build page artifacts for all pages in a document.
        
        Extracts text and optionally segments into blocks.
        
        Args:
            doc_id: Document ID
            config: App configuration
            
        Returns:
            List of PageArtifact objects
        """
        meta = self.store.get_document(doc_id)
        if not meta:
            raise ValueError(f"Document not found: {doc_id}")

        artifacts = []
        source_path = Path(meta.source_path)

        # Open PDF with PyMuPDF for page rendering
        pdf_doc = fitz.open(source_path)

        with pdfplumber.open(source_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_id = page_idx  # 0-indexed

                # Extract text
                text_content = page.extract_text() or ""

                page_text = PageText(
                    doc_id=doc_id,
                    page_id=page_id,
                    text=text_content,
                    language="en"  # TODO: detect language
                )

                # Build blocks if chunk_level is "block"
                blocks = []
                if config.chunk_level == "block":
                    blocks = self._segment_text_to_blocks(
                        doc_id=doc_id,
                        page_id=page_id,
                        text=text_content
                    )

                # Render page image using PyMuPDF
                page_image_path = None
                try:
                    page_dir = self.store._get_page_dir(doc_id, page_id)
                    page_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Render at 2x DPI for better quality
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = pdf_doc[page_idx].get_pixmap(matrix=mat)
                    
                    image_path = page_dir / "page.png"
                    pix.save(str(image_path))
                    page_image_path = str(image_path)
                except Exception as e:
                    print(f"Warning: Failed to render page {page_id}: {e}")

                artifact = PageArtifact(
                    doc_id=doc_id,
                    page_id=page_id,
                    text=page_text,
                    blocks=blocks,
                    image_path=page_image_path
                )

                # Save artifact
                self.store.save_page_artifact(artifact)
                artifacts.append(artifact)

        pdf_doc.close()
        return artifacts

    def _generate_doc_id(self, path: Path) -> str:
        """Generate a unique document ID."""
        # Use filename + timestamp for uniqueness
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{path.stem}_{timestamp}_{uuid.uuid4().hex[:8]}"

    def _compute_sha256(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _segment_text_to_blocks(
        self, doc_id: str, page_id: int, text: str
    ) -> List[Block]:
        """
        Segment page text into blocks (simple paragraph-based).
        
        V0 strategy: split by double newlines (paragraph breaks).
        """
        blocks = []
        paragraphs = text.split("\n\n")

        for block_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            block_id = f"p{page_id:04d}_b{block_idx:04d}"

            block = Block(
                doc_id=doc_id,
                page_id=page_id,
                block_id=block_id,
                text=para,
                bbox=None,  # V0: no bbox
                block_type="paragraph"
            )
            blocks.append(block)

        return blocks
