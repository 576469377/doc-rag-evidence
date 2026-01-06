# infra/store_local.py
"""
Local file-based DocumentStore implementation.
Stores documents as JSON files following the data/ artifact path convention.

Path structure:
  data/docs/{doc_id}/meta.json
  data/docs/{doc_id}/pages/{page_id:04d}/text.json
  data/docs/{doc_id}/pages/{page_id:04d}/blocks.json
  data/docs/{doc_id}/pages/{page_id:04d}/page.png (optional)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional

from core.schemas import DocumentMeta, PageArtifact, AppConfig


class DocumentStoreLocal:
    """Local file-based document storage."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.docs_dir = Path(config.docs_dir)
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    def save_document(self, meta: DocumentMeta) -> None:
        """Save document metadata to meta.json."""
        doc_dir = self.docs_dir / meta.doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        meta_path = doc_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.model_dump(), f, indent=2, ensure_ascii=False)

    def get_document(self, doc_id: str) -> Optional[DocumentMeta]:
        """Load document metadata from meta.json."""
        meta_path = self.docs_dir / doc_id / "meta.json"
        if not meta_path.exists():
            return None

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DocumentMeta(**data)

    def list_documents(self) -> List[DocumentMeta]:
        """List all documents by scanning for meta.json files."""
        docs = []
        if not self.docs_dir.exists():
            return docs

        for doc_dir in self.docs_dir.iterdir():
            if doc_dir.is_dir():
                meta_path = doc_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    docs.append(DocumentMeta(**data))
        return docs

    def delete_document(self, doc_id: str) -> None:
        """Delete entire document directory."""
        doc_dir = self.docs_dir / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)

    def save_page_artifact(self, artifact: PageArtifact) -> None:
        """
        Save page artifact to the standard directory structure.
        
        Saves:
          - text.json (PageText if present)
          - blocks.json (list of Block if present)
        """
        page_dir = self._get_page_dir(artifact.doc_id, artifact.page_id)
        page_dir.mkdir(parents=True, exist_ok=True)

        # Save text
        if artifact.text is not None:
            text_path = page_dir / "text.json"
            with open(text_path, "w", encoding="utf-8") as f:
                json.dump(artifact.text.model_dump(), f, indent=2, ensure_ascii=False)

        # Save blocks
        if artifact.blocks:
            blocks_path = page_dir / "blocks.json"
            blocks_data = [b.model_dump() for b in artifact.blocks]
            with open(blocks_path, "w", encoding="utf-8") as f:
                json.dump(blocks_data, f, indent=2, ensure_ascii=False)

        # Copy page image if provided
        if artifact.image_path and Path(artifact.image_path).exists():
            dest_image = page_dir / "page.png"
            # Only copy if source and destination are different
            src_path = Path(artifact.image_path).resolve()
            dest_path = dest_image.resolve()
            if src_path != dest_path:
                shutil.copy2(artifact.image_path, dest_image)

    def load_page_artifact(self, doc_id: str, page_id: int) -> Optional[PageArtifact]:
        """Load page artifact from disk."""
        from core.schemas import PageText, Block

        page_dir = self._get_page_dir(doc_id, page_id)
        if not page_dir.exists():
            return None

        artifact = PageArtifact(doc_id=doc_id, page_id=page_id)

        # Load text
        text_path = page_dir / "text.json"
        if text_path.exists():
            with open(text_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            artifact.text = PageText(**data)

        # Load blocks
        blocks_path = page_dir / "blocks.json"
        if blocks_path.exists():
            with open(blocks_path, "r", encoding="utf-8") as f:
                blocks_data = json.load(f)
            artifact.blocks = [Block(**b) for b in blocks_data]

        # Check for page image
        image_path = page_dir / "page.png"
        if image_path.exists():
            artifact.image_path = str(image_path)

        return artifact

    def load_blocks(self, doc_id: str, page_id: int) -> List:
        """Load blocks for a specific page."""
        from core.schemas import Block
        
        page_dir = self._get_page_dir(doc_id, page_id)
        blocks_path = page_dir / "blocks.json"
        
        if not blocks_path.exists():
            return []
        
        with open(blocks_path, "r", encoding="utf-8") as f:
            blocks_data = json.load(f)
        
        return [Block(**b) for b in blocks_data]

    def _get_page_dir(self, doc_id: str, page_id: int) -> Path:
        """Get standardized page directory path."""
        return self.docs_dir / doc_id / "pages" / f"{page_id:04d}"

    def list_pages(self, doc_id: str) -> List[dict]:
        """List page metadata for a document."""
        from core.schemas import PageMeta
        pages = []
        pages_base = self.docs_dir / doc_id / "pages"
        if not pages_base.exists():
            return pages

        for page_dir in sorted(pages_base.iterdir()):
            if page_dir.is_dir():
                page_id = int(page_dir.name)
                # Create basic page metadata
                page_meta = PageMeta(
                    doc_id=doc_id,
                    page_id=page_id,
                    has_text=(page_dir / "text.json").exists(),
                    has_image=(page_dir / "page.png").exists()
                )
                pages.append(page_meta)
        return pages

    def get_all_pages(self, doc_id: str) -> List[PageArtifact]:
        """Load all page artifacts for a document."""
        pages = []
        pages_base = self.docs_dir / doc_id / "pages"
        if not pages_base.exists():
            return pages

        for page_dir in sorted(pages_base.iterdir()):
            if page_dir.is_dir():
                page_id = int(page_dir.name)
                artifact = self.load_page_artifact(doc_id, page_id)
                if artifact:
                    pages.append(artifact)
        return pages
