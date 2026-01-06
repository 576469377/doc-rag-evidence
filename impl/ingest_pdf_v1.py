"""
Enhanced PDF ingestor with OCR and unified block building.
V0.1+: Supports page rendering, OCR, and standardized block generation.
"""
from pathlib import Path
from typing import List, Optional
import hashlib
from datetime import datetime

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from core.schemas import DocumentMeta, PageArtifact, PageText, AppConfig
from infra.store_local import DocumentStoreLocal
from impl.ocr_client import OcrProvider, MockOcrClient, SGLangOcrClient
from impl.block_builder import BlockBuilder


class PDFIngestorV1:
    """
    Enhanced PDF ingestor with OCR support.
    
    Pipeline:
    1. Render page to PNG using PyMuPDF
    2. Extract text via OCR (optional, cache-aware)
    3. Build standardized blocks via BlockBuilder
    4. Save all artifacts (page.png, ocr.json, blocks.json)
    """
    
    def __init__(
        self,
        config: AppConfig,
        store: DocumentStoreLocal,
        ocr_client: Optional[OcrProvider] = None,
        use_ocr: bool = True
    ):
        """
        Initialize enhanced PDF ingestor.
        
        Args:
            config: Application configuration
            store: Document store for persistence
            ocr_client: OCR provider (optional, will create from config if None)
            use_ocr: Whether to use OCR (if False, falls back to pdfplumber text extraction)
        """
        if pdfplumber is None:
            raise ImportError("pdfplumber is required. Install with: pip install pdfplumber")
        if fitz is None:
            raise ImportError("PyMuPDF is required. Install with: pip install pymupdf")
        
        self.config = config
        self.store = store
        self.use_ocr = use_ocr
        
        # Initialize OCR client if not provided
        if ocr_client is None and use_ocr:
            ocr_config = config.ocr
            provider = ocr_config.get("provider", "mock")
            
            if provider in ["sglang", "vllm"]:
                # Both vLLM and SGLang use OpenAI-compatible API
                self.ocr_client = SGLangOcrClient(
                    endpoint=ocr_config["endpoint"],
                    model=ocr_config["model"],
                    timeout=ocr_config.get("timeout", 60),
                    cache_dir=Path(config.docs_dir) if ocr_config.get("cache_enabled") else None
                )
                print(f"✅ Initialized OCR client: {provider} @ {ocr_config['endpoint']}")
            elif provider == "mock":
                self.ocr_client = MockOcrClient()
                print("⚠️  Using MockOcrClient (returns empty text)")
            else:
                print(f"❌ Unknown OCR provider: {provider}, falling back to MockOcrClient")
                self.ocr_client = MockOcrClient()
        else:
            self.ocr_client = ocr_client or MockOcrClient()
        
        # Initialize block builder
        self.block_builder = BlockBuilder()
    
    def ingest(self, pdf_path: str, doc_id: Optional[str] = None) -> DocumentMeta:
        """
        Ingest a PDF document with OCR support.
        
        Args:
            pdf_path: Path to PDF file
            doc_id: Optional document ID (generated from filename if None)
            
        Returns:
            DocumentMeta object
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Generate doc_id from filename if not provided
        if doc_id is None:
            doc_id = pdf_path.stem.replace(" ", "_").lower()
        
        # Compute SHA256 hash
        sha256 = self._compute_sha256(pdf_path)
        
        # Count pages
        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
        
        # Create document metadata
        meta = DocumentMeta(
            doc_id=doc_id,
            title=pdf_path.stem,
            source_path=str(pdf_path.absolute()),
            sha256=sha256,
            created_at=datetime.utcnow().isoformat() + "Z",
            page_count=page_count
        )
        
        # Save document metadata
        self.store.save_document(meta)
        
        # Build page artifacts
        artifacts = self.build_page_artifacts(meta, self.config)
        
        print(f"Ingested {doc_id}: {page_count} pages, {sum(len(a.blocks) for a in artifacts)} blocks")
        return meta
    
    def build_page_artifacts(self, meta: DocumentMeta, config: AppConfig) -> List[PageArtifact]:
        """
        Build page artifacts with OCR and standardized blocks.
        
        Args:
            meta: Document metadata
            config: Application configuration
            
        Returns:
            List of PageArtifact objects
        """
        doc_id = meta.doc_id
        artifacts = []
        source_path = Path(meta.source_path)
        
        # Open PDF with PyMuPDF for page rendering
        pdf_doc = fitz.open(source_path)
        
        # Open with pdfplumber for fallback text extraction
        with pdfplumber.open(source_path) as pdf:
            for page_idx in range(len(pdf.pages)):
                page_id = page_idx  # 0-indexed
                
                print(f"Processing {doc_id} page {page_id}...")
                
                # 1. Render page image
                page_image_path = self._render_page_image(pdf_doc, doc_id, page_id)
                
                # 2. Extract text (OCR or pdfplumber fallback)
                if self.use_ocr and page_image_path:
                    # Use OCR
                    ocr_result = self.ocr_client.ocr_page(page_image_path)
                    text_content = ocr_result.text
                    
                    # Build blocks from OCR
                    blocks = self.block_builder.build_from_ocr(doc_id, page_id, ocr_result)
                else:
                    # Fallback to pdfplumber
                    page = pdf.pages[page_idx]
                    text_content = page.extract_text() or ""
                    
                    # Build blocks from text
                    if config.chunk_level == "block":
                        blocks = self.block_builder.build_from_text(doc_id, page_id, text_content)
                    else:
                        blocks = []
                
                # 3. Create page text object
                page_text = PageText(
                    doc_id=doc_id,
                    page_id=page_id,
                    text=text_content,
                    language=None  # Let model handle multi-language
                )
                
                # 4. Create page artifact
                artifact = PageArtifact(
                    doc_id=doc_id,
                    page_id=page_id,
                    text=page_text,
                    blocks=blocks,
                    image_path=page_image_path
                )
                
                # 5. Save artifact
                self.store.save_page_artifact(artifact)
                artifacts.append(artifact)
        
        pdf_doc.close()
        return artifacts
    
    def _render_page_image(
        self,
        pdf_doc,
        doc_id: str,
        page_id: int,
        dpi_scale: float = 2.0
    ) -> Optional[str]:
        """
        Render page to PNG using PyMuPDF.
        
        Args:
            pdf_doc: PyMuPDF document object
            doc_id: Document ID
            page_id: Page ID
            dpi_scale: DPI scaling factor (2.0 = 144 DPI)
            
        Returns:
            Path to rendered page image, or None if failed
        """
        try:
            page_dir = self.store._get_page_dir(doc_id, page_id)
            page_dir.mkdir(parents=True, exist_ok=True)
            
            # Render at specified DPI
            mat = fitz.Matrix(dpi_scale, dpi_scale)
            pix = pdf_doc[page_id].get_pixmap(matrix=mat)
            
            # Save to page.png
            image_path = page_dir / "page.png"
            pix.save(str(image_path))
            
            return str(image_path)
            
        except Exception as e:
            print(f"Warning: Failed to render page {page_id}: {e}")
            return None
    
    def _compute_sha256(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
