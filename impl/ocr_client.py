"""
OCR provider client for extracting text from page images.
Supports multiple OCR backends via SGLang API.
"""
from typing import Protocol, List, Dict, Optional, Any
from pathlib import Path
import json
import base64
from io import BytesIO
import requests
from PIL import Image

from core.schemas import PageText


class OcrResult:
    """Result from OCR processing."""
    def __init__(
        self,
        text: str,
        blocks: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.blocks = blocks or []  # List of {text, bbox, confidence, ...}
        self.metadata = metadata or {}


class OcrProvider(Protocol):
    """Protocol for OCR providers."""
    
    def ocr_page(self, image_path: str) -> OcrResult:
        """
        Run OCR on a page image.
        
        Args:
            image_path: Path to page image file
            
        Returns:
            OcrResult with extracted text and optional block information
        """
        ...


class SGLangOcrClient:
    """OCR client using SGLang-served vision models."""
    
    def __init__(
        self,
        endpoint: str,
        model: str = "deepseek_ocr",
        timeout: int = 60,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize SGLang OCR client.
        
        Args:
            endpoint: SGLang server endpoint (e.g., http://127.0.0.1:30000)
            model: Model name (deepseek_ocr, hunyuan_ocr, etc.)
            timeout: Request timeout in seconds
            cache_dir: Optional directory to cache OCR results
        """
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
    def _get_cache_path(self, image_path: str) -> Optional[Path]:
        """Get cache file path for an image."""
        if not self.cache_dir:
            return None
        
        image_path = Path(image_path)
        # Use image path structure to create cache path
        # e.g., data/docs/{doc_id}/pages/{page_id:04d}/page.png -> ocr.json
        if "pages" in image_path.parts:
            cache_path = image_path.parent / "ocr.json"
            return cache_path
        return None
    
    def _load_cached_result(self, image_path: str) -> Optional[OcrResult]:
        """Load cached OCR result if available."""
        cache_path = self._get_cache_path(image_path)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return OcrResult(
                    text=data.get("text", ""),
                    blocks=data.get("blocks", []),
                    metadata=data.get("metadata", {})
                )
            except Exception as e:
                print(f"Warning: Failed to load cached OCR result: {e}")
        return None
    
    def _save_cached_result(self, image_path: str, result: OcrResult):
        """Save OCR result to cache."""
        cache_path = self._get_cache_path(image_path)
        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "text": result.text,
                        "blocks": result.blocks,
                        "metadata": result.metadata
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Warning: Failed to save cached OCR result: {e}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if too large (max 2048px on longest side)
            max_size = 2048
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Encode to base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def ocr_page(self, image_path: str) -> OcrResult:
        """
        Run OCR on a page image using vllm/SGLang API (OpenAI-compatible).
        
        Args:
            image_path: Path to page image
            
        Returns:
            OcrResult with extracted text
        """
        # Check cache first
        cached = self._load_cached_result(image_path)
        if cached:
            return cached
        
        # Encode image
        try:
            image_b64 = self._encode_image(image_path)
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return OcrResult(text="", metadata={"error": str(e)})
        
        # Prepare request
        # Use OpenAI-compatible chat completions API
        url = f"{self.endpoint}/v1/chat/completions"
        
        # HunyuanOCR optimized prompt
        prompt = (
            "Extract all information from the main body of the document image "
            "and represent it in markdown format, ignoring headers and footers. "
            "Tables should be expressed in HTML format, formulas in the document "
            "should be represented using LaTeX format, and the parsing should be "
            "organized according to the reading order."
        )
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": ""},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096
        }
        
        # Add extra_body for vllm (optional, won't break SGLang)
        if "hunyuan" in self.model.lower():
            payload["extra_body"] = {
                "top_k": 1,
                "repetition_penalty": 1.0
            }
        
        # Make request
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result_data = response.json()
            text = result_data["choices"][0]["message"]["content"]
            
            result = OcrResult(
                text=text,
                metadata={
                    "model": self.model,
                    "image_path": str(image_path),
                    "usage": result_data.get("usage", {})
                }
            )
            
            # Cache result
            self._save_cached_result(image_path, result)
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling OCR API: {e}")
            return OcrResult(text="", metadata={"error": str(e)})
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            print(f"Error parsing OCR response: {e}")
            return OcrResult(text="", metadata={"error": str(e)})


class MockOcrClient:
    """Mock OCR client for testing (returns empty text)."""
    
    def ocr_page(self, image_path: str) -> OcrResult:
        """Return empty OCR result."""
        return OcrResult(
            text="",
            metadata={"provider": "mock"}
        )
