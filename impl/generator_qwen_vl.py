"""
Multi-modal LLM generator using Qwen3-VL with image inputs.
V2: Supports both text snippets and full page images as evidence.
"""
from typing import List, Optional, Union
from pathlib import Path
import requests
import json
import time
import base64

from core.schemas import GenerationRequest, GenerationResult, GenerationOutput, EvidenceItem, AppConfig


class QwenVLGenerator:
    """
    Multi-modal generator using Qwen3-VL.
    
    Supports:
    - Text-only mode (like V1)
    - Image mode: uses full page images instead of text snippets
    - Mixed mode: combines images and text
    """
    
    def __init__(self, config: AppConfig, use_images: bool = False):
        """
        Initialize generator.
        
        Args:
            config: Application config with LLM settings
            use_images: If True, use page images instead of text snippets
        """
        self.config = config
        self.llm_config = config.llm
        self.use_images = use_images
        self.endpoint = self.llm_config["endpoint"]
        self.model = self.llm_config["model"]
        
        print(f"✅ Initialized QwenVLGenerator (use_images={use_images})")
    
    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _get_page_image_path(self, doc_id: str, page_id: int) -> Optional[str]:
        """
        Get path to page image.
        
        Args:
            doc_id: Document ID
            page_id: Page ID
            
        Returns:
            Path to page.png or None if not found
        """
        docs_dir = Path(self.config.docs_dir)
        image_path = docs_dir / doc_id / "pages" / f"{page_id:04d}" / "page.png"
        
        if image_path.exists():
            return str(image_path)
        return None
    
    def _deduplicate_pages(self, evidence: List[EvidenceItem]) -> List[tuple]:
        """
        Deduplicate evidence by (doc_id, page_id).
        
        Returns:
            List of unique (doc_id, page_id, evidence_indices) tuples
        """
        page_to_indices = {}
        for i, ev in enumerate(evidence):
            key = (ev.doc_id, ev.page_id)
            if key not in page_to_indices:
                page_to_indices[key] = []
            page_to_indices[key].append(i + 1)  # 1-based citation number
        
        # Convert to list and sort by first occurrence
        pages = []
        for key, indices in page_to_indices.items():
            pages.append((key[0], key[1], indices))
        
        return pages
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate answer based on evidence.
        
        Args:
            request: Generation request with question and evidence
            
        Returns:
            Generation result with answer and citations
        """
        start_time = time.time()
        
        if self.use_images:
            answer, cited_units = self._generate_with_images(request)
        else:
            answer, cited_units = self._generate_with_text(request)
        
        # Ensure citations if policy is strict
        citation_policy = self.llm_config.get("citation_policy", "strict")
        if citation_policy == "strict" and not cited_units:
            cited_units = [ev.unit_id for ev in request.evidence]
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return GenerationResult(
            query_id=request.query_id,
            output=GenerationOutput(
                answer=answer,
                cited_units=cited_units
            ),
            prompt=None,
            elapsed_ms=elapsed_ms
        )
    
    def _generate_with_images(self, request: GenerationRequest) -> tuple:
        """
        Generate answer using page images as evidence.
        
        Instead of text snippets, sends full page images to VL model.
        Deduplicates pages (multiple snippets from same page → one image).
        
        Returns:
            (answer, cited_units)
        """
        # Deduplicate pages
        pages = self._deduplicate_pages(request.evidence)
        
        if not pages:
            return "无可用的证据页面。", []
        
        # Limit number of images to avoid token limit issues
        # With max_model_len=32768, we can handle more images
        # Each image typically uses 1000-2000 tokens depending on resolution
        MAX_IMAGES = 5
        if len(pages) > MAX_IMAGES:
            print(f"⚠️  Limiting from {len(pages)} pages to {MAX_IMAGES} image(s)")
            pages = pages[:MAX_IMAGES]
        
        # Build multi-modal messages
        # Note: We don't use system message to save tokens
        # Following Qwen3-VL documentation pattern
        
        # Build user content with images
        user_content = []
        
        # Add page images
        page_info = []
        for i, (doc_id, page_id, evidence_indices) in enumerate(pages, 1):
            image_path = self._get_page_image_path(doc_id, page_id)
            
            if image_path:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode_image_base64(image_path)}"
                    }
                })
                page_info.append(f"页面{i}: {doc_id} 第{page_id}页")
            else:
                print(f"⚠️  Image not found: {doc_id} page {page_id}")
        
        # Add text instruction (simplified to save tokens)
        pages_list = "\n".join(page_info)
        user_content.append({
            "type": "text",
            "text": f"证据：{pages_list}\n\n问题：{request.question}\n\n请基于图片内容回答。"
        })
        
        # Call vision API
        url = f"{self.endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_content}
            ],
            "max_tokens": self.llm_config.get("max_new_tokens", 2048),
            "temperature": 0.7,           # Qwen3-VL recommended
            "top_p": 0.8,                 # Qwen3-VL recommended
            "top_k": 20,                  # Qwen3-VL recommended
            "repetition_penalty": 1.0,    # Qwen3-VL recommended
            "presence_penalty": 1.5,      # Qwen3-VL recommended (prevents repetition)
        }
        
        num_images = len([c for c in user_content if c['type'] == 'image_url'])
        print(f"🔍 Sending vision API request with {num_images} images")
        
        # Calculate approximate payload size for debugging
        import sys
        payload_size_mb = sys.getsizeof(str(payload)) / (1024 * 1024)
        print(f"📦 Approximate payload size: {payload_size_mb:.2f} MB")
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Extract citations from answer
            # Map page numbers back to unit_ids
            cited_units = []
            for i, (doc_id, page_id, evidence_indices) in enumerate(pages, 1):
                if f"页面{i}" in answer or f"[{i}]" in answer:
                    # Add all evidence items from this page
                    for idx in evidence_indices:
                        if idx - 1 < len(request.evidence):
                            cited_units.append(request.evidence[idx - 1].unit_id)
            
            if not cited_units:
                # Fallback: cite all
                cited_units = [ev.unit_id for ev in request.evidence]
            
            return answer, cited_units
            
        except Exception as e:
            print(f"❌ Vision API generation failed: {e}")
            # Print response details for debugging
            if 'response' in locals():
                try:
                    error_detail = response.json()
                    print(f"API error response: {error_detail}")
                except:
                    print(f"API error response (raw): {response.text}")
            import traceback
            traceback.print_exc()
            return f"生成答案时出错: {str(e)}", []
    
    def _generate_with_text(self, request: GenerationRequest) -> tuple:
        """
        Generate answer using text snippets (original mode).
        
        Returns:
            (answer, cited_units)
        """
        # Format evidence with citation markers
        evidence_texts = []
        for i, ev in enumerate(request.evidence, 1):
            text = ev.snippet if ev.snippet else ""
            if not text:
                continue
            evidence_texts.append(f"[{i}] {text}")
        
        context = "\n\n".join(evidence_texts)
        
        system_prompt = (
            "你是一个专业的文档问答助手。请基于提供的证据回答问题。\n"
            "要求：\n"
            "1. 仅使用提供的证据回答，不要编造信息\n"
            "2. 必须在答案中引用证据，使用[1]、[2]等标记\n"
            "3. 如果证据不足以回答问题，明确说明\n"
            "4. 回答要准确、简洁、专业"
        )
        
        user_prompt = (
            f"## 证据材料\n\n{context}\n\n"
            f"## 问题\n\n{request.question}\n\n"
            f"请基于以上证据回答问题，并在答案中标注引用的证据编号（如[1][2]）。"
        )
        
        # Call completions API
        url = f"{self.endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": self.llm_config.get("max_new_tokens", 1024),
            "temperature": self.llm_config.get("temperature", 0.1),
            "top_p": self.llm_config.get("top_p", 0.9),
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Extract citations
            cited_units = []
            for i, ev in enumerate(request.evidence, 1):
                if f"[{i}]" in answer:
                    cited_units.append(ev.unit_id)
            
            if not cited_units:
                cited_units = [ev.unit_id for ev in request.evidence]
            
            return answer, cited_units
            
        except Exception as e:
            print(f"❌ Text API generation failed: {e}")
            return f"生成答案时出错: {str(e)}", []
