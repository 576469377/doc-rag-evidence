"""
Unified block builder that converts various text sources into standardized blocks.
Supports both OCR results and text extraction outputs.
"""
from typing import List, Optional
import re

from core.schemas import Block
from impl.ocr_client import OcrResult


class BlockBuilder:
    """
    Unified block builder for creating standardized text blocks.
    Handles both bbox-rich OCR results and plain text inputs.
    """
    
    def __init__(
        self,
        min_block_len: int = 30,
        paragraph_sep_pattern: str = r'\n\s*\n'
    ):
        """
        Initialize block builder.
        
        Args:
            min_block_len: Minimum length for a block (chars)
            paragraph_sep_pattern: Regex pattern for paragraph separation
        """
        self.min_block_len = min_block_len
        self.paragraph_sep_pattern = paragraph_sep_pattern
    
    def build_from_ocr(
        self,
        doc_id: str,
        page_id: int,
        ocr_result: OcrResult
    ) -> List[Block]:
        """
        Build blocks from OCR result.
        
        If OCR result contains block-level info with bboxes, use them.
        Otherwise, segment the text into paragraphs.
        
        Args:
            doc_id: Document ID
            page_id: Page ID
            ocr_result: OCR result with text and optional blocks
            
        Returns:
            List of Block objects
        """
        # If OCR provides structured blocks with bboxes, use them
        if ocr_result.blocks:
            return self._build_from_ocr_blocks(doc_id, page_id, ocr_result.blocks)
        
        # Otherwise, segment text into blocks
        return self._build_from_text(doc_id, page_id, ocr_result.text)
    
    def build_from_text(
        self,
        doc_id: str,
        page_id: int,
        text: str
    ) -> List[Block]:
        """
        Build blocks from plain text by segmenting into paragraphs.
        
        Args:
            doc_id: Document ID
            page_id: Page ID
            text: Plain text content
            
        Returns:
            List of Block objects
        """
        return self._build_from_text(doc_id, page_id, text)
    
    def _build_from_ocr_blocks(
        self,
        doc_id: str,
        page_id: int,
        ocr_blocks: List[dict]
    ) -> List[Block]:
        """
        Build blocks from OCR block-level information.
        
        OCR blocks should have format:
        {
            "text": str,
            "bbox": [x0, y0, x1, y1],  # optional
            "confidence": float,  # optional
            ...
        }
        """
        blocks = []
        
        for block_idx, ocr_block in enumerate(ocr_blocks):
            text = ocr_block.get("text", "").strip()
            if not text or len(text) < self.min_block_len:
                continue
            
            block_id = f"{doc_id}__page{page_id:04d}__blk{block_idx:03d}"
            
            block = Block(
                block_id=block_id,
                doc_id=doc_id,
                page_id=page_id,
                block_idx=block_idx,
                text=text,
                bbox=ocr_block.get("bbox"),
                metadata={
                    "source": "ocr_blocks",
                    "confidence": ocr_block.get("confidence")
                }
            )
            blocks.append(block)
        
        return blocks
    
    def _build_from_text(
        self,
        doc_id: str,
        page_id: int,
        text: str
    ) -> List[Block]:
        """
        Build blocks by segmenting text into paragraphs.
        
        Uses paragraph separator pattern (typically double newlines).
        Filters out blocks that are too short.
        """
        if not text:
            return []
        
        # Split by paragraph separator
        segments = re.split(self.paragraph_sep_pattern, text)
        
        blocks = []
        block_idx = 0
        
        for seg in segments:
            seg = seg.strip()
            if not seg or len(seg) < self.min_block_len:
                continue
            
            block_id = f"{doc_id}__page{page_id:04d}__blk{block_idx:03d}"
            
            block = Block(
                block_id=block_id,
                doc_id=doc_id,
                page_id=page_id,
                block_idx=block_idx,
                text=seg,
                bbox=None,
                metadata={"source": "text_segmentation"}
            )
            blocks.append(block)
            block_idx += 1
        
        return blocks
    
    def merge_blocks(
        self,
        blocks: List[Block],
        max_block_len: int = 512
    ) -> List[Block]:
        """
        Merge consecutive small blocks to reach target length.
        Useful for creating more substantial retrieval units.
        
        Args:
            blocks: List of blocks to merge
            max_block_len: Target maximum length for merged blocks
            
        Returns:
            List of merged blocks with updated indices
        """
        if not blocks:
            return []
        
        merged = []
        current_texts = []
        current_start_idx = 0
        
        for block in blocks:
            current_texts.append(block.text)
            combined_len = sum(len(t) for t in current_texts)
            
            if combined_len >= max_block_len:
                # Create merged block
                merged_text = " ".join(current_texts)
                merged_block = Block(
                    block_id=f"{block.doc_id}__page{block.page_id:04d}__blk{len(merged):03d}",
                    doc_id=block.doc_id,
                    page_id=block.page_id,
                    block_idx=len(merged),
                    text=merged_text,
                    bbox=None,
                    metadata={"source": "merged", "original_indices": list(range(current_start_idx, block.block_idx + 1))}
                )
                merged.append(merged_block)
                
                # Reset
                current_texts = []
                current_start_idx = block.block_idx + 1
        
        # Handle remaining texts
        if current_texts:
            last_block = blocks[-1]
            merged_text = " ".join(current_texts)
            merged_block = Block(
                block_id=f"{last_block.doc_id}__page{last_block.page_id:04d}__blk{len(merged):03d}",
                doc_id=last_block.doc_id,
                page_id=last_block.page_id,
                block_idx=len(merged),
                text=merged_text,
                bbox=None,
                metadata={"source": "merged", "original_indices": list(range(current_start_idx, last_block.block_idx + 1))}
            )
            merged.append(merged_block)
        
        return merged
