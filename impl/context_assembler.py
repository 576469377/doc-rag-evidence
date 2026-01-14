# impl/context_assembler.py
"""Context assembler for evidence preparation."""

from typing import List, Dict, Tuple
from collections import defaultdict
from core.schemas import EvidenceItem


class ContextAssembler:
    """
    Assembles evidence items into context for generation.
    
    Responsibilities:
    - Deduplication: merge overlapping/duplicate evidence
    - Truncation: fit within token/char budget
    - Citation mapping: stable unit_id -> [N] mapping
    - Snippet/full_text separation for UI vs generation
    """
    
    def __init__(
        self,
        max_context_chars: int = 8000,
        dedup_threshold: float = 0.9,
        prefer_full_text: bool = False
    ):
        """
        Initialize context assembler.
        
        Args:
            max_context_chars: Maximum total characters in context
            dedup_threshold: Jaccard similarity threshold for deduplication
            prefer_full_text: Use full_text if available (for generation quality)
        """
        self.max_context_chars = max_context_chars
        self.dedup_threshold = dedup_threshold
        self.prefer_full_text = prefer_full_text
    
    def assemble(
        self,
        evidence_items: List[EvidenceItem],
        store=None
    ) -> Tuple[str, List[EvidenceItem], Dict[str, int]]:
        """
        Assemble evidence into context string.
        
        Args:
            evidence_items: List of evidence items
            store: Optional store for fetching full_text
        
        Returns:
            (context_string, deduplicated_items, citation_map)
            - context_string: Formatted context with citations
            - deduplicated_items: Evidence after deduplication
            - citation_map: {unit_id: citation_number}
        """
        if not evidence_items:
            return "", [], {}
        
        # Step 1: Deduplicate
        deduped = self._deduplicate(evidence_items)
        
        # Step 2: Fetch full text if needed
        if self.prefer_full_text and store:
            deduped = self._enrich_with_full_text(deduped, store)
        
        # Step 3: Truncate to budget
        selected = self._truncate_to_budget(deduped)
        
        # Step 4: Build citation map and context string
        context_string, citation_map = self._format_context(selected)
        
        return context_string, selected, citation_map
    
    def _deduplicate(self, items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Deduplicate evidence items.
        
        Strategy:
        - Group by (doc_id, page_id)
        - Within each group, compute pairwise similarity
        - Merge highly similar items (keep higher score)
        """
        if len(items) <= 1:
            return items
        
        # Group by page
        page_groups = defaultdict(list)
        for item in items:
            key = (item.doc_id, item.page_id)
            page_groups[key].append(item)
        
        deduped = []
        
        for (doc_id, page_id), group in page_groups.items():
            if len(group) == 1:
                deduped.extend(group)
                continue
            
            # Sort by score descending
            group_sorted = sorted(group, key=lambda x: x.score, reverse=True)
            
            # Greedy deduplication: keep item if not similar to any kept item
            kept = []
            for item in group_sorted:
                is_duplicate = False
                for kept_item in kept:
                    similarity = self._compute_similarity(item.snippet, kept_item.snippet)
                    if similarity >= self.dedup_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    kept.append(item)
            
            deduped.extend(kept)
        
        # Sort by original rank/score
        deduped.sort(key=lambda x: x.score, reverse=True)
        
        return deduped
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute Jaccard similarity between two texts.
        
        Args:
            text1, text2: Text strings
        
        Returns:
            Jaccard similarity in [0, 1]
        """
        if not text1 or not text2:
            return 0.0
        
        # Tokenize by whitespace
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _enrich_with_full_text(
        self,
        items: List[EvidenceItem],
        store
    ) -> List[EvidenceItem]:
        """
        Enrich evidence items with full_text from store.
        
        For generation: use full_text if available (better context)
        For UI display: keep snippet (concise)
        """
        enriched = []
        
        for item in items:
            try:
                # Fetch block from store
                block = store.get_block(item.unit_id)
                
                if block and block.get("text"):
                    # Create new item with full text
                    full_text = block["text"]
                    enriched_item = EvidenceItem(
                        rank=item.rank,  # Preserve rank
                        unit_id=item.unit_id,
                        doc_id=item.doc_id,
                        page_id=item.page_id,
                        block_id=item.block_id,
                        snippet=item.snippet,  # Keep original snippet for UI
                        score=item.score,
                        source=item.source,
                        bbox=item.bbox,
                        metadata={
                            **item.metadata,
                            "full_text": full_text,  # Add full text
                            "full_text_length": len(full_text)
                        }
                    )
                    enriched.append(enriched_item)
                else:
                    enriched.append(item)
            except Exception as e:
                # Fallback: use original item
                enriched.append(item)
        
        return enriched
    
    def _truncate_to_budget(self, items: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        Truncate evidence list to fit within character budget.
        
        Strategy: Greedy selection by rank until budget exhausted
        """
        selected = []
        total_chars = 0
        
        for item in items:
            # Use full_text if available, otherwise snippet
            text = item.metadata.get("full_text", item.snippet)
            text_len = len(text) if text else 0
            
            # Add citation overhead: "[N] " prefix
            item_chars = text_len + 6  # "[123] " prefix
            
            if total_chars + item_chars <= self.max_context_chars:
                selected.append(item)
                total_chars += item_chars
            else:
                # Budget exhausted
                break
        
        return selected
    
    def _format_context(
        self,
        items: List[EvidenceItem]
    ) -> Tuple[str, Dict[str, int]]:
        """
        Format evidence items into context string with citations.
        
        Returns:
            (context_string, citation_map)
        """
        if not items:
            return "", {}
        
        context_parts = []
        citation_map = {}
        
        for idx, item in enumerate(items, start=1):
            # Build citation map
            citation_map[item.unit_id] = idx
            
            # Use full_text if available, otherwise snippet
            text = item.metadata.get("full_text", item.snippet)
            
            if not text:
                continue
            
            # Format: [N] text
            context_parts.append(f"[{idx}] {text}")
        
        context_string = "\n\n".join(context_parts)
        
        return context_string, citation_map


class ContextAssemblerV2(ContextAssembler):
    """
    Enhanced context assembler with table/figure awareness.
    
    For V1.3+: Handle structured content differently
    """
    
    def __init__(
        self,
        max_context_chars: int = 8000,
        dedup_threshold: float = 0.9,
        prefer_full_text: bool = False,
        table_expansion: bool = True
    ):
        super().__init__(max_context_chars, dedup_threshold, prefer_full_text)
        self.table_expansion = table_expansion
    
    def _enrich_with_full_text(
        self,
        items: List[EvidenceItem],
        store
    ) -> List[EvidenceItem]:
        """
        Enhanced enrichment with table/figure detection.
        
        For tables: fetch full table structure
        For figures: add caption context
        """
        enriched = super()._enrich_with_full_text(items, store)
        
        if not self.table_expansion:
            return enriched
        
        # Additional processing for tables
        for item in enriched:
            block = store.get_block(item.unit_id)
            if block and block.get("type") == "table":
                # Mark as table for special handling
                item.metadata["is_table"] = True
                # Could fetch table structure here
        
        return enriched
