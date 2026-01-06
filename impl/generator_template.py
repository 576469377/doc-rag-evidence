# impl/generator_template.py
"""
Template-based Generator V0 implementation.

V0 uses rule-based generation for offline demo:
  - No real LLM calls
  - Template-based answer generation
  - Always includes citations
  - Demonstrates the full pipeline without API dependencies

Future: Can replace with real LLM (OpenAI, Anthropic, etc.)
"""
from __future__ import annotations

import time
from typing import List

from core.schemas import (
    GenerationRequest, GenerationOutput, GenerationResult, 
    PromptPackage, EvidenceItem
)


class TemplateGenerator:
    """Template-based generator for V0 demo."""

    def __init__(self, mode: str = "summary"):
        """
        Args:
            mode: Generation mode
                - "summary": Summarize evidence with citations
                - "extract": Extract key points with citations
        """
        self.mode = mode

    def generate(self, req: GenerationRequest) -> GenerationResult:
        """
        Generate answer based on evidence.
        
        V0 template strategy:
          1. Build context from evidence
          2. Generate template-based answer with citations
          3. Map citations back to evidence unit_ids
          
        Args:
            req: Generation request
            
        Returns:
            GenerationResult with answer and citations
        """
        start_time = time.time()

        # Build prompt (for logging/traceability)
        prompt = self._build_prompt(req)

        # Generate answer
        output = self._generate_template_answer(req)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return GenerationResult(
            query_id=req.query_id,
            output=output,
            prompt=prompt,
            elapsed_ms=elapsed_ms
        )

    def _build_prompt(self, req: GenerationRequest) -> PromptPackage:
        """Build prompt package (for traceability)."""
        
        # System prompt
        system_prompt = (
            "You are a helpful assistant that answers questions based on provided evidence. "
            "Always cite your sources using [1], [2], etc. "
            "If the evidence doesn't contain the answer, say so clearly."
        )

        # Context from evidence
        context_parts = []
        for i, ev in enumerate(req.evidence, 1):
            context_parts.append(
                f"[{i}] (Doc: {ev.doc_id}, Page: {ev.page_id})\n{ev.snippet}"
            )
        context = "\n\n".join(context_parts)

        # User prompt
        user_prompt = (
            f"Question: {req.question}\n\n"
            f"Based on the following evidence, provide a comprehensive answer:\n\n"
            f"{context}"
        )

        return PromptPackage(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=context,
            citations=req.evidence
        )

    def _generate_template_answer(self, req: GenerationRequest) -> GenerationOutput:
        """
        Generate template-based answer.
        
        V0 strategy:
          - If no evidence: return "insufficient evidence" message
          - If evidence exists: generate summary with citations
        """
        evidence = req.evidence

        if not evidence:
            return GenerationOutput(
                answer="I couldn't find sufficient evidence to answer this question.",
                cited_units=[],
                warnings=["No evidence retrieved"]
            )

        # Generate answer with citations
        if self.mode == "summary":
            answer = self._generate_summary(req.question, evidence)
        else:
            answer = self._generate_extract(req.question, evidence)

        # Collect all cited unit_ids
        cited_units = [ev.unit_id for ev in evidence]

        return GenerationOutput(
            answer=answer,
            cited_units=cited_units,
            warnings=[]
        )

    def _generate_summary(self, question: str, evidence: List[EvidenceItem]) -> str:
        """Generate a summary-style answer with citations."""
        
        # Build answer parts
        parts = [
            f"Based on the retrieved evidence, here's what I found regarding '{question}':\n"
        ]

        # Add evidence snippets with citations
        for i, ev in enumerate(evidence, 1):
            # Truncate snippet for answer
            snippet = ev.snippet[:200] + "..." if len(ev.snippet) > 200 else ev.snippet
            parts.append(f"[{i}] {snippet}")

        # Add summary statement
        parts.append(
            f"\n\nThis information is drawn from {len(evidence)} document section(s). "
            f"For full context, please refer to the evidence citations above."
        )

        return "\n\n".join(parts)

    def _generate_extract(self, question: str, evidence: List[EvidenceItem]) -> str:
        """Generate an extract-style answer (key points with citations)."""
        
        parts = [
            f"Key points related to '{question}':\n"
        ]

        for i, ev in enumerate(evidence, 1):
            # First sentence or first 150 chars
            snippet = ev.snippet.split('.')[0] + '.'
            if len(snippet) > 150:
                snippet = ev.snippet[:150] + "..."
            
            parts.append(f"{i}. {snippet} [{i}]")

        parts.append(
            f"\n\nNote: The above {len(evidence)} points are extracted from the retrieved evidence. "
            f"Numbers in brackets indicate the evidence source."
        )

        return "\n".join(parts)
