"""
LLM-based generator using Qwen models via vLLM/transformers.
V1.1: Text-only RAG with citation enforcement.
"""
from typing import List, Optional
import requests
import json
import time

from core.schemas import GenerationRequest, GenerationResult, GenerationOutput, EvidenceItem, AppConfig


class QwenLLMGenerator:
    """
    Generator using Qwen LLM for answer generation.
    
    Supports:
    - vLLM backend (HTTP API)
    - Transformers backend (local)
    - Citation enforcement
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize generator.
        
        Args:
            config: Application config with LLM settings
        """
        self.config = config
        self.llm_config = config.llm
        self.backend = self.llm_config.get("backend", "vllm")
        
        if self.backend == "transformers":
            self._init_transformers()
        elif self.backend in ["vllm", "sglang"]:
            self.endpoint = self.llm_config["endpoint"]
            self.model = self.llm_config["model"]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        print(f"✅ Initialized QwenLLMGenerator (backend={self.backend})")
    
    def _init_transformers(self):
        """Initialize transformers backend (lazy import)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            model_path = self.llm_config.get("model_path") or self.llm_config["model"]
            
            print(f"Loading model from {model_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            self.device = self.model.device
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load transformers model: {e}")
            raise
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Generate answer based on evidence.
        
        Args:
            request: Generation request with question and evidence
            
        Returns:
            Generation result with answer and citations
        """
        start_time = time.time()
        
        # Build prompt
        prompt = self._build_prompt(request)
        
        # Generate
        if self.backend == "transformers":
            answer, cited_units = self._generate_transformers(prompt, request.evidence)
        else:
            answer, cited_units = self._generate_api(prompt, request.evidence)
        
        # Ensure citations if policy is strict
        citation_policy = self.llm_config.get("citation_policy", "strict")
        if citation_policy == "strict" and not cited_units:
            # Fallback: cite all evidence
            cited_units = [ev.unit_id for ev in request.evidence]
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return GenerationResult(
            query_id=request.query_id,
            output=GenerationOutput(
                answer=answer,
                cited_units=cited_units
            ),
            prompt=None,  # Don't save prompt by default (can enable for debug)
            elapsed_ms=elapsed_ms
        )
    
    def _build_prompt(self, request: GenerationRequest) -> str:
        """
        Build prompt with evidence and citation instructions.
        
        Format:
        System: You are a helpful assistant...
        User: Context: [evidence] Question: [question]
        """
        # Format evidence with citation markers
        evidence_texts = []
        for i, ev in enumerate(request.evidence, 1):
            # Use snippet if available, otherwise full text
            text = ev.snippet if ev.snippet else ""
            if not text:
                continue
            
            evidence_texts.append(f"[{i}] {text}")
        
        context = "\n\n".join(evidence_texts)
        
        # Build messages
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
        
        # Format for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def _generate_api(self, messages: List[dict], evidence: List[EvidenceItem]) -> tuple:
        """
        Generate using vLLM/SGLang API.
        
        Returns:
            (answer, cited_units)
        """
        # Convert messages to prompt string for Qwen3-VL
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(prompt_parts)
        
        url = f"{self.endpoint}/v1/completions"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.llm_config.get("max_new_tokens", 1024),
            "temperature": self.llm_config.get("temperature", 0.1),
            "top_p": self.llm_config.get("top_p", 0.9),
            "stop": ["<|im_end|>", "<|endoftext|>"]
        }
        
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            answer = result["choices"][0]["text"].strip()
            
            # Extract citations from answer
            cited_units = self._extract_citations(answer, evidence)
            
            return answer, cited_units
            
        except Exception as e:
            print(f"❌ API generation failed: {e}")
            # Fallback: simple concatenation
            return self._fallback_answer(evidence), [ev.unit_id for ev in evidence]
    
    def _generate_transformers(self, messages: List[dict], evidence: List[EvidenceItem]) -> tuple:
        """
        Generate using transformers backend.
        
        Returns:
            (answer, cited_units)
        """
        try:
            # Apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.llm_config.get("max_new_tokens", 1024),
                temperature=self.llm_config.get("temperature", 0.1),
                top_p=self.llm_config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            answer = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Extract citations
            cited_units = self._extract_citations(answer, evidence)
            
            return answer, cited_units
            
        except Exception as e:
            print(f"❌ Transformers generation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_answer(evidence), [ev.unit_id for ev in evidence]
    
    def _extract_citations(self, answer: str, evidence: List[EvidenceItem]) -> List[str]:
        """
        Extract citation markers [1], [2] etc from answer and map to unit_ids.
        
        Returns:
            List of cited unit_ids
        """
        import re
        
        # Find all [N] patterns
        citations = re.findall(r'\[(\d+)\]', answer)
        
        # Map to unit_ids
        cited_units = []
        for cite in citations:
            idx = int(cite) - 1  # 1-indexed
            if 0 <= idx < len(evidence):
                cited_units.append(evidence[idx].unit_id)
        
        return list(set(cited_units))  # Remove duplicates
    
    def _fallback_answer(self, evidence: List[EvidenceItem]) -> str:
        """Generate simple fallback answer when LLM fails."""
        snippets = []
        for i, ev in enumerate(evidence, 1):
            text = ev.snippet if ev.snippet else ""
            if text:
                snippets.append(f"[{i}] {text[:200]}...")
        
        return "根据检索到的相关证据：\n\n" + "\n\n".join(snippets)
