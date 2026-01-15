# impl/eval_vl_scorer.py
"""
VL Model-based Answer Scorer.

Uses Qwen3-VL-Instruct to evaluate the quality of generated answers
against ground truth answers using LLM-as-a-judge approach.
"""
from __future__ import annotations

import json
import requests
from typing import Dict, Optional
from dataclasses import dataclass

from core.schemas import AppConfig


@dataclass
class AnswerScore:
    """Score result from VL model evaluation."""
    score: float  # 0-10 scale
    reasoning: str
    correctness: str  # "correct", "partial", "incorrect"
    

class VLAnswerScorer:
    """Use Qwen3-VL to score answer quality against ground truth."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.endpoint = config.llm.get("endpoint", "http://localhost:8002")
        self.model_name = config.llm.get("model", "Qwen3-VL-4B-Instruct")
        
    def score_answer(
        self,
        question: str,
        answer_pred: str,
        answer_gt: str
    ) -> Optional[AnswerScore]:
        """
        Score predicted answer against ground truth.
        
        Args:
            question: The original question
            answer_pred: Predicted answer from the system
            answer_gt: Ground truth answer
            
        Returns:
            AnswerScore with score (0-10) and reasoning
        """
        if not answer_gt or not answer_pred:
            return None
            
        # Construct evaluation prompt
        prompt = self._build_eval_prompt(question, answer_pred, answer_gt)
        
        try:
            # Call VL model API
            response = self._call_vl_api(prompt)
            
            # Parse response
            score_result = self._parse_score_response(response)
            return score_result
            
        except Exception as e:
            print(f"⚠️ VL scoring failed: {e}")
            return None
    
    def _build_eval_prompt(self, question: str, answer_pred: str, answer_gt: str) -> str:
        """Build evaluation prompt for VL model."""
        prompt = f"""你是一个专业的答案质量评估专家。请评估以下系统生成的答案相对于参考答案的质量。

**问题**：
{question}

**系统生成的答案**：
{answer_pred}

**参考答案（Ground Truth）**：
{answer_gt}

请从以下几个维度进行评估：
1. **准确性**：生成答案是否包含参考答案中的关键信息？
2. **完整性**：生成答案是否覆盖了参考答案的主要内容？
3. **相关性**：生成答案是否直接回答了问题？
4. **错误信息**：生成答案是否包含明显错误或与参考答案矛盾的内容？

请按照以下JSON格式输出评估结果：
```json
{{
  "score": <0-10之间的分数>,
  "correctness": "<correct/partial/incorrect>",
  "reasoning": "<详细的评分理由，说明优点和不足>"
}}
```

评分标准：
- 9-10分：完全正确，覆盖所有关键信息
- 7-8分：基本正确，覆盖大部分关键信息，可能有小的遗漏
- 5-6分：部分正确，包含部分关键信息但有明显遗漏
- 3-4分：相关但不够准确，关键信息缺失较多
- 0-2分：错误或完全不相关

请给出评估结果（只输出JSON，不要其他内容）："""
        
        return prompt
    
    def _call_vl_api(self, prompt: str) -> str:
        """Call VL model API."""
        url = f"{self.endpoint}/v1/chat/completions"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature for consistent evaluation
            "max_tokens": 512
        }
        
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            return content.strip()
        else:
            raise ValueError("Invalid API response format")
    
    def _parse_score_response(self, response: str) -> AnswerScore:
        """Parse VL model response to extract score."""
        # Try to extract JSON from response
        import re
        
        # Find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: treat entire response as potential JSON
                json_str = response
        
        try:
            data = json.loads(json_str)
            
            score = float(data.get("score", 0))
            score = max(0.0, min(10.0, score))  # Clamp to 0-10
            
            correctness = data.get("correctness", "unknown").lower()
            if correctness not in ["correct", "partial", "incorrect"]:
                # Map score to correctness if not provided
                if score >= 8:
                    correctness = "correct"
                elif score >= 5:
                    correctness = "partial"
                else:
                    correctness = "incorrect"
            
            reasoning = data.get("reasoning", "No reasoning provided")
            
            return AnswerScore(
                score=score,
                reasoning=reasoning,
                correctness=correctness
            )
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse JSON response: {e}")
            print(f"Response was: {response[:200]}")
            # Return default score
            return AnswerScore(
                score=0.0,
                reasoning="Failed to parse evaluation response",
                correctness="unknown"
            )
    
    def batch_score(
        self,
        questions: list,
        answers_pred: list,
        answers_gt: list
    ) -> list[Optional[AnswerScore]]:
        """Score multiple answers in batch."""
        scores = []
        total = len(questions)
        
        for i, (q, pred, gt) in enumerate(zip(questions, answers_pred, answers_gt), 1):
            print(f"  Scoring {i}/{total}...")
            score = self.score_answer(q, pred, gt)
            scores.append(score)
        
        return scores
