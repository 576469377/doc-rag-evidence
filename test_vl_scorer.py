#!/usr/bin/env python3
"""
Quick test script for VL-based evaluation.
Tests the eval_vl_scorer independently.
"""
import sys
sys.path.insert(0, '/workspace/doc-rag-evidence')

from impl.eval_vl_scorer import VLAnswerScorer
from core.schemas import AppConfig
import yaml

# Load config
with open('configs/app.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
config = AppConfig(**config_dict)

# Create scorer
scorer = VLAnswerScorer(config)

# Test case
question = "食品接触用塑料材料的总迁移量限量是多少？"
answer_pred = "根据GB 4806.7-2023标准，食品接触用塑料材料及制品的总迁移量应≤10 mg/dm²。"
answer_gt = "根据GB 4806.7-2023标准，食品接触用塑料材料及制品的总迁移量应≤10 mg/dm²。"

print("Testing VL Answer Scorer...")
print(f"Question: {question}")
print(f"Predicted: {answer_pred}")
print(f"Ground Truth: {answer_gt}")
print("\nCalling VL model for scoring...")

score_result = scorer.score_answer(question, answer_pred, answer_gt)

if score_result:
    print(f"\n✅ Score: {score_result.score:.1f}/10")
    print(f"Correctness: {score_result.correctness}")
    print(f"Reasoning: {score_result.reasoning}")
else:
    print("❌ Scoring failed")
