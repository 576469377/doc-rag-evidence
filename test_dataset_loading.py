#!/usr/bin/env python3
"""
Test evaluation dataset loading to verify answer_gt is read correctly.
"""
import sys
sys.path.insert(0, '/workspace/doc-rag-evidence')

from impl.eval_runner import load_dataset_from_csv

# Load the test dataset
dataset = load_dataset_from_csv('data/eval_plastic_rose_carbon.csv')

print(f"Dataset: {dataset.name}")
print(f"Number of items: {len(dataset.items)}\n")

# Show first 3 items to verify answer_gt is loaded
for i, item in enumerate(dataset.items[:3], 1):
    print(f"Item {i}:")
    print(f"  QID: {item.qid}")
    print(f"  Question: {item.question}")
    print(f"  Answer GT: {item.answer_gt[:80] if item.answer_gt else 'None'}...")
    print()

# Verify all items have answer_gt
items_with_gt = sum(1 for item in dataset.items if item.answer_gt)
print(f"✅ Items with ground truth: {items_with_gt}/{len(dataset.items)}")
