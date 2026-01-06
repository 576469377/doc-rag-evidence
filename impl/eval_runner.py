# impl/eval_runner.py
"""
Evaluation Runner V0 implementation.

Runs batch evaluation on a dataset of questions.
Outputs:
  - predictions.csv (per-question results)
  - report.json (aggregate metrics)
  
V0 metrics:
  - Success rate (status_ok)
  - Average latency
  - Optional: exact match / contains match (if ground truth provided)
"""
from __future__ import annotations

import csv
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from core.schemas import (
    AppConfig, EvalDataset, EvalMetrics, EvalRow, EvalReport,
    QueryInput, QueryRunRecord
)
from core.pipeline import Pipeline


class EvalRunner:
    """Batch evaluation runner."""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def evaluate(self, dataset: EvalDataset, config: AppConfig) -> EvalReport:
        """
        Run evaluation on a dataset.
        
        Args:
            dataset: Evaluation dataset with questions
            config: App configuration
            
        Returns:
            EvalReport with metrics and per-question results
        """
        print(f"Starting evaluation on dataset: {dataset.name}")
        print(f"Number of questions: {len(dataset.items)}")

        rows: List[EvalRow] = []
        latencies: List[int] = []
        success_count = 0

        for item in dataset.items:
            print(f"Processing question {item.qid}: {item.question[:50]}...")

            # Create query
            query = QueryInput(
                query_id=f"eval_{item.qid}_{uuid.uuid4().hex[:8]}",
                question=item.question,
                doc_filter=None,
                user_meta={"eval_qid": item.qid}
            )

            # Run pipeline
            start_time = time.time()
            try:
                record: QueryRunRecord = self.pipeline.answer(query, config)
                elapsed_ms = int((time.time() - start_time) * 1000)

                # Extract results
                status_ok = record.status.ok
                answer_pred = ""
                cited_units = []

                if record.generation and record.generation.output:
                    answer_pred = record.generation.output.answer
                    cited_units = record.generation.output.cited_units

                row = EvalRow(
                    qid=item.qid,
                    question=item.question,
                    answer_pred=answer_pred,
                    cited_units=cited_units,
                    latency_ms=elapsed_ms,
                    status_ok=status_ok,
                    error_type=record.status.error_type if not status_ok else None
                )

                if status_ok:
                    success_count += 1
                    latencies.append(elapsed_ms)

            except Exception as e:
                elapsed_ms = int((time.time() - start_time) * 1000)
                row = EvalRow(
                    qid=item.qid,
                    question=item.question,
                    answer_pred="",
                    cited_units=[],
                    latency_ms=elapsed_ms,
                    status_ok=False,
                    error_type=type(e).__name__
                )
                print(f"  ERROR: {e}")

            rows.append(row)

        # Compute metrics
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        metrics = EvalMetrics(
            n=len(dataset.items),
            exact_match=None,  # V0: not implemented
            contains_match=None,  # V0: not implemented
            avg_latency_ms=avg_latency,
            evidence_hit_rate=None,  # V0: not implemented
            extra={
                "success_count": success_count,
                "success_rate": success_count / len(dataset.items) if dataset.items else 0.0
            }
        )

        # Save artifacts
        artifact_paths = self._save_artifacts(dataset.name, rows, metrics, config)

        report = EvalReport(
            dataset_name=dataset.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            rows=rows,
            artifact_paths=artifact_paths
        )

        print(f"\nEvaluation complete!")
        print(f"  Success rate: {metrics.extra['success_rate']:.2%}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"  Artifacts saved to: {artifact_paths.get('report')}")

        return report

    def _save_artifacts(
        self, dataset_name: str, rows: List[EvalRow], metrics: EvalMetrics, config: AppConfig
    ) -> dict:
        """Save evaluation artifacts to disk."""
        # Create report directory
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_dir = Path(config.reports_dir) / dataset_name / timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

        artifact_paths = {}

        # Save predictions.csv
        csv_path = report_dir / "predictions.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "qid", "question", "answer_pred", "cited_units", 
                "latency_ms", "status_ok", "error_type"
            ])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "qid": row.qid,
                    "question": row.question,
                    "answer_pred": row.answer_pred,
                    "cited_units": ",".join(row.cited_units),
                    "latency_ms": row.latency_ms,
                    "status_ok": row.status_ok,
                    "error_type": row.error_type or ""
                })
        artifact_paths["predictions_csv"] = str(csv_path)

        # Save report.json
        report_path = report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_name": dataset_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics.model_dump(),
                "summary": {
                    "total_questions": metrics.n,
                    "success_rate": metrics.extra.get("success_rate", 0.0),
                    "avg_latency_ms": metrics.avg_latency_ms
                }
            }, f, indent=2)
        artifact_paths["report"] = str(report_path)

        return artifact_paths


def load_dataset_from_csv(csv_path: str, dataset_name: Optional[str] = None) -> EvalDataset:
    """
    Load evaluation dataset from CSV file.
    
    Expected columns:
      - qid: Question ID
      - question: Question text
      - answer_gt (optional): Ground truth answer
      
    Args:
        csv_path: Path to CSV file
        dataset_name: Optional dataset name (defaults to filename)
        
    Returns:
        EvalDataset object
    """
    from core.schemas import EvalItem

    csv_path = Path(csv_path)
    if dataset_name is None:
        dataset_name = csv_path.stem

    items = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = EvalItem(
                qid=row["qid"],
                question=row["question"],
                answer_gt=row.get("answer_gt"),
                citations_gt=None
            )
            items.append(item)

    return EvalDataset(name=dataset_name, items=items)


def load_dataset_from_json(json_path: str, dataset_name: Optional[str] = None) -> EvalDataset:
    """
    Load evaluation dataset from JSON file.
    
    Expected format:
    {
      "name": "dataset_name",
      "items": [
        {"qid": "q1", "question": "...", "answer_gt": "..."},
        ...
      ]
    }
    
    Args:
        json_path: Path to JSON file
        dataset_name: Optional dataset name (overrides file content)
        
    Returns:
        EvalDataset object
    """
    json_path = Path(json_path)
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if dataset_name is None:
        dataset_name = data.get("name", json_path.stem)

    return EvalDataset(**data)
