# infra/runlog_local.py
"""
Local file-based RunLogger implementation.
Stores query run records as JSON files for full traceability.

Path structure:
  data/runs/{query_id}.json
  
Each run record contains:
  - Input query
  - Retrieval hits
  - Evidence selection
  - Generation output
  - Status & timing
  - Config snapshot
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from core.schemas import QueryRunRecord, AppConfig


class RunLoggerLocal:
    """Local file-based run logger for query traceability."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.runs_dir = Path(config.runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, record: QueryRunRecord) -> str:
        """
        Save a query run record to disk.
        
        Returns:
            The run_id (same as query_id) for reference.
        """
        query_id = record.query.query_id
        run_path = self.runs_dir / f"{query_id}.json"

        with open(run_path, "w", encoding="utf-8") as f:
            json.dump(record.model_dump(), f, indent=2, ensure_ascii=False)

        return query_id

    def load_run(self, run_id: str) -> Optional[QueryRunRecord]:
        """Load a run record from disk."""
        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            return None

        with open(run_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return QueryRunRecord(**data)

    def list_runs(self, limit: int = 100) -> list[str]:
        """List recent run IDs (sorted by modification time, newest first)."""
        if not self.runs_dir.exists():
            return []

        run_files = sorted(
            self.runs_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return [f.stem for f in run_files[:limit]]

    def get_failed_runs(self, limit: int = 50) -> list[QueryRunRecord]:
        """Get recent failed runs for debugging."""
        failed = []
        for run_id in self.list_runs(limit=limit):
            record = self.load_run(run_id)
            if record and not record.status.ok:
                failed.append(record)
        return failed
