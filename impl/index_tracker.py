"""
Incremental index tracker - tracks which documents are indexed.
Enables incremental index updates instead of full rebuilds.
"""
from pathlib import Path
from typing import Dict, Set, Optional
import json
from datetime import datetime


class IndexTracker:
    """
    Tracks which documents are indexed and their modification times.
    Enables incremental index updates.
    """
    
    def __init__(self, index_dir: Path):
        """
        Initialize tracker.
        
        Args:
            index_dir: Directory where index is stored
        """
        self.index_dir = Path(index_dir)
        self.tracker_file = self.index_dir / "indexed_docs.json"
        self.indexed_docs: Dict[str, dict] = {}  # doc_id -> {timestamp, page_count, unit_count}
        self.load()
    
    def load(self) -> bool:
        """
        Load tracker from disk.
        
        Returns:
            True if loaded successfully, False if not found
        """
        if not self.tracker_file.exists():
            return False
        
        try:
            with open(self.tracker_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.indexed_docs = data.get("indexed_docs", {})
            return True
        except Exception as e:
            print(f"⚠️  Failed to load index tracker: {e}")
            return False
    
    def save(self):
        """Save tracker to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "indexed_docs": self.indexed_docs,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(self.tracker_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def is_indexed(self, doc_id: str) -> bool:
        """
        Check if a document is already indexed.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document is in index
        """
        return doc_id in self.indexed_docs
    
    def mark_indexed(self, doc_id: str, page_count: int, unit_count: int):
        """
        Mark a document as indexed.
        
        Args:
            doc_id: Document ID
            page_count: Number of pages
            unit_count: Number of index units created
        """
        self.indexed_docs[doc_id] = {
            "timestamp": datetime.now().isoformat(),
            "page_count": page_count,
            "unit_count": unit_count
        }
    
    def remove(self, doc_id: str):
        """
        Remove a document from tracker.
        
        Args:
            doc_id: Document ID to remove
        """
        if doc_id in self.indexed_docs:
            del self.indexed_docs[doc_id]
    
    def get_indexed_docs(self) -> Set[str]:
        """
        Get set of all indexed document IDs.
        
        Returns:
            Set of document IDs
        """
        return set(self.indexed_docs.keys())
    
    def get_stats(self) -> dict:
        """
        Get tracker statistics.
        
        Returns:
            Dict with stats
        """
        total_units = sum(info["unit_count"] for info in self.indexed_docs.values())
        total_pages = sum(info["page_count"] for info in self.indexed_docs.values())
        
        return {
            "doc_count": len(self.indexed_docs),
            "total_pages": total_pages,
            "total_units": total_units,
            "docs": self.indexed_docs
        }
    
    def clear(self):
        """Clear all tracking information."""
        self.indexed_docs = {}
