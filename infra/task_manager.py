"""
Task manager for background processing (e.g., document ingestion).
Tasks are persistent and can survive UI restarts.
"""
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    task_id: str
    task_type: str  # "ingest", "ocr", "index", etc.
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    total_items: int
    processed_items: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        data['status'] = TaskStatus(data['status'])
        return cls(**data)


class TaskManager:
    """Manages background tasks with persistent state."""
    
    def __init__(self, tasks_dir: str = "data/tasks"):
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        
        self.tasks: Dict[str, Task] = {}
        self.worker_thread: Optional[threading.Thread] = None
        self.task_queue: List[tuple] = []  # (task_id, func, args, kwargs)
        self.lock = threading.Lock()
        self.running = False
        
        # Load existing tasks from disk
        self._load_tasks()
        
        # Start worker thread
        self._start_worker()
    
    def _load_tasks(self):
        """Load all tasks from disk."""
        for task_file in self.tasks_dir.glob("*.json"):
            try:
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    task = Task.from_dict(task_data)
                    self.tasks[task.task_id] = task
            except Exception as e:
                print(f"Error loading task {task_file}: {e}")
    
    def _save_task(self, task: Task):
        """Save task state to disk."""
        task_file = self.tasks_dir / f"{task.task_id}.json"
        with open(task_file, 'w') as f:
            json.dump(task.to_dict(), f, indent=2)
    
    def _start_worker(self):
        """Start background worker thread."""
        if not self.running:
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            print("✅ Task worker thread started")
    
    def _worker_loop(self):
        """Worker loop that processes tasks from queue."""
        while self.running:
            task_item = None
            
            with self.lock:
                if self.task_queue:
                    task_item = self.task_queue.pop(0)
            
            if task_item:
                task_id, func, args, kwargs = task_item
                self._execute_task(task_id, func, args, kwargs)
            else:
                time.sleep(0.5)  # Wait a bit before checking again
    
    def _execute_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict):
        """Execute a task function."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        # Update task status to running
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now().isoformat()
        self._save_task(task)
        
        try:
            # Execute the task function
            # The function should update task progress via update_task_progress()
            result = func(task_id, self, *args, **kwargs)
            
            # Mark as completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now().isoformat()
            task.progress = 1.0
            task.result = result
            
        except Exception as e:
            # Mark as failed
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now().isoformat()
            task.error_message = str(e)
            import traceback
            print(f"Task {task_id} failed: {e}")
            traceback.print_exc()
        
        finally:
            self._save_task(task)
    
    def submit_task(
        self,
        task_id: str,
        task_type: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        total_items: int = 1,
        description: str = "Starting..."
    ) -> Task:
        """Submit a new task to the queue."""
        if kwargs is None:
            kwargs = {}
        
        # Create task
        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step=description,
            total_items=total_items,
            processed_items=0,
            created_at=datetime.now().isoformat()
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.task_queue.append((task_id, func, args, kwargs))
        
        self._save_task(task)
        return task
    
    def update_task_progress(
        self,
        task_id: str,
        processed_items: int = None,
        current_step: str = None,
        progress: float = None
    ):
        """Update task progress (called from task function)."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        with self.lock:
            if processed_items is not None:
                task.processed_items = processed_items
                task.progress = min(1.0, processed_items / max(1, task.total_items))
            
            if current_step is not None:
                task.current_step = current_step
            
            if progress is not None:
                task.progress = min(1.0, max(0.0, progress))
        
        self._save_task(task)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID."""
        return self.tasks.get(task_id)
    
    def list_tasks(self, task_type: Optional[str] = None, limit: int = 50) -> List[Task]:
        """List all tasks, optionally filtered by type."""
        tasks = list(self.tasks.values())
        
        if task_type:
            tasks = [t for t in tasks if t.task_type == task_type]
        
        # Sort by created_at descending
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        
        return tasks[:limit]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        task = self.tasks.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        
        with self.lock:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now().isoformat()
            # Remove from queue
            self.task_queue = [(tid, f, a, k) for tid, f, a, k in self.task_queue if tid != task_id]
        
        self._save_task(task)
        return True
    
    def clear_completed_tasks(self, keep_recent: int = 10):
        """Clear old completed/failed tasks, keeping only recent ones."""
        completed_tasks = [
            t for t in self.tasks.values()
            if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
        ]
        
        # Sort by completed_at
        completed_tasks.sort(
            key=lambda t: t.completed_at or t.created_at,
            reverse=True
        )
        
        # Remove old tasks
        for task in completed_tasks[keep_recent:]:
            with self.lock:
                del self.tasks[task.task_id]
            
            # Delete file
            task_file = self.tasks_dir / f"{task.task_id}.json"
            if task_file.exists():
                task_file.unlink()
        
        return len(completed_tasks) - keep_recent
    
    def shutdown(self):
        """Shutdown the task manager."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
