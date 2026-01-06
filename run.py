#!/usr/bin/env python3
# run.py
"""
Quick launcher for the Doc RAG Evidence System UI.

Usage:
    python run.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.ui.main import main

if __name__ == "__main__":
    print("🚀 Launching Doc RAG Evidence System V0...")
    print("=" * 60)
    main()
