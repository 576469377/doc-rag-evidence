# Changelog

All notable changes to Doc RAG Evidence will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2026-01-23

### Added
- ColPali multiprocessing support with 2 parallel workers
- Command-line index building tool (`scripts/build_indices_v1.py`)
- Incremental index saving with checkpoints every 10 documents
- Real-time progress bars for each worker with timing information
- Image resize parameter (`max_image_size`) for ColPali and Dense-VL
- Worker PID tracking for better task management

### Changed
- Default `max_image_size` from 1024px to 768px for faster processing
- Updated `build_indices_v1.py` to use `IncrementalIndexManager`
- Improved startup progress bar with 7 steps initialization

### Fixed
- Dense-VL index duplication bug (indentation error in `build_dense_vl_index.py`)
- ColPali `max_image_size` parameter not being passed through `index_incremental.py`
- ColPali worker variable name typo (`_worker_colpali_max_image_SIZE`)
- Removed inaccurate time estimation from UI index build messages

### Performance
- ColPali indexing: 40s/page → 25s/page with 768px + 2 workers
- Dense-VL indexing: ~0.3s/page with 4 workers + Flash Attention 2

## [1.2.0] - 2026-01-21

### Added
- Dense-VL multimodal retrieval using Qwen3-VL-Embedding-2B
- Flash Attention 2 support for automatic acceleration
- Image resize optimization for ColPali and Dense-VL
- Multiprocessing support for Dense-VL (4 workers)
- Hit normalization: page-level → block-level expansion
- Hybrid retrieval support for Dense-VL combinations

### Performance
- Dense-VL indexing: 140s → 15s for 56 pages (9.3x speedup)
  - Flash Attention 2: ~2x acceleration
  - Image resize to 1024px: ~2x acceleration
  - 4 parallel workers: ~4x acceleration

### Changed
- ColPali and Dense-VL can now share GPU2 (lazy loading)
- Updated GPU memory usage documentation

## [1.1.0] - 2026-01-18

### Added
- Qwen3-VL-4B-Instruct multimodal generation engine
- Support for text (OCR) and image-based evidence formats
- Citation control policies: strict/relaxed/none
- vLLM backend for high-performance inference

### Changed
- Generation engine can now process page images for chart/formula understanding
- Updated LLM configuration in `configs/app.yaml`

## [1.0.0] - 2026-01-14

### Added
- Initial release of Doc RAG Evidence system
- PDF ingestion with HunyuanOCR support
- Block-level text chunking
- Three retrieval modes: BM25, Dense, ColPali
- Hybrid retrieval with custom combinations
- Weighted sum and RRF fusion strategies
- Gradio Web UI for document management and Q&A
- Batch evaluation framework (CSV/JSON support)
- Incremental index updates

---

## Version Naming Convention

- **Major version** (X.0.0): Architecture changes, major features
- **Minor version** (1.X.0): New features, significant improvements
- **Patch version** (1.2.X): Bug fixes, minor improvements, optimizations
