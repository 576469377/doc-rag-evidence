# Quick Reference Card - V0.1

## One-Line Commands

### Full Pipeline (Ingest → Index → UI)
```bash
python run_v1.py --ingest-dir data/pdfs --use-ocr --build-all --ui
```

### Ingest Only
```bash
# Single PDF with OCR
python scripts/ingest_docs_v1.py --pdf myfile.pdf --use-ocr

# Directory without OCR (faster)
python scripts/ingest_docs_v1.py --pdf-dir data/pdfs
```

### Index Only
```bash
# All enabled indices
python scripts/build_indices_v1.py --all

# Selective
python scripts/build_indices_v1.py --bm25
python scripts/build_indices_v1.py --dense
python scripts/build_indices_v1.py --colpali
```

### UI Only
```bash
python app/ui/main_v1.py
```

---

## Configuration Quick Edit

### Enable All Features
```yaml
# configs/app.yaml

ocr:
  provider: "sglang"
  model: "deepseek_ocr"
  endpoint: "http://127.0.0.1:30000"

dense:
  enabled: true
  model: "Qwen/Qwen3-Embedding-0.6B"
  endpoint: "http://127.0.0.1:30000"

colpali:
  enabled: true
  model: "vidore/colqwen2-v0.1"
  device: "cuda:0"

retrieval_mode: "bm25"  # or dense, colpali
```

---

## SGLang Server Setup

### Embedding Server
```bash
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model Qwen/Qwen3-Embedding-0.6B \
  --port 30000
```

### OCR Server (Optional)
```bash
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --model deepseek-ai/deepseek-ocr \
  --port 30001
```

### Test Connection
```bash
curl http://127.0.0.1:30000/v1/models
```

---

## Common Issues

### OCR Timeout
**Fix**: Increase timeout in config
```yaml
ocr:
  timeout: 120  # seconds
```

### ColPali GPU OOM
**Fix**: Use correct GPU
```bash
# Check available GPUs
nvidia-smi

# Set in config
colpali:
  device: "cuda:0"  # Use available GPU
```

### Dense API Error
**Fix**: Verify SGLang server running
```bash
curl http://127.0.0.1:30000/v1/models
# Should return model info
```

### Index Not Found
**Fix**: Build indices first
```bash
python scripts/build_indices_v1.py --all
```

---

## File Locations

### Documents
```
data/docs/{doc_id}/
├── meta.json
└── pages/{page_id:04d}/
    ├── page.png
    ├── ocr.json
    ├── blocks.json
    └── text.json
```

### Indices
```
data/indices/
├── bm25/
├── dense/
└── colpali/
```

### Logs
```
data/runs/{query_id}.json
```

---

## API Examples

### Python Code

```python
# OCR
from impl.ocr_client import SGLangOcrClient
client = SGLangOcrClient(endpoint="http://127.0.0.1:30000")
result = client.ocr_page("page.png")

# Dense Retrieval
from impl.index_dense import DenseIndexerRetriever, SGLangEmbedder
embedder = SGLangEmbedder(endpoint="http://127.0.0.1:30000")
retriever = DenseIndexerRetriever(embedder=embedder)
hits = retriever.retrieve("question", top_k=10)

# ColPali
from impl.index_colpali import ColPaliRetriever
retriever = ColPaliRetriever(model_name="vidore/colqwen2-v0.1")
hits = retriever.retrieve("show diagrams", top_k=5)
```

---

## Performance Targets

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Page rendering | 0.1s | PyMuPDF |
| OCR (first run) | 2-5s | Per page |
| OCR (cached) | 0.01s | Skip processing |
| Dense indexing | 30s | 1000 pages |
| ColPali indexing | 300s | 1000 pages |
| BM25 query | 50ms | Top-10 |
| Dense query | 100ms | Top-10 |
| ColPali query | 500ms | Top-10 |

---

## GPU Memory Usage

| Component | GPU | Memory | Notes |
|-----------|-----|--------|-------|
| ColPali | 1x | 12GB | Transformers model |
| SGLang Embedding | 1x | 6GB | API server |
| SGLang OCR | 1x | 8GB | API server |
| Reserved | 1x | - | For reranker/LLM |

---

## Troubleshooting Checklist

- [ ] Python 3.9+ installed
- [ ] All requirements installed (`pip install -r requirements.txt`)
- [ ] SGLang server running (check with curl)
- [ ] GPU available (if using ColPali)
- [ ] Config file has correct endpoints
- [ ] Indices built (`scripts/build_indices_v1.py`)
- [ ] Documents ingested (check `data/docs/`)

---

## Getting Help

1. **Check docs**: `docs/v0.1_multimodal_retrieval.md`
2. **Check implementation**: `docs/v0.1_implementation_summary.md`
3. **Run smoke test**: See IMPLEMENTATION_COMPLETE.md
4. **Check logs**: `data/runs/{query_id}.json`

---

## Quick Links

- [Full User Guide](docs/v0.1_multimodal_retrieval.md)
- [Technical Summary](docs/v0.1_implementation_summary.md)
- [V0 Manual](docs/user_manual_v0.md)
- [Implementation Complete](IMPLEMENTATION_COMPLETE.md)
