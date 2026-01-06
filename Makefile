.PHONY: help install run test build-index demo-query demo-eval clean

help:
	@echo "Doc RAG Evidence System - V0"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make run          - Launch Gradio UI"
	@echo "  make test         - Run basic smoke tests"
	@echo "  make build-index  - Build/rebuild BM25 index"
	@echo "  make demo-query   - Run demo query (requires QUESTION variable)"
	@echo "  make demo-eval    - Run demo evaluation (requires DATASET variable)"
	@echo "  make clean        - Clean generated artifacts"

install:
	pip install -r requirements.txt

run:
	python run.py

test:
	python tests/test_basic.py

build-index:
	python scripts/build_index.py

demo-query:
	@if [ -z "$(QUESTION)" ]; then \
		echo "Usage: make demo-query QUESTION='Your question here'"; \
		exit 1; \
	fi
	python scripts/demo_run.py "$(QUESTION)"

demo-eval:
	@if [ -z "$(DATASET)" ]; then \
		echo "Usage: make demo-eval DATASET=path/to/dataset.csv"; \
		exit 1; \
	fi
	python scripts/demo_eval.py "$(DATASET)"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "✅ Cleaned cache files"

clean-data:
	@echo "⚠️  This will delete all data (docs, indices, runs, reports)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/docs/* data/indices/* data/runs/* data/reports/*; \
		echo "✅ Data cleaned"; \
	else \
		echo "Cancelled"; \
	fi
