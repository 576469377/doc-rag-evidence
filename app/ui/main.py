# app/ui/main.py
"""
Gradio UI V0 for doc-rag-evidence system.

Three-tab interface:
  1. Document Management (upload, list, delete)
  2. Query & Answer (ask questions, view evidence)
  3. Evaluation (batch eval, download results)
"""
from __future__ import annotations

import uuid
import yaml
from pathlib import Path
from typing import Optional, List, Tuple

try:
    import gradio as gr
except ImportError:
    gr = None

from core.schemas import AppConfig, QueryInput
from core.pipeline import Pipeline
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
from impl.ingest_pdf import PDFIngestorV0
from impl.index_bm25 import BM25IndexerRetriever
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator
from impl.eval_runner import EvalRunner, load_dataset_from_csv, load_dataset_from_json


class DocRAGUI:
    """Gradio UI wrapper for doc-rag-evidence system."""

    def __init__(self, config_path: str = "configs/app.yaml"):
        # Load config
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.config = AppConfig(**config_dict)

        # Initialize components
        self.store = DocumentStoreLocal(self.config)
        self.logger = RunLoggerLocal(self.config)
        self.ingestor = PDFIngestorV0(self.store)
        self.indexer = BM25IndexerRetriever(self.store)
        self.selector = TopKEvidenceSelector(snippet_length=500)
        self.generator = TemplateGenerator(mode="summary")

        # Create pipeline
        self.pipeline = Pipeline(
            retriever=self.indexer,
            selector=self.selector,
            generator=self.generator,
            logger=self.logger,
            reranker=None
        )

        # Eval runner
        self.eval_runner = EvalRunner(self.pipeline)

        # Load index if exists
        self.indexer.load(self.config, index_name="bm25_default")

        print(f"UI initialized with config: {config_path}")
        print(f"Index loaded: {len(self.indexer.units)} units")

    def launch(self, share: bool = False):
        """Launch Gradio UI."""
        if gr is None:
            raise ImportError("gradio is required. Install with: pip install gradio")

        with gr.Blocks(title="Doc RAG Evidence System V0") as demo:
            gr.Markdown("# 📚 Document RAG Evidence System V0")
            gr.Markdown("Multi-modal document retrieval with evidence traceability")

            with gr.Tabs():
                # Tab 1: Document Management
                with gr.Tab("📄 Document Management"):
                    self._build_document_tab()

                # Tab 2: Query & Answer
                with gr.Tab("🔍 Query & Answer"):
                    self._build_query_tab()

                # Tab 3: Evaluation
                with gr.Tab("📊 Evaluation"):
                    self._build_eval_tab()

        demo.launch(share=share)

    def _build_document_tab(self):
        """Build document management tab."""
        gr.Markdown("## Upload and Manage Documents")

        with gr.Row():
            with gr.Column():
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("📤 Ingest & Index", variant="primary")
                upload_status = gr.Textbox(label="Status", lines=3, interactive=False)

            with gr.Column():
                refresh_btn = gr.Button("🔄 Refresh Document List")
                doc_list = gr.Dataframe(
                    headers=["Doc ID", "Title", "Pages", "Created At"],
                    label="Documents",
                    interactive=False
                )
                delete_docid = gr.Textbox(label="Document ID to Delete", placeholder="Enter doc_id")
                delete_btn = gr.Button("🗑️ Delete Document", variant="stop")
                delete_status = gr.Textbox(label="Delete Status", lines=1, interactive=False)

        # Event handlers
        upload_btn.click(
            fn=self._handle_upload,
            inputs=[pdf_file],
            outputs=[upload_status, doc_list]
        )

        refresh_btn.click(
            fn=self._handle_refresh_docs,
            inputs=[],
            outputs=[doc_list]
        )

        delete_btn.click(
            fn=self._handle_delete_doc,
            inputs=[delete_docid],
            outputs=[delete_status, doc_list]
        )

    def _build_query_tab(self):
        """Build query & answer tab."""
        gr.Markdown("## Ask Questions")

        with gr.Row():
            with gr.Column(scale=1):
                question = gr.Textbox(
                    label="Your Question",
                    placeholder="What is the main topic of the document?",
                    lines=3
                )
                doc_filter = gr.Textbox(
                    label="Filter by Doc IDs (comma-separated, optional)",
                    placeholder="doc1,doc2",
                    lines=1
                )
                query_btn = gr.Button("🚀 Ask Question", variant="primary")

            with gr.Column(scale=2):
                answer_box = gr.Textbox(
                    label="Answer",
                    lines=8,
                    interactive=False
                )

        gr.Markdown("### 📑 Evidence")
        evidence_display = gr.Dataframe(
            headers=["Rank", "Doc ID", "Page", "Score", "Snippet"],
            label="Retrieved Evidence",
            interactive=False
        )

        query_id_box = gr.Textbox(label="Query ID (for traceability)", interactive=False)

        # Event handler
        query_btn.click(
            fn=self._handle_query,
            inputs=[question, doc_filter],
            outputs=[answer_box, evidence_display, query_id_box]
        )

    def _build_eval_tab(self):
        """Build evaluation tab."""
        gr.Markdown("## Batch Evaluation")

        with gr.Row():
            with gr.Column():
                eval_file = gr.File(label="Upload Eval Dataset (CSV or JSON)", file_types=[".csv", ".json"])
                eval_btn = gr.Button("▶️ Run Evaluation", variant="primary")
                eval_status = gr.Textbox(label="Evaluation Status", lines=5, interactive=False)

            with gr.Column():
                eval_metrics = gr.JSON(label="Metrics")
                download_csv = gr.File(label="Download predictions.csv")
                download_json = gr.File(label="Download report.json")

        # Event handler
        eval_btn.click(
            fn=self._handle_eval,
            inputs=[eval_file],
            outputs=[eval_status, eval_metrics, download_csv, download_json]
        )

    # ========== Event Handlers ==========

    def _handle_upload(self, pdf_file) -> Tuple[str, List]:
        """Handle PDF upload and indexing."""
        try:
            if pdf_file is None:
                return "❌ No file uploaded", self._get_doc_list()

            pdf_path = pdf_file.name
            status_lines = []

            # Ingest
            status_lines.append("📄 Ingesting PDF...")
            meta = self.ingestor.ingest(pdf_path, self.config)
            status_lines.append(f"✅ Document ingested: {meta.doc_id}")
            status_lines.append(f"   Pages: {meta.page_count}")

            # Build page artifacts
            status_lines.append("📝 Extracting text and building blocks...")
            artifacts = self.ingestor.build_page_artifacts(meta.doc_id, self.config)
            status_lines.append(f"✅ Processed {len(artifacts)} pages")

            # Build index units
            status_lines.append("🔧 Building index units...")
            units = self.indexer.build_units(meta.doc_id, self.config)
            status_lines.append(f"✅ Created {len(units)} index units")

            # Rebuild index
            status_lines.append("🔍 Rebuilding BM25 index...")
            all_docs = self.store.list_documents()
            all_units = []
            for doc in all_docs:
                doc_units = self.indexer.build_units(doc.doc_id, self.config)
                all_units.extend(doc_units)

            stats = self.indexer.build_index(all_units, self.config)
            self.indexer.persist(self.config, index_name="bm25_default")
            status_lines.append(f"✅ Index rebuilt: {stats.unit_count} units, {stats.elapsed_ms}ms")

            status_lines.append("\n🎉 Upload complete!")

            return "\n".join(status_lines), self._get_doc_list()

        except Exception as e:
            return f"❌ Error: {e}", self._get_doc_list()

    def _handle_refresh_docs(self) -> List:
        """Refresh document list."""
        return self._get_doc_list()

    def _handle_delete_doc(self, doc_id: str) -> Tuple[str, List]:
        """Handle document deletion."""
        try:
            if not doc_id or not doc_id.strip():
                return "❌ Please enter a document ID", self._get_doc_list()

            doc_id = doc_id.strip()
            self.store.delete_document(doc_id)

            # Rebuild index
            all_docs = self.store.list_documents()
            all_units = []
            for doc in all_docs:
                doc_units = self.indexer.build_units(doc.doc_id, self.config)
                all_units.extend(doc_units)

            if all_units:
                self.indexer.build_index(all_units, self.config)
                self.indexer.persist(self.config, index_name="bm25_default")
            else:
                # Clear index
                self.indexer.units = []
                self.indexer.index = None

            return f"✅ Document deleted: {doc_id}", self._get_doc_list()

        except Exception as e:
            return f"❌ Error: {e}", self._get_doc_list()

    def _handle_query(self, question: str, doc_filter: str) -> Tuple[str, List, str]:
        """Handle question answering."""
        try:
            if not question or not question.strip():
                return "❌ Please enter a question", [], ""

            # Parse doc filter
            doc_filter_list = None
            if doc_filter and doc_filter.strip():
                doc_filter_list = [d.strip() for d in doc_filter.split(",")]

            # Create query
            query_id = f"ui_{uuid.uuid4().hex[:12]}"
            query = QueryInput(
                query_id=query_id,
                question=question.strip(),
                doc_filter=doc_filter_list
            )

            # Run pipeline
            record = self.pipeline.answer(query, self.config)

            if not record.status.ok:
                return f"❌ Error: {record.status.error_message}", [], query_id

            # Extract answer
            answer = ""
            if record.generation and record.generation.output:
                answer = record.generation.output.answer

            # Extract evidence
            evidence_rows = []
            if record.evidence:
                for ev in record.evidence.evidence:
                    evidence_rows.append([
                        ev.rank + 1,
                        ev.doc_id[:20],  # Truncate for display
                        ev.page_id,
                        f"{ev.score:.4f}",
                        ev.snippet[:100] + "..." if len(ev.snippet) > 100 else ev.snippet
                    ])

            return answer, evidence_rows, query_id

        except Exception as e:
            return f"❌ Error: {e}", [], ""

    def _handle_eval(self, eval_file) -> Tuple[str, dict, Optional[str], Optional[str]]:
        """Handle batch evaluation."""
        try:
            if eval_file is None:
                return "❌ No file uploaded", {}, None, None

            file_path = eval_file.name
            file_ext = Path(file_path).suffix.lower()

            status_lines = ["📊 Loading dataset..."]

            # Load dataset
            if file_ext == ".csv":
                dataset = load_dataset_from_csv(file_path)
            elif file_ext == ".json":
                dataset = load_dataset_from_json(file_path)
            else:
                return "❌ Unsupported file format (use CSV or JSON)", {}, None, None

            status_lines.append(f"✅ Loaded {len(dataset.items)} questions")
            status_lines.append(f"▶️ Running evaluation...")

            # Run evaluation
            report = self.eval_runner.evaluate(dataset, self.config)

            status_lines.append(f"✅ Evaluation complete!")
            status_lines.append(f"   Success rate: {report.metrics.extra['success_rate']:.2%}")
            status_lines.append(f"   Avg latency: {report.metrics.avg_latency_ms:.0f}ms")

            # Prepare metrics
            metrics_dict = {
                "dataset": report.dataset_name,
                "total_questions": report.metrics.n,
                "success_rate": report.metrics.extra.get("success_rate", 0.0),
                "avg_latency_ms": report.metrics.avg_latency_ms
            }

            # Get artifact paths
            csv_path = report.artifact_paths.get("predictions_csv")
            json_path = report.artifact_paths.get("report")

            return "\n".join(status_lines), metrics_dict, csv_path, json_path

        except Exception as e:
            return f"❌ Error: {e}", {}, None, None

    def _get_doc_list(self) -> List:
        """Get list of documents as table rows."""
        docs = self.store.list_documents()
        rows = []
        for doc in docs:
            rows.append([
                doc.doc_id,
                doc.title,
                doc.page_count,
                doc.created_at[:19]  # Truncate timestamp
            ])
        return rows


def main():
    """Main entry point for UI."""
    ui = DocRAGUI(config_path="configs/app.yaml")
    ui.launch(share=False)


if __name__ == "__main__":
    main()
