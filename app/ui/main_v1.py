"""
Enhanced Gradio UI V1 for doc-rag-evidence system.
Supports multiple retrieval modes: BM25, Dense, ColPali, Hybrid.
"""
from __future__ import annotations

import uuid
import yaml
from pathlib import Path
from typing import Optional, List, Tuple
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import gradio as gr
except ImportError:
    gr = None

from core.schemas import AppConfig, QueryInput
from core.pipeline import Pipeline
from infra.store_local import DocumentStoreLocal
from infra.runlog_local import RunLoggerLocal
from impl.ingest_pdf_v1 import PDFIngestorV1
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, SGLangEmbedder
from impl.index_colpali import ColPaliRetriever
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator
from impl.eval_runner import EvalRunner


class DocRAGUIV1:
    """Enhanced Gradio UI with multi-mode retrieval."""

    def __init__(self, config_path: str = "configs/app.yaml"):
        # Load config
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.config = AppConfig(**config_dict)

        # Initialize infrastructure
        self.store = DocumentStoreLocal(self.config)
        self.logger = RunLoggerLocal(self.config)
        
        # Initialize retrievers
        self.retrievers = {}
        self._init_retrievers()
        
        # Initialize other components
        self.selector = TopKEvidenceSelector(snippet_length=500)
        self.generator = TemplateGenerator(mode="summary")
        
        # Create pipeline (default retriever)
        default_retriever = self.retrievers.get(self.config.retrieval_mode)
        if not default_retriever:
            default_retriever = self.retrievers.get("bm25")
        
        self.pipeline = Pipeline(
            retriever=default_retriever,
            selector=self.selector,
            generator=self.generator,
            logger=self.logger,
            reranker=None
        )
        
        # Eval runner
        self.eval_runner = EvalRunner(self.pipeline)
        
        print(f"UI initialized with config: {config_path}")
        print(f"Available retrieval modes: {list(self.retrievers.keys())}")

    def _init_retrievers(self):
        """Initialize available retrievers based on config."""
        indices_dir = Path(self.config.indices_dir)
        
        # BM25 (always try to load)
        bm25_index_name = "bm25_default"
        try:
            retriever = BM25IndexerRetriever(self.store)
            retriever.load(self.config, index_name=bm25_index_name)
            self.retrievers["bm25"] = retriever
            print(f"Loaded BM25 index: {len(retriever.units)} units")
        except Exception as e:
            print(f"Failed to load BM25 index: {e}")
        
        # Dense
        if self.config.dense.get("enabled"):
            dense_index_dir = indices_dir / "dense"
            if dense_index_dir.exists():
                try:
                    embedder = SGLangEmbedder(
                        endpoint=self.config.dense["endpoint"],
                        model=self.config.dense["model"]
                    )
                    retriever = DenseIndexerRetriever.load(dense_index_dir, embedder)
                    self.retrievers["dense"] = retriever
                    print(f"Loaded Dense index: {len(retriever.units)} units")
                except Exception as e:
                    print(f"Failed to load Dense index: {e}")
                    print(f"Loaded Dense index: {len(retriever.units)} units")
                except Exception as e:
                    print(f"Failed to load Dense index: {e}")
        
        # ColPali
        if self.config.colpali.get("enabled"):
            colpali_index_dir = indices_dir / "colpali"
            if colpali_index_dir.exists():
                try:
                    retriever = ColPaliRetriever.load(
                        colpali_index_dir,
                        model_name=self.config.colpali["model"],
                        device=self.config.colpali.get("device", "cuda:0")
                    )
                    self.retrievers["colpali"] = retriever
                    print(f"Loaded ColPali index")
                except Exception as e:
                    print(f"Failed to load ColPali index: {e}")

    def launch(self, share: bool = False):
        """Launch Gradio UI."""
        if gr is None:
            raise ImportError("gradio is required. Install with: pip install gradio")

        with gr.Blocks(title="Doc RAG Evidence System V1") as demo:
            gr.Markdown("# 📚 Document RAG Evidence System V1")
            gr.Markdown("Multi-modal document retrieval with BM25 / Dense / ColPali support")

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

        demo.launch(share=share, server_name="0.0.0.0", server_port=7860)

    def _build_document_tab(self):
        """Build document management tab."""
        gr.Markdown("## Upload and Manage Documents")

        # Section 1: Upload & Ingest
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Upload PDF")
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                use_ocr = gr.Checkbox(label="Use OCR (slower, better quality)", value=False)
                upload_btn = gr.Button("📤 Ingest Document", variant="primary")
                upload_status = gr.Textbox(label="Ingestion Status", lines=3, interactive=False)

            with gr.Column():
                gr.Markdown("### 📚 Document List")
                refresh_btn = gr.Button("🔄 Refresh Document List")
                doc_list = gr.Dataframe(
                    headers=["Doc ID", "Title", "Pages", "Created At"],
                    label="Documents",
                    interactive=False
                )
                delete_docid = gr.Textbox(label="Document ID to Delete", placeholder="Enter doc_id")
                delete_btn = gr.Button("🗑️ Delete Document", variant="stop")
                delete_status = gr.Textbox(label="Delete Status", lines=1, interactive=False)

        gr.Markdown("---")
        
        # Section 2: Build Indices
        gr.Markdown("### 🔧 Build Indices")
        gr.Markdown("After uploading documents, build indices for retrieval")
        
        with gr.Row():
            with gr.Column():
                build_bm25 = gr.Checkbox(label="Build BM25 Index (keyword search)", value=True)
                build_dense = gr.Checkbox(label="Build Dense Index (semantic embedding)", value=False)
                build_colpali = gr.Checkbox(label="Build ColPali Index (vision-based)", value=False)
                index_name_suffix = gr.Textbox(
                    label="Index Name Suffix (optional)",
                    placeholder="default",
                    value="default"
                )
                build_btn = gr.Button("⚙️ Build Indices", variant="primary", size="lg")
                
            with gr.Column():
                build_status = gr.Textbox(
                    label="Build Status",
                    lines=10,
                    interactive=False,
                    placeholder="Status will appear here..."
                )

        # Event handlers
        upload_btn.click(
            fn=self._handle_upload,
            inputs=[pdf_file, use_ocr],
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
        
        build_btn.click(
            fn=self._handle_build_indices,
            inputs=[build_bm25, build_dense, build_colpali, index_name_suffix],
            outputs=[build_status]
        )

    def _build_query_tab(self):
        """Build query & answer tab."""
        gr.Markdown("## Ask Questions")

        with gr.Row():
            with gr.Column(scale=1):
                # Retrieval mode selector
                retrieval_mode = gr.Radio(
                    choices=list(self.retrievers.keys()),
                    value=self.config.retrieval_mode if self.config.retrieval_mode in self.retrievers else list(self.retrievers.keys())[0],
                    label="Retrieval Mode",
                    info="BM25: keyword search | Dense: semantic embedding | ColPali: vision-based"
                )
                
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
            headers=["Rank", "Source", "Doc ID", "Page", "Score", "Snippet"],
            label="Retrieved Evidence",
            interactive=False
        )

        query_id_box = gr.Textbox(label="Query ID (for traceability)", interactive=False)

        # Event handler
        query_btn.click(
            fn=self._handle_query,
            inputs=[question, doc_filter, retrieval_mode],
            outputs=[answer_box, evidence_display, query_id_box]
        )

    def _build_eval_tab(self):
        """Build evaluation tab."""
        gr.Markdown("## Batch Evaluation")

        with gr.Row():
            with gr.Column():
                eval_file = gr.File(label="Upload Eval Dataset (CSV or JSON)", file_types=[".csv", ".json"])
                eval_mode = gr.Radio(
                    choices=list(self.retrievers.keys()),
                    value=list(self.retrievers.keys())[0],
                    label="Retrieval Mode for Evaluation"
                )
                eval_btn = gr.Button("▶️ Run Evaluation", variant="primary")
                eval_status = gr.Textbox(label="Evaluation Status", lines=5, interactive=False)

            with gr.Column():
                eval_metrics = gr.JSON(label="Metrics")
                download_csv = gr.File(label="Download predictions.csv")
                download_json = gr.File(label="Download report.json")

        # Event handler
        eval_btn.click(
            fn=self._handle_eval,
            inputs=[eval_file, eval_mode],
            outputs=[eval_status, eval_metrics, download_csv, download_json]
        )

    # ========== Event Handlers ==========

    def _handle_upload(self, pdf_file, use_ocr: bool) -> Tuple[str, List]:
        """Handle PDF upload and ingestion."""
        try:
            if pdf_file is None:
                return "❌ Error: No file uploaded", self._get_doc_list()

            # Ingest with V1 ingestor
            ingestor = PDFIngestorV1(
                config=self.config,
                store=self.store,
                use_ocr=use_ocr
            )
            
            status_msg = f"⏳ Ingesting: {Path(pdf_file.name).name}\n"
            if use_ocr:
                status_msg += "OCR enabled - this may take 10-30 seconds per page...\n"
            
            meta = ingestor.ingest(pdf_file.name)
            
            status = f"✅ Ingested Successfully!\n"
            status += f"Document ID: {meta.doc_id}\n"
            status += f"Title: {meta.title}\n"
            status += f"Pages: {meta.page_count}\n"
            status += f"OCR: {'✓ enabled' if use_ocr else '✗ disabled'}\n\n"
            status += "⚠️ Next Step: Build indices below to enable retrieval"
            
            return status, self._get_doc_list()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            return error_msg, self._get_doc_list()

    def _handle_build_indices(
        self,
        build_bm25: bool,
        build_dense: bool,
        build_colpali: bool,
        index_name_suffix: str
    ) -> str:
        """Handle index building."""
        try:
            if not any([build_bm25, build_dense, build_colpali]):
                return "❌ Error: Please select at least one index type to build"
            
            status = "🔧 Building Indices...\n\n"
            suffix = index_name_suffix.strip() or "default"
            
            # Get all documents
            docs = self.store.list_documents()
            if not docs:
                return "❌ Error: No documents found. Please ingest documents first."
            
            status += f"📚 Found {len(docs)} document(s)\n\n"
            
            # Collect all index units
            from core.schemas import IndexUnit
            all_units = []
            
            for meta in docs:
                # Get pages
                pages = self.store.list_pages(meta.doc_id)
                for page_meta in pages:
                    # Get blocks
                    blocks = self.store.load_blocks(meta.doc_id, page_meta.page_id)
                    
                    if blocks:
                        # Use blocks if available
                        for block in blocks:
                            unit = IndexUnit(
                                unit_id=f"{meta.doc_id}_p{page_meta.page_id:04d}_{block.block_id}",
                                doc_id=meta.doc_id,
                                page_id=page_meta.page_id,
                                block_id=block.block_id,
                                text=block.text
                            )
                            all_units.append(unit)
                    else:
                        # Fallback: use page text as single unit
                        artifact = self.store.load_page_artifact(meta.doc_id, page_meta.page_id)
                        if artifact and artifact.text and artifact.text.text.strip():
                            unit = IndexUnit(
                                unit_id=f"{meta.doc_id}_p{page_meta.page_id:04d}",
                                doc_id=meta.doc_id,
                                page_id=page_meta.page_id,
                                block_id=None,
                                text=artifact.text.text
                            )
                            all_units.append(unit)
            
            status += f"📦 Collected {len(all_units)} index units\n\n"
            
            # Check if we have any units
            if len(all_units) == 0:
                return (status + 
                    "❌ Error: No index units found (documents have no text)\n\n"
                    "Possible causes:\n"
                    "  • Documents were imported without OCR\n"
                    "  • OCR service was not running during import\n"
                    "  • OCR service returned errors (check logs)\n\n"
                    "Solutions:\n"
                    "  1. Delete existing documents\n"
                    "  2. Make sure OCR service is running: curl http://localhost:8000/health\n"
                    "  3. Re-upload PDFs with 'Use OCR' enabled\n"
                )
            
            # Build BM25
            if build_bm25:
                status += "⏳ Building BM25 index...\n"
                try:
                    retriever = BM25IndexerRetriever(self.store)
                    retriever.build_index(all_units, self.config)
                    index_name = f"bm25_{suffix}"
                    retriever.persist(self.config, index_name=index_name)
                    
                    # Reload
                    self.retrievers["bm25"] = retriever
                    status += f"✅ BM25 index built: {index_name} ({len(all_units)} units)\n\n"
                except Exception as e:
                    status += f"❌ BM25 build failed: {str(e)}\n\n"
            
            # Build Dense
            if build_dense:
                status += "⏳ Building Dense index...\n"
                try:
                    from impl.index_dense import VLLMEmbedder
                    embedder = VLLMEmbedder(
                        endpoint=self.config.dense["endpoint"],
                        model=self.config.dense["model"]
                    )
                    retriever = DenseIndexerRetriever(embedder)
                    retriever.build_index(all_units, self.config)
                    
                    index_dir = Path(self.config.indices_dir) / f"dense_{suffix}"
                    index_dir.mkdir(parents=True, exist_ok=True)
                    retriever.save(index_dir)
                    
                    # Reload
                    self.retrievers["dense"] = retriever
                    status += f"✅ Dense index built: dense_{suffix} ({len(all_units)} units)\n\n"
                except Exception as e:
                    status += f"❌ Dense build failed: {str(e)}\n"
                    status += "   Make sure vllm embedding server is running on port 8001\n\n"
            
            # Build ColPali
            if build_colpali:
                status += "⏳ Building ColPali index...\n"
                try:
                    retriever = ColPaliRetriever(
                        model_name=self.config.colpali["model"],
                        device=self.config.colpali.get("device", "cuda:0")
                    )
                    
                    # Build with page images
                    page_data = []
                    for meta in docs:
                        pages = self.store.list_pages(meta.doc_id)
                        for page_meta in pages:
                            image_path = self.store._get_page_dir(meta.doc_id, page_meta.page_id) / "page.png"
                            if image_path.exists():
                                page_data.append({
                                    "doc_id": meta.doc_id,
                                    "page_id": page_meta.page_id,
                                    "image_path": str(image_path)
                                })
                    
                    if not page_data:
                        status += "❌ ColPali build failed: No page images found\n"
                        status += "   Please ingest documents with OCR to generate page images\n\n"
                    else:
                        # Convert page_data to format expected by build_index
                        page_image_paths = [(p["doc_id"], p["page_id"], p["image_path"]) for p in page_data]
                        retriever.build_index(page_image_paths)
                        
                        index_dir = Path(self.config.indices_dir) / f"colpali_{suffix}"
                        index_dir.mkdir(parents=True, exist_ok=True)
                        retriever.save(index_dir)
                        
                        # Reload
                        self.retrievers["colpali"] = retriever
                        status += f"✅ ColPali index built: colpali_{suffix} ({len(page_data)} pages)\n\n"
                except Exception as e:
                    import traceback
                    status += f"❌ ColPali build failed: {str(e)}\n{traceback.format_exc()}\n\n"
            
            status += "=" * 50 + "\n"
            status += "🎉 Index building complete!\n"
            status += f"Available modes: {list(self.retrievers.keys())}\n"
            status += "\nYou can now use the Query & Answer tab."
            
            return status
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            return error_msg

    def _handle_refresh_docs(self) -> List:
        """Refresh document list."""
        return self._get_doc_list()

    def _handle_delete_doc(self, doc_id: str) -> Tuple[str, List]:
        """Delete a document."""
        try:
            if not doc_id:
                return "Error: No doc_id provided", self._get_doc_list()
            
            self.store.delete_document(doc_id)
            return f"✅ Deleted: {doc_id}", self._get_doc_list()
            
        except Exception as e:
            return f"Error: {str(e)}", self._get_doc_list()

    def _handle_query(
        self,
        question: str,
        doc_filter: str,
        retrieval_mode: str
    ) -> Tuple[str, List, str]:
        """Handle query with selected retrieval mode."""
        try:
            if not question:
                return "Please enter a question", [], ""
            
            # Switch retriever
            retriever = self.retrievers.get(retrieval_mode)
            if not retriever:
                return f"Error: Retrieval mode '{retrieval_mode}' not available", [], ""
            
            self.pipeline.retriever = retriever
            
            # Parse doc filter
            doc_ids = None
            if doc_filter.strip():
                doc_ids = [d.strip() for d in doc_filter.split(",") if d.strip()]
            
            # Create query input
            query_input = QueryInput(
                query_id=str(uuid.uuid4()),
                question=question,
                doc_ids=doc_ids
            )
            
            # Run pipeline
            result = self.pipeline.run(query_input, self.config)
            
            # Format evidence table
            evidence_rows = []
            for i, ev in enumerate(result.evidence_items, 1):
                source = ev.metadata.get("source", retrieval_mode)
                evidence_rows.append([
                    i,
                    source,
                    ev.doc_id,
                    ev.page_id,
                    f"{ev.score:.4f}",
                    ev.snippet[:100] + "..." if len(ev.snippet) > 100 else ev.snippet
                ])
            
            return result.answer, evidence_rows, query_input.query_id
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return error_msg, [], ""

    def _handle_eval(
        self,
        eval_file,
        eval_mode: str
    ) -> Tuple[str, dict, Optional[str], Optional[str]]:
        """Handle batch evaluation."""
        try:
            if eval_file is None:
                return "Error: No evaluation file uploaded", {}, None, None
            
            # Switch retriever
            retriever = self.retrievers.get(eval_mode)
            if not retriever:
                return f"Error: Retrieval mode '{eval_mode}' not available", {}, None, None
            
            self.pipeline.retriever = retriever
            
            # Run evaluation
            from impl.eval_runner import load_dataset_from_csv, load_dataset_from_json
            
            if eval_file.name.endswith('.csv'):
                dataset = load_dataset_from_csv(eval_file.name)
            else:
                dataset = load_dataset_from_json(eval_file.name)
            
            report = self.eval_runner.run(dataset, self.config)
            
            # Save results
            report_dir = Path(self.config.reports_dir) / f"eval_{uuid.uuid4().hex[:8]}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            csv_path = str(report_dir / "predictions.csv")
            json_path = str(report_dir / "report.json")
            
            self.eval_runner.save_results(report, report_dir)
            
            status = f"✅ Evaluation complete\n"
            status += f"Mode: {eval_mode}\n"
            status += f"Samples: {len(dataset)}\n"
            status += f"Results saved to: {report_dir}"
            
            return status, report.metrics, csv_path, json_path
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return error_msg, {}, None, None

    def _check_ocr_service(self) -> bool:
        """Check if OCR service is available."""
        import requests
        
        endpoint = self.config.ocr.get('endpoint', 'http://localhost:8000')
        
        try:
            response = requests.get(f"{endpoint}/health", timeout=3)
            return response.status_code == 200
        except:
            return False

    def _get_doc_list(self) -> List:
        """Get list of documents."""
        try:
            docs = self.store.list_documents()
            rows = []
            for meta in docs:
                rows.append([
                    meta.doc_id,
                    meta.title,
                    meta.page_count,
                    meta.created_at
                ])
            return rows
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []


def main():
    """Launch UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Doc RAG UI V1")
    parser.add_argument("--config", default="configs/app.yaml", help="Config file path")
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    
    args = parser.parse_args()
    
    ui = DocRAGUIV1(args.config)
    ui.launch(share=args.share)


if __name__ == "__main__":
    main()
