"""
Enhanced Gradio UI V1 for doc-rag-evidence system.
Supports multiple retrieval modes: BM25, Dense, ColPali, Hybrid.
"""
from __future__ import annotations

import os
import uuid
import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Generator
import sys

# Clear proxy settings to avoid localhost connection issues
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 
                  'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY']:
    os.environ.pop(proxy_var, None)

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
from impl.index_incremental import IncrementalIndexManager
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, VLLMEmbedder
from impl.index_colpali import ColPaliRetriever
from impl.selector_topk import TopKEvidenceSelector
from impl.generator_template import TemplateGenerator
from impl.eval_runner import EvalRunner


class DocRAGUIV1:
    """Enhanced Gradio UI with multi-mode retrieval."""

    def __init__(self, config_path: str = "configs/app.yaml"):
        # Simple progress bar using sys.stdout
        import sys

        def progress_step(step: int, total: int, message: str):
            """Display progress step."""
            bar_width = 40
            filled = int(bar_width * step / total)
            bar = "█" * filled + "░" * (bar_width - filled)
            pct = int(100 * step / total)
            sys.stdout.write(f"\r[{bar}] {pct}% - {message}")
            sys.stdout.flush()
            if step == total:
                print()  # New line when complete

        total_steps = 7  # Total initialization steps

        # Step 1: Load config
        progress_step(1, total_steps, "Loading configuration...")
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        self.config = AppConfig(**config_dict)

        # Step 2: Initialize infrastructure
        progress_step(2, total_steps, "Initializing document store...")
        self.store = DocumentStoreLocal(self.config)

        progress_step(3, total_steps, "Initializing run logger...")
        self.logger = RunLoggerLocal(self.config)

        # Step 4: Initialize task manager
        from infra.task_manager import TaskManager
        self.task_manager = TaskManager(tasks_dir="data/tasks")

        # Step 5: Initialize retrievers
        progress_step(4, total_steps, "Loading retrieval indices...")
        self.retrievers = {}
        self._init_retrievers()

        # Step 6: Initialize selector and generator
        progress_step(5, total_steps, "Initializing evidence selector...")
        self.selector = TopKEvidenceSelector(snippet_length=500)

        progress_step(6, total_steps, "Loading LLM generator...")
        generator_type = self.config.generator.get("type", "template")
        if generator_type == "qwen3_vl":
            try:
                from impl.generator_qwen_llm import QwenLLMGenerator
                self.generator = QwenLLMGenerator(self.config)
                self._generator_name = "QwenLLMGenerator"
            except Exception as e:
                print(f"\n⚠️  Failed to load QwenLLMGenerator: {e}, falling back to template")
                from impl.generator_template import TemplateGenerator
                self.generator = TemplateGenerator(mode="summary")
                self._generator_name = "TemplateGenerator"
        else:
            from impl.generator_template import TemplateGenerator
            self.generator = TemplateGenerator(mode="summary")
            self._generator_name = "TemplateGenerator"

        # Step 7: Create pipeline
        progress_step(7, total_steps, "Building retrieval pipeline...")
        default_retriever = self.retrievers.get(self.config.retrieval_mode)
        if not default_retriever:
            default_retriever = self.retrievers.get("bm25")

        self.pipeline = Pipeline(
            retriever=default_retriever,
            selector=self.selector,
            generator=self.generator,
            logger=self.logger,
            reranker=None,
            store=self.store  # Enable hit normalization
        )

        # Eval runner
        self.eval_runner = EvalRunner(self.pipeline)

        # Print summary
        print(f"\n{'='*60}")
        print(f"🚀 Doc RAG Evidence System V1 - Ready!")
        print(f"{'='*60}")
        print(f"📁 Config: {config_path}")
        print(f"🔍 Available modes: {', '.join(list(self.retrievers.keys()))}")
        print(f"🤖 Generator: {self._generator_name}")
        print(f"🎯 Default mode: {self.config.retrieval_mode}")
        print(f"{'='*60}\n")

    def _init_retrievers(self):
        """Initialize available retrievers based on config."""
        indices_dir = Path(self.config.indices_dir)
        
        # BM25 (always try to load)
        bm25_index_name = "bm25_default"
        try:
            retriever = BM25IndexerRetriever(self.store)
            retriever.load(self.config, index_name=bm25_index_name)
            self.retrievers["bm25"] = retriever
            print(f"✅ Loaded BM25 index: {len(retriever.units)} units")
        except Exception as e:
            print(f"❌ Failed to load BM25 index: {e}")
        
        # OCR-BM25 (load if exists)
        ocr_bm25_index_name = "bm25_ocr"
        ocr_bm25_index_dir = indices_dir / ocr_bm25_index_name
        if ocr_bm25_index_dir.exists():
            try:
                retriever = BM25IndexerRetriever(self.store)
                retriever.load(self.config, index_name=ocr_bm25_index_name)
                self.retrievers["ocr-bm25"] = retriever
                print(f"✅ Loaded OCR-BM25 index: {len(retriever.units)} units")
            except Exception as e:
                print(f"❌ Failed to load OCR-BM25 index: {e}")
        
        # Dense (vLLM embedding)
        if self.config.dense.get("enabled"):
            dense_index_name = "dense_default"
            dense_index_dir = indices_dir / dense_index_name
            dense_meta_path = dense_index_dir / "dense_meta.json"
            # Check if both directory and meta file exist
            if dense_index_dir.exists() and dense_meta_path.exists():
                try:
                    embedder = VLLMEmbedder(
                        endpoint=self.config.dense["endpoint"],
                        model=self.config.dense["model"],
                        batch_size=self.config.dense.get("batch_size", 32)
                    )
                    retriever = DenseIndexerRetriever.load(dense_index_dir, embedder)
                    self.retrievers["dense"] = retriever
                    print(f"✅ Loaded Dense index: {len(retriever.units)} units (vLLM @ {self.config.dense['endpoint']})")
                except Exception as e:
                    print(f"❌ Failed to load Dense index: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  Dense index not found at {dense_index_dir}")
            
            # OCR-Dense (load if exists)
            ocr_dense_index_name = "dense_ocr"
            ocr_dense_index_dir = indices_dir / ocr_dense_index_name
            ocr_dense_meta_path = ocr_dense_index_dir / "dense_meta.json"
            if ocr_dense_index_dir.exists() and ocr_dense_meta_path.exists():
                try:
                    embedder = VLLMEmbedder(
                        endpoint=self.config.dense["endpoint"],
                        model=self.config.dense["model"],
                        batch_size=self.config.dense.get("batch_size", 32)
                    )
                    retriever = DenseIndexerRetriever.load(ocr_dense_index_dir, embedder)
                    self.retrievers["ocr-dense"] = retriever
                    print(f"✅ Loaded OCR-Dense index: {len(retriever.units)} units (vLLM @ {self.config.dense['endpoint']})")
                except Exception as e:
                    print(f"❌ Failed to load OCR-Dense index: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  OCR-Dense index not found at {ocr_dense_index_dir}")
        
        # ColPali (vision embedding) - 延迟加载，只记录配置
        if self.config.colpali.get("enabled"):
            colpali_index_name = "colpali_default"
            colpali_index_dir = indices_dir / colpali_index_name
            if colpali_index_dir.exists():
                # 不立即加载模型，只注册可用性
                self.retrievers["colpali"] = None  # Placeholder，延迟加载
                gpu_id = self.config.colpali.get("gpu", 0)
                device = f"cuda:{gpu_id}"
                self._colpali_config = {
                    "index_dir": colpali_index_dir,
                    "model_name": self.config.colpali["model"],
                    "device": device
                }
                print(f"✅ ColPali index available (延迟加载模式, GPU {gpu_id})")
            else:
                print(f"⚠️  ColPali index not found at {colpali_index_dir}")
        
        # Dense-VL (multimodal embedding, lazy load like ColPali)
        if self.config.dense_vl.get("enabled"):
            dense_vl_index_name = "dense_vl_default"
            dense_vl_index_dir = indices_dir / dense_vl_index_name
            if dense_vl_index_dir.exists():
                # Register as available with lazy loading
                self.retrievers["dense_vl"] = None  # Placeholder
                self._dense_vl_config = {
                    "index_dir": dense_vl_index_dir,
                    "model_path": self.config.dense_vl["model_path"],
                    "gpu": self.config.dense_vl.get("gpu", 1),
                    "gpu_memory": self.config.dense_vl.get("gpu_memory", 0.45),
                    "batch_size": self.config.dense_vl.get("batch_size", 8),
                    "max_image_size": self.config.dense_vl.get("max_image_size", 1024)
                }
                print(f"✅ Dense-VL index available at {dense_vl_index_dir} (延迟加载模式)")
            else:
                print(f"⚠️  Dense-VL index not found at {dense_vl_index_dir}")
        
        # Note: Hybrid retrievers are created dynamically in UI based on user selection
        # No pre-configured hybrid combinations needed

    def cleanup(self):
        """Clean up lazy-loaded models and release GPU memory."""
        print("\n🧹 Cleaning up resources...")
        
        # Shutdown task manager
        try:
            self.task_manager.shutdown()
            print("  ✓ Task manager shutdown")
        except Exception as e:
            print(f"  ⚠️  Error shutting down task manager: {e}")
        
        # Clear ColPali model
        if self.retrievers.get("colpali") is not None:
            try:
                del self.retrievers["colpali"]
                self.retrievers["colpali"] = None
                print("  ✓ ColPali model released")
            except Exception as e:
                print(f"  ⚠️  Error releasing ColPali: {e}")
        
        # Clear Dense-VL model
        if self.retrievers.get("dense_vl") is not None:
            try:
                del self.retrievers["dense_vl"]
                self.retrievers["dense_vl"] = None
                print("  ✓ Dense-VL model released")
            except Exception as e:
                print(f"  ⚠️  Error releasing Dense-VL: {e}")
        
        # Force garbage collection and clear CUDA cache
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("  ✓ CUDA cache cleared")
        except:
            pass
        
        print("✅ Cleanup complete\n")

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

        try:
            demo.launch(
                share=share, 
                server_name="127.0.0.1",  # Changed from 0.0.0.0 to fix 502 error
                server_port=7860,
                show_error=True,
                quiet=False
            )
        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt received")
            self.cleanup()
            raise
        except Exception as e:
            print(f"❌ Failed to launch Gradio: {e}")
            # Try alternative port
            print("⚠️  Trying alternative port 7861...")
            try:
                demo.launch(
                    share=share,
                    server_name="127.0.0.1",
                    server_port=7861,
                    show_error=True
                )
            except KeyboardInterrupt:
                print("\n⚠️  Keyboard interrupt received")
                self.cleanup()
                raise
        finally:
            # Always cleanup when exiting
            self.cleanup()

    def _build_document_tab(self):
        """Build document management tab."""
        gr.Markdown("## Upload and Manage Documents")
        
        gr.Markdown("""
        ### 💡 Tips for Large Batch Upload
        - **Batch Processing**: Files are automatically processed in batches of 50 to avoid memory issues
        - **Progress Tracking**: Real-time progress updates every 10 files
        - **OCR Performance**: With OCR enabled, expect ~10-30s per file depending on pages and quality
        - **Recommended**: For 1000+ files, consider uploading in multiple sessions or use command-line tools
        """)

        # Section 1: Upload & Ingest
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📤 Upload PDF (Supports Multiple Files)")
                
                with gr.Accordion("Select Files", open=True):
                    pdf_files = gr.File(
                        label="", 
                        file_types=[".pdf"],
                        file_count="multiple",
                        type="filepath",
                        show_label=False,
                        height=300  # 增加到300px
                    )
                    # File info display
                    file_info = gr.Dataframe(
                        headers=["Filename", "Size (MB)", "Pages"],
                        label="Uploaded Files Info",
                        interactive=False,
                        max_height=200
                    )
                
                with gr.Row():
                    upload_btn = gr.Button("📤 Upload Files", variant="primary", scale=2)
                    preview_btn = gr.Button("👁️ Preview Selected", variant="secondary", scale=1)
                
                with gr.Accordion("PDF Preview", open=False) as preview_accordion:
                    # File selector for preview (if multiple files)
                    preview_file_selector = gr.Dropdown(
                        label="Select file to preview",
                        choices=[],
                        interactive=True
                    )
                    
                    # Page navigation controls
                    with gr.Row():
                        prev_page_btn = gr.Button("◀ Previous", size="sm", scale=1)
                        page_info_display = gr.Markdown("Page 1 / 1")
                        next_page_btn = gr.Button("Next ▶", size="sm", scale=1)
                    
                    preview_page_num = gr.Slider(
                        minimum=1, 
                        maximum=1, 
                        value=1, 
                        step=1, 
                        label="Jump to page"
                    )
                    
                    # Zoom control
                    zoom_level = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=0.5,
                        label="Zoom level (higher = better quality, slower)"
                    )
                    
                    preview_image = gr.Image(
                        label="Preview (click to view full size)", 
                        type="filepath",
                        show_label=True,
                        height=600,
                        interactive=False
                    )
                
                with gr.Accordion("Upload Status", open=True):
                    upload_status = gr.Textbox(
                        label="",
                        lines=6,
                        max_lines=6,
                        interactive=False
                    )
                
                with gr.Accordion("📋 Background Tasks", open=True):
                    gr.Markdown("View running and completed tasks. **Click 🔄 Refresh to see latest status**.")
                    with gr.Row():
                        refresh_tasks_btn = gr.Button("🔄 Refresh Tasks", scale=1)
                        clear_tasks_btn = gr.Button("🗑️ Clear Completed", scale=1)
                    tasks_display = gr.Dataframe(
                        headers=["Task ID", "Type", "Status", "Progress", "Current Step", "Created"],
                        label="Task List",
                        interactive=False,
                        max_height=300
                    )

            with gr.Column():
                gr.Markdown("### 📚 Document List")
                
                # Statistics and pagination controls
                with gr.Row():
                    # Get initial stats
                    initial_stats, initial_list, initial_page_info = self._get_doc_list_paginated(1, 20)
                    doc_stats = gr.Markdown(initial_stats)
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                
                with gr.Row():
                    page_number = gr.Number(label="Page", value=1, minimum=1, precision=0, scale=1)
                    page_size = gr.Dropdown(
                        choices=[10, 20, 50, 100],
                        value=20,
                        label="Items per page",
                        scale=1
                    )
                
                doc_list = gr.Dataframe(
                    headers=["Filename", "Doc ID", "Ingested", "Pages", "OCR", "Index Status", "Uploaded At"],
                    label="",
                    interactive=False,
                    wrap=True,
                    value=initial_list
                )
                
                page_info = gr.Markdown(initial_page_info)
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Process Documents")
                
                with gr.Row():
                    ingest_all_btn = gr.Button("📥 Ingest All Unprocessed", variant="primary", scale=1)
                    ingest_status = gr.Textbox(label="Ingest Status", lines=2, interactive=False, scale=2)
                
                delete_docid = gr.Textbox(label="Document ID to Delete", placeholder="Enter doc_id")
                delete_btn = gr.Button("🗑️ Delete Document", variant="stop")
                delete_status = gr.Textbox(label="Delete Status", lines=1, interactive=False)
                
                gr.Markdown("---")
                
                # OCR Processing section
                gr.Markdown("### 🔍 OCR Processing")
                gr.Markdown("Re-process documents with OCR (useful if you uploaded without OCR initially)")
                
                ocr_docid = gr.Textbox(
                    label="Document IDs for OCR", 
                    placeholder="doc1,doc2 (comma-separated) or leave empty",
                    lines=1
                )
                
                with gr.Row():
                    ocr_selected_btn = gr.Button("🔍 OCR Selected Documents", scale=1)
                    ocr_all_non_ocr_btn = gr.Button("🔍 OCR All Non-OCR Documents", scale=1, variant="primary")
                
                ocr_status = gr.Textbox(label="OCR Processing Status", lines=3, interactive=False)

        gr.Markdown("---")
        
        # Section 2: Build Indices
        gr.Markdown("### 🔧 Build Indices")
        gr.Markdown("After uploading documents, build indices for retrieval")
        gr.Markdown("⚠️ **Note**: Build one index at a time to avoid GPU OOM")
        
        with gr.Row():
            with gr.Column():
                index_type = gr.Radio(
                    choices=[
                        "bm25",
                        "ocr-bm25",
                        "dense",
                        "ocr-dense",
                        "dense_vl",
                        "colpali"
                    ],
                    value="bm25",
                    label="Select Index Type",
                    info="OCR variants only index documents processed with OCR"
                )
                index_name_suffix = gr.Textbox(
                    label="Index Name Suffix (optional)",
                    placeholder="default",
                    value="default"
                )
                build_btn = gr.Button("⚙️ Build Index", variant="primary", size="lg")
                
            with gr.Column():
                build_status = gr.Textbox(
                    label="Build Status",
                    lines=15,
                    max_lines=15,
                    interactive=False,
                    placeholder="Status will appear here..."
                )

        # Event handlers
        # File selection handler - show file info and populate preview file selector
        pdf_files.change(
            fn=self._handle_file_selection,
            inputs=[pdf_files],
            outputs=[file_info, preview_file_selector]
        )
        
        # Preview handler - open accordion and show preview
        preview_btn.click(
            fn=self._handle_preview_pdf,
            inputs=[pdf_files, preview_file_selector, preview_page_num, zoom_level],
            outputs=[preview_image, preview_page_num, page_info_display, preview_accordion]
        )
        
        # Page navigation
        prev_page_btn.click(
            fn=self._handle_prev_page,
            inputs=[pdf_files, preview_file_selector, preview_page_num, zoom_level],
            outputs=[preview_image, preview_page_num, page_info_display]
        )
        
        next_page_btn.click(
            fn=self._handle_next_page,
            inputs=[pdf_files, preview_file_selector, preview_page_num, zoom_level],
            outputs=[preview_image, preview_page_num, page_info_display]
        )
        
        # Page number or zoom change
        preview_page_num.change(
            fn=self._handle_preview_update,
            inputs=[pdf_files, preview_file_selector, preview_page_num, zoom_level],
            outputs=[preview_image, page_info_display]
        )
        
        zoom_level.change(
            fn=self._handle_preview_update,
            inputs=[pdf_files, preview_file_selector, preview_page_num, zoom_level],
            outputs=[preview_image, page_info_display]
        )
        
        # File selector change
        preview_file_selector.change(
            fn=self._handle_file_selector_change,
            inputs=[pdf_files, preview_file_selector, zoom_level],
            outputs=[preview_image, preview_page_num, page_info_display]
        )
        
        upload_btn.click(
            fn=self._handle_batch_upload,
            inputs=[pdf_files],
            outputs=[upload_status, doc_stats, doc_list, page_info]
        )
        
        # Task management
        refresh_tasks_btn.click(
            fn=self._handle_refresh_tasks,
            inputs=[],
            outputs=[tasks_display]
        )
        
        clear_tasks_btn.click(
            fn=self._handle_clear_completed_tasks,
            inputs=[],
            outputs=[upload_status, tasks_display]
        )

        # Pagination event handlers
        refresh_btn.click(
            fn=self._handle_refresh_docs_paginated,
            inputs=[page_number, page_size],
            outputs=[doc_stats, doc_list, page_info]
        )
        
        page_number.change(
            fn=self._handle_refresh_docs_paginated,
            inputs=[page_number, page_size],
            outputs=[doc_stats, doc_list, page_info]
        )
        
        page_size.change(
            fn=self._handle_refresh_docs_paginated,
            inputs=[page_number, page_size],
            outputs=[doc_stats, doc_list, page_info]
        )
        
        # Ingest all button
        ingest_all_btn.click(
            fn=self._handle_ingest_all,
            inputs=[],
            outputs=[ingest_status, doc_stats, doc_list, page_info]
        )

        delete_btn.click(
            fn=self._handle_delete_doc,
            inputs=[delete_docid, page_number, page_size],
            outputs=[delete_status, doc_stats, doc_list, page_info]
        )
        
        # OCR processing event handlers
        ocr_selected_btn.click(
            fn=self._handle_ocr_selected,
            inputs=[ocr_docid, page_number, page_size],
            outputs=[ocr_status, doc_stats, doc_list, page_info],
            show_progress=True
        )
        
        ocr_all_non_ocr_btn.click(
            fn=self._handle_ocr_all_non_ocr,
            inputs=[page_number, page_size],
            outputs=[ocr_status, doc_stats, doc_list, page_info],
            show_progress=True
        )
        
        build_btn.click(
            fn=self._handle_build_indices,
            inputs=[index_type, index_name_suffix],
            outputs=[build_status]
        )

    def _build_query_tab(self):
        """Build query & answer tab."""
        gr.Markdown("## Ask Questions")

        with gr.Row():
            with gr.Column(scale=1):
                # Retrieval mode selector - simplified to basic modes + hybrid
                # Include retrievers that exist (including None placeholders with configs)
                available_modes = []
                for mode in ["bm25", "ocr-bm25", "dense", "ocr-dense", "colpali", "dense_vl"]:
                    if mode in self.retrievers:
                        # Include if retriever exists or if it's a lazy-load placeholder with config
                        if self.retrievers[mode] is not None:
                            available_modes.append(mode)
                        elif mode == "colpali" and hasattr(self, "_colpali_config"):
                            available_modes.append(mode)
                        elif mode == "dense_vl" and self._dense_vl_config:
                            available_modes.append(mode)
                
                if len(available_modes) >= 2:
                    available_modes.append("hybrid")
                
                retrieval_mode = gr.Radio(
                    choices=available_modes,
                    value=available_modes[0] if available_modes else "bm25",
                    label="Retrieval Mode",
                    info="单一检索 or Hybrid | OCR variants use only OCR-indexed documents"
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
        
        # Evidence format selector (outside columns for better visibility)
        evidence_mode = gr.Radio(
            choices=["text", "image"],
            value="text",
            label="Evidence Format",
            info="text: 使用文本snippet | image: 使用完整页面图片（更准确，适合VL模型）"
        )
        
        # Hybrid fusion settings - only visible when hybrid mode is selected
        with gr.Accordion("⚙️ Hybrid Retrieval Configuration", open=True, visible=False) as hybrid_config:
            gr.Markdown("### 自定义混合检索配置")
            gr.Markdown("选择两个不同的检索器并设置权重，系统将自动融合结果")
            
            with gr.Row():
                with gr.Column():
                    fusion_method = gr.Radio(
                        choices=["weighted_sum", "rrf"],
                        value="rrf",
                        label="Fusion Method (融合方法)",
                        info="rrf: 推荐，对分数尺度不敏感 | weighted_sum: 分数加权"
                    )
                    
                    gr.Markdown("""
                    **融合方法说明**：
                    - **RRF (推荐)**: `score = w1/(60+rank1) + w2/(60+rank2)`
                      - 基于排名，对分数尺度不敏感
                      - 适合跨模态融合（如 BM25 + ColPali）
                    
                    - **Weighted Sum**: `score = w1*score1 + w2*score2`
                      - 基于原始分数加权
                      - 适合分数尺度相似的检索器
                    """)
                
                with gr.Column():
                    gr.Markdown("#### 检索器与权重")
                    
                    retriever_1 = gr.Dropdown(
                        choices=["bm25", "ocr-bm25", "dense", "ocr-dense", "colpali", "dense_vl"],
                        value="bm25",
                        label="检索器 1",
                        info="第一个检索器"
                    )
                    
                    retriever_2 = gr.Dropdown(
                        choices=["bm25", "ocr-bm25", "dense", "ocr-dense", "colpali", "dense_vl"],
                        value="dense",
                        label="检索器 2",
                        info="第二个检索器（必须与检索器1不同）"
                    )
                    
                    weight_1 = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="检索器 1 权重",
                        info="检索器2权重自动为 1 - weight_1"
                    )
                    
                    weight_display = gr.Markdown(
                        "**当前权重**: 检索器1 = 0.50, 检索器2 = 0.50"
                    )
            
            gr.Markdown("---")
            gr.Markdown("💡 **推荐配置**: BM25(0.5) + Dense(0.5) with RRF | Dense-VL(0.6) + BM25(0.4) with RRF | Dense(0.6) + ColPali(0.4) with RRF")
            
            # Update weight display when slider changes
            def update_weight_display(w1):
                w2 = 1.0 - w1
                return f"**当前权重**: 检索器1 = {w1:.2f}, 检索器2 = {w2:.2f}"
            
            weight_1.change(
                fn=update_weight_display,
                inputs=[weight_1],
                outputs=[weight_display]
            )
            
            # Toggle hybrid config visibility based on retrieval mode
            def toggle_hybrid_config(mode):
                return gr.update(visible=(mode == "hybrid"), open=(mode == "hybrid"))
            
            retrieval_mode.change(
                fn=toggle_hybrid_config,
                inputs=[retrieval_mode],
                outputs=[hybrid_config]
            )

        gr.Markdown("### 📑 Evidence")
        evidence_display = gr.Dataframe(
            headers=["Rank", "Source", "Doc ID", "Page", "Score", "Snippet"],
            label="Retrieved Evidence",
            interactive=False
        )
        
        # Image gallery for displaying page images (visible in image mode)
        with gr.Accordion("🖼️ Page Images", open=True, visible=False) as image_gallery_section:
            gr.Markdown("**检索到的页面图片** (按相关性排序)")
            page_images = gr.Gallery(
                label="Page Images",
                show_label=False,
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain"
            )

        query_id_box = gr.Textbox(label="Query ID (for traceability)", interactive=False)

        # Toggle image gallery visibility based on evidence mode
        def toggle_image_gallery(mode):
            return gr.update(visible=(mode == "image"), open=(mode == "image"))
        
        evidence_mode.change(
            fn=toggle_image_gallery,
            inputs=[evidence_mode],
            outputs=[image_gallery_section]
        )
        
        # Event handler
        query_btn.click(
            fn=self._handle_query,
            inputs=[
                question, 
                doc_filter, 
                retrieval_mode, 
                evidence_mode,
                fusion_method,
                retriever_1,
                retriever_2,
                weight_1
            ],
            outputs=[answer_box, evidence_display, query_id_box, page_images]
        )

    def _build_eval_tab(self):
        """Build evaluation tab."""
        gr.Markdown("## Batch Evaluation")
        gr.Markdown("""
        ### 📋 评估流程说明
        1. **上传CSV文件**：包含 `qid`, `question`, `answer_gt` 三列
        2. **系统自动问答**：对每个question运行完整RAG pipeline
        3. **生成预测答案**：得到系统的predicted answer
        4. **VL自动评分**：使用Qwen3-VL对比predicted和ground truth
        5. **输出详细报告**：包含评分、理由和统计指标
        """)

        with gr.Row():
            with gr.Column():
                eval_file = gr.File(label="Upload Eval Dataset (CSV or JSON)", file_types=[".csv", ".json"])
                
                # Retrieval mode for evaluation - include lazy-load modes
                available_modes = []
                for mode in ["bm25", "ocr-bm25", "dense", "ocr-dense", "colpali", "dense_vl"]:
                    if mode in self.retrievers:
                        if self.retrievers[mode] is not None:
                            available_modes.append(mode)
                        elif mode == "colpali" and hasattr(self, "_colpali_config"):
                            available_modes.append(mode)
                        elif mode == "dense_vl" and self._dense_vl_config:
                            available_modes.append(mode)
                
                if len(available_modes) >= 2:
                    available_modes.append("hybrid")
                
                eval_mode = gr.Radio(
                    choices=available_modes,
                    value=available_modes[0] if available_modes else "bm25",
                    label="Retrieval Mode for Evaluation",
                    info="单一检索 or Hybrid | OCR variants use only OCR-indexed documents"
                )
                
                # Evidence format selector
                eval_evidence_mode = gr.Radio(
                    choices=["text", "image"],
                    value="text",
                    label="Evidence Format",
                    info="text: 使用文本snippet | image: 使用完整页面图片（更准确，适合VL模型）"
                )
                
                # VL-based answer scoring option
                enable_vl_scoring = gr.Checkbox(
                    label="Enable VL Answer Scoring",
                    value=True,
                    info="✨ 使用Qwen3-VL自动评估：对比系统生成答案和ground truth，给出0-10分及详细评价"
                )
                
                # Hybrid fusion settings for evaluation - only visible for hybrid mode
                with gr.Accordion("⚙️ Hybrid Configuration", open=True, visible=False) as eval_hybrid_config:
                    gr.Markdown("### 评估中的混合检索配置")
                    
                    with gr.Row():
                        with gr.Column():
                            eval_fusion_method = gr.Radio(
                                choices=["weighted_sum", "rrf"],
                                value="rrf",
                                label="Fusion Method",
                                info="rrf: 推荐 | weighted_sum: 分数加权"
                            )
                        
                        with gr.Column():
                            eval_retriever_1 = gr.Dropdown(
                                choices=["bm25", "ocr-bm25", "dense", "ocr-dense", "colpali", "dense_vl"],
                                value="bm25",
                                label="检索器 1"
                            )
                            
                            eval_retriever_2 = gr.Dropdown(
                                choices=["bm25", "ocr-bm25", "dense", "ocr-dense", "colpali", "dense_vl"],
                                value="dense",
                                label="检索器 2"
                            )
                            
                            eval_weight_1 = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="检索器 1 权重"
                            )
                            
                            eval_weight_display = gr.Markdown(
                                "**当前权重**: 检索器1 = 0.50, 检索器2 = 0.50"
                            )
                    
                    gr.Markdown("💡 评估将使用以上 Hybrid 配置")
                    
                    # Update weight display
                    def update_eval_weight_display(w1):
                        w2 = 1.0 - w1
                        return f"**当前权重**: 检索器1 = {w1:.2f}, 检索器2 = {w2:.2f}"
                    
                    eval_weight_1.change(
                        fn=update_eval_weight_display,
                        inputs=[eval_weight_1],
                        outputs=[eval_weight_display]
                    )
                    
                    # Toggle hybrid config visibility
                    def toggle_eval_hybrid_config(mode):
                        return gr.update(visible=(mode == "hybrid"), open=(mode == "hybrid"))
                    
                    eval_mode.change(
                        fn=toggle_eval_hybrid_config,
                        inputs=[eval_mode],
                        outputs=[eval_hybrid_config]
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
            inputs=[
                eval_file, 
                eval_mode,
                eval_evidence_mode,
                enable_vl_scoring,
                eval_fusion_method,
                eval_retriever_1,
                eval_retriever_2,
                eval_weight_1
            ],
            outputs=[eval_status, eval_metrics, download_csv, download_json]
        )

    # ========== Event Handlers ==========
    
    # Background task functions
    @staticmethod
    def _background_ingest_task(task_id: str, task_manager, config, store, uploaded_files):
        """Background task for document ingestion."""
        from impl.ingest_pdf_v1 import PDFIngestorV1
        
        success_count = 0
        failed_count = 0
        
        for idx, (filename, persistent_path, doc_id) in enumerate(uploaded_files, 1):
            try:
                # Update progress
                task_manager.update_task_progress(
                    task_id,
                    processed_items=idx - 1,
                    current_step=f"Processing {filename}..."
                )
                
                # Ingest document
                ingestor = PDFIngestorV1(
                    config=config,
                    store=store,
                    use_ocr=False
                )
                
                meta = ingestor.ingest(str(persistent_path))
                success_count += 1
                
            except Exception as e:
                failed_count += 1
                print(f"❌ Failed to process {filename}: {e}")
        
        # Final update
        task_manager.update_task_progress(
            task_id,
            processed_items=len(uploaded_files),
            current_step="Completed"
        )
        
        return {
            "success": success_count,
            "failed": failed_count,
            "total": len(uploaded_files)
        }

    def _handle_batch_upload(self, pdf_files) -> Tuple[str, str, List, str]:
        """Handle batch PDF upload - only copy files to uploads directory."""
        try:
            print(f"[DEBUG] _handle_batch_upload called with {len(pdf_files) if pdf_files else 0} files")
            
            if pdf_files is None or len(pdf_files) == 0:
                print("[DEBUG] No files uploaded")
                stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
                return "❌ Error: No files uploaded", stats, doc_list, page_info
            
            # Handle single file or multiple files
            if not isinstance(pdf_files, list):
                pdf_files = [pdf_files]
            
            total_files = len(pdf_files)
            
            import shutil
            
            # Create uploads directory for persistent storage
            uploads_dir = Path("data/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            # Upload files to backend
            uploaded_count = 0
            skipped_count = 0
            failed_files = []
            
            for pdf_file in pdf_files:
                try:
                    filename = Path(pdf_file.name).name
                    persistent_path = uploads_dir / filename
                    
                    # Check if file already exists in uploads
                    if persistent_path.exists():
                        skipped_count += 1
                        continue
                    
                    # Copy file to persistent storage
                    shutil.copy2(pdf_file.name, persistent_path)
                    uploaded_count += 1
                    
                except Exception as e:
                    print(f"Failed to upload {pdf_file.name}: {e}")
                    failed_files.append((Path(pdf_file.name).name, str(e)))
            
            stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
            
            status = f"✅ Upload Complete!\n"
            status += "=" * 50 + "\n"
            status += f"📤 Total files: {total_files}\n"
            status += f"✅ Uploaded: {uploaded_count}\n"
            status += f"⏭️ Skipped (already exists): {skipped_count}\n"
            
            if failed_files:
                status += f"❌ Failed: {len(failed_files)}\n"
                for fname, err in failed_files[:3]:
                    status += f"  - {fname}: {err[:50]}\n"
            
            status += "\n💡 Files saved to data/uploads/\n"
            status += "⚠️  Click 'Ingest All Unprocessed' to process documents"
            
            return status, stats, doc_list, page_info
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Upload Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
            return error_msg, stats, doc_list, page_info
    
    def _handle_upload(self, pdf_file) -> Tuple[str, str, List, str]:
        """Handle single PDF upload (legacy, kept for compatibility)."""
        # Redirect to batch handler
        return self._handle_batch_upload([pdf_file] if pdf_file else None)
    
    def _handle_ingest_all(self) -> Tuple[str, str, List, str]:
        """Ingest all unprocessed files in uploads directory."""
        try:
            from datetime import datetime
            
            # Get all PDF files from uploads directory
            uploads_dir = Path("data/uploads")
            pdf_files = list(uploads_dir.glob("*.pdf")) + list(uploads_dir.glob("*.PDF"))
            
            if not pdf_files:
                stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
                return "⚠️ No files found in uploads directory", stats, doc_list, page_info
            
            # Get ingested documents
            ingested_docs = {doc.doc_id: doc for doc in self.store.list_documents()}
            
            # Find unprocessed files
            unprocessed_files = []
            for pdf_file in pdf_files:
                doc_id = pdf_file.stem.replace(" ", "_").lower()
                if doc_id not in ingested_docs:
                    unprocessed_files.append((pdf_file.name, pdf_file, doc_id))
            
            if not unprocessed_files:
                stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
                status = f"✅ All files already ingested!\n"
                status += f"Total files: {len(pdf_files)}\n"
                status += f"Already ingested: {len(ingested_docs)}"
                return status, stats, doc_list, page_info
            
            # Submit background task for ingestion
            task_id = f"ingest_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.task_manager.submit_task(
                task_id=task_id,
                task_type="ingest",
                func=self._background_ingest_task,
                args=(self.config, self.store, unprocessed_files),
                total_items=len(unprocessed_files),
                description=f"Ingesting {len(unprocessed_files)} documents..."
            )
            
            stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
            
            status = f"✅ Ingest task submitted!\n"
            status += "=" * 50 + "\n"
            status += f"📋 Task ID: {task_id}\n"
            status += f"📁 Total files in uploads: {len(pdf_files)}\n"
            status += f"✅ Already ingested: {len(ingested_docs)}\n"
            status += f"⚙️ Queued for processing: {len(unprocessed_files)}\n\n"
            status += "💡 Processing in background. Click 'Refresh Tasks' to see progress.\n"
            status += "⚠️ You can close the UI - processing will continue."
            
            return status, stats, doc_list, page_info
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            stats, doc_list, page_info = self._get_doc_list_paginated(1, 20)
            return error_msg, stats, doc_list, page_info
    
    def _handle_refresh_tasks(self) -> List:
        """Refresh and display task list."""
        tasks = self.task_manager.list_tasks(limit=20)
        
        rows = []
        for task in tasks:
            task_id_short = task.task_id[:20] + "..." if len(task.task_id) > 20 else task.task_id
            progress_pct = f"{task.progress * 100:.1f}%"
            created_time = task.created_at[:19].replace('T', ' ')  # Truncate to minutes
            
            # Add status emoji
            status_emoji = {
                "pending": "⏳",
                "running": "▶️",
                "completed": "✓",
                "failed": "❌",
                "cancelled": "🚫"
            }
            status_display = f"{status_emoji.get(task.status.value, '❓')} {task.status.value}"
            
            # Current step or error message
            if task.status.value == "failed" and task.error_message:
                current_info = f"❌ {task.error_message[:40]}"
            else:
                current_info = task.current_step[:40]
            
            rows.append([
                task_id_short,
                task.task_type,
                status_display,
                progress_pct,
                current_info,
                created_time
            ])
        
        return rows
    
    def _handle_clear_completed_tasks(self) -> Tuple[str, List]:
        """Clear completed tasks."""
        removed = self.task_manager.clear_completed_tasks(keep_recent=5)
        tasks_rows = self._handle_refresh_tasks()
        
        message = f"✅ Cleared {removed} old tasks (kept 5 most recent)"
        return message, tasks_rows

    def _run_dense_vl_indexing(self, task_id: str, task_manager, index_name: str, gpu_id: int):
        """Background task function for Dense-VL indexing."""
        import subprocess
        import os
        import re
        
        script_path = Path(__file__).parent.parent.parent / "scripts" / "build_dense_vl_index.py"
        
        # Prepare environment with GPU setting
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Update task
        task_manager.update_task_progress(
            task_id,
            current_step=f"Starting Dense-VL indexing on GPU {gpu_id}...",
            progress=0.0
        )
        
        try:
            # Run the script
            process = subprocess.Popen(
                ["python", "-u", str(script_path), "--index-name", index_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1,
                universal_newlines=True
            )
            
            # Save PID for tracking
            task_manager.update_task_pid(task_id, process.pid)
            
            # Monitor progress from script output
            output_lines = []
            total_pages = None
            processed_pages = 0
            
            # Get task to access total_items if needed
            task = task_manager.get_task(task_id)
            if task and task.total_items > 0:
                # Use total_items from task creation as fallback
                total_pages = task.total_items
            
            for line in process.stdout:
                output_lines.append(line.rstrip())
                clean_line = line.strip()
                
                # Extract total pages count (multiple formats)
                # Format 1: "Total pages to embed: N" (single batch mode)
                # Format 2: "Total pages to embed: N new pages" (incremental mode)
                if "Total pages to embed:" in clean_line:
                    match = re.search(r'Total pages to embed:\s*(\d+)', clean_line)
                    if match:
                        pages_in_msg = int(match.group(1))
                        # In incremental mode, this is new pages count
                        # Use it if we don't have total_pages yet, or update it
                        if "new pages" in clean_line:
                            # Incremental mode: this is new pages, not total
                            # Keep using task.total_items as total
                            pass
                        else:
                            # Single batch mode: this is total
                            total_pages = pages_in_msg
                        
                        task_manager.update_task_progress(
                            task_id,
                            current_step=f"Total pages: {total_pages or pages_in_msg}",
                            progress=0.01
                        )
                
                # Extract worker progress from tqdm-style progress bars
                # Format: "Worker 0:  45%|████▌     | 100/225 [00:30<00:37,  3.33page/s]"
                elif "Worker" in clean_line and "|" in clean_line:
                    match = re.search(r'(\d+)/(\d+)', clean_line)
                    if match:
                        current = int(match.group(1))
                        worker_total = int(match.group(2))
                        # Update current step but don't calculate global progress yet
                        # (we need to aggregate all workers)
                        task_manager.update_task_progress(
                            task_id,
                            current_step=clean_line[:100]
                        )
                
                # Extract batch save progress
                elif "Batch saved:" in clean_line and "total pages indexed" in clean_line:
                    match = re.search(r'(\d+)\s+total pages indexed', clean_line)
                    if match:
                        processed_pages = int(match.group(1))
                        # Calculate progress based on processed vs total
                        if total_pages and total_pages > 0:
                            progress = min(0.95, processed_pages / total_pages)
                            task_manager.update_task_progress(
                                task_id,
                                current_step=f"Saved: {processed_pages}/{total_pages} pages",
                                progress=progress
                            )
                        else:
                            # No total_pages available, just show count
                            task_manager.update_task_progress(
                                task_id,
                                current_step=f"Saved: {processed_pages} pages",
                                progress=0.5  # Default to 50% if we can't calculate
                            )
                
                # Extract embeddings shape (indicates encoding done)
                elif "Embeddings shape:" in clean_line:
                    task_manager.update_task_progress(
                        task_id,
                        current_step="Building FAISS index...",
                        progress=0.9
                    )
                
                # Final save
                elif "Index saved to" in clean_line:
                    match = re.search(r'Total pages:\s*(\d+)', clean_line)
                    if match:
                        final_pages = int(match.group(1))
                        task_manager.update_task_progress(
                            task_id,
                            current_step=f"Complete: {final_pages} pages indexed",
                            progress=0.95
                        )
            
            process.wait()
            
            if process.returncode != 0:
                raise Exception(f"Script failed with exit code {process.returncode}")
            
            # Reset retriever to trigger lazy reload
            if "dense_vl" in self.retrievers:
                self.retrievers["dense_vl"] = None
                
        except Exception as e:
            # Re-raise the exception so TaskManager marks it as failed
            raise
        
        # Update config
        dense_vl_index_dir = Path(self.config.indices_dir) / index_name
        if dense_vl_index_dir.exists():
            self._dense_vl_config = {
                "index_dir": dense_vl_index_dir,
                "model_path": self.config.dense_vl["model_path"],
                "gpu": self.config.dense_vl.get("gpu", 1),
                "gpu_memory": self.config.dense_vl.get("gpu_memory", 0.45),
                "batch_size": self.config.dense_vl.get("batch_size", 8),
                "max_image_size": self.config.dense_vl.get("max_image_size", 1024)
            }
        
        return {
            "index_name": index_name,
            "status": "success",
            "message": "Dense-VL index built successfully"
        }

    def _run_colpali_indexing(self, task_id: str, task_manager, index_name: str, device: str, suffix: str, num_workers: int = 1):
        """Background task function for ColPali indexing."""
        from impl.index_incremental import IncrementalIndexManager
        from impl.index_colpali import ColPaliRetriever
        from impl.index_tracker import IndexTracker
        import os
        import sys
        from io import StringIO

        try:
            # Set current process as task process (for tracking)
            current_pid = os.getpid()
            task_manager.update_task_pid(task_id, current_pid)

            # Update task
            task_manager.update_task_progress(
                task_id,
                current_step=f"Starting ColPali indexing on {device} (workers={num_workers})...",
                progress=0.0
            )

            # Get initial page count for progress tracking
            index_dir = Path(self.config.indices_dir) / index_name
            if index_dir.exists():
                try:
                    tracker = IndexTracker(index_dir)
                    initial_pages = sum(
                        tracker.indexed_docs[d].get('page_count', 0)
                        for d in tracker.indexed_docs
                    )
                except:
                    initial_pages = 0
            else:
                initial_pages = 0

            # Calculate expected total
            docs = self.store.list_documents()
            total_pages = sum(doc.page_count for doc in docs)
            expected_new_pages = max(0, total_pages - initial_pages)

            print(f"[ColPali Task {task_id}] Starting index update...")
            print(f"[ColPali Task {task_id}] Workers: {num_workers}, GPU: {device}")
            print(f"[ColPali Task {task_id}] Initial pages: {initial_pages}, Expected new: {expected_new_pages}")

            # Capture stdout to parse progress
            old_stdout = sys.stdout
            sys.stdout = StringIO()

            try:
                # Initialize index manager
                index_manager = IncrementalIndexManager(self.config, self.store)

                # Update ColPali index (output will be captured)
                result = index_manager.update_colpali_index(
                    index_name=index_name
                )

            finally:
                # Get captured output
                captured_output = sys.stdout.getvalue()
                sys.stdout = old_stdout

                # Print captured output for debugging
                print(captured_output)

                # Parse progress from captured output
                # Look for "Total: X documents, Y pages to index"
                import re
                for line in captured_output.split('\n'):
                    if "Total:" in line and "documents" in line:
                        match = re.search(r'Total:\s*(\d+)\s+documents,\s*(\d+)\s+pages', line)
                        if match:
                            total_docs = int(match.group(1))
                            total_pg = int(match.group(2))
                            task_manager.update_task_progress(
                                task_id,
                                current_step=f"Processing {total_docs} documents ({total_pg} pages)...",
                                progress=0.1,
                                total_items=total_pg
                            )
                    elif "Batch" in line and "Documents" in line:
                        # Parse batch progress
                        match = re.search(r'Documents\s+(\d+)-(\d+)\s+of\s+(\d+)', line)
                        if match:
                            current_doc = int(match.group(2))
                            total_docs = int(match.group(3))
                            progress = 0.1 + 0.6 * (current_doc / total_docs)
                            task_manager.update_task_progress(
                                task_id,
                                current_step=line.strip()[:100],
                                progress=progress
                            )
                    elif "[█" in line and "% - Batch complete" in line:
                        # Parse progress bar
                        match = re.search(r'\[(█+░*)\]\s+([\d.]+)%', line)
                        if match:
                            pct = float(match.group(2))
                            progress = 0.1 + 0.6 * (pct / 100)
                            # Extract total indexed from output
                            total_match = re.search(r'Total indexed:\s*(\d+)\s+pages', captured_output)
                            if total_match:
                                indexed_pages = int(total_match.group(1))
                                task_manager.update_task_progress(
                                    task_id,
                                    current_step=f"Indexed {indexed_pages} pages...",
                                    progress=progress,
                                    processed_items=indexed_pages
                                )

            print(f"[ColPali Task {task_id}] Index update completed: {result}")

            # Update progress after indexing completes
            final_pages = result.get('total_pages', 0)
            task_manager.update_task_progress(
                task_id,
                current_step=f"Indexing complete. Processed {result.get('new_pages', 0)} pages",
                progress=0.8,
                processed_items=final_pages
            )

            if result["status"] != "success":
                raise Exception(f"Index update failed: {result.get('message', 'Unknown error')}")

            if result["status"] == "success":
                task_manager.update_task_progress(
                    task_id,
                    current_step=f"Loading retriever model...",
                    progress=0.9
                )

                # Reload retriever
                retriever = ColPaliRetriever(
                    model_name=self.config.colpali["model"],
                    device=device,
                    max_global_pool_pages=self.config.colpali.get("max_global_pool", 100),
                    max_image_size=self.config.colpali.get("max_image_size", 1024)
                )
                index_dir = Path(self.config.indices_dir) / index_name
                retriever.load_instance(index_dir)
                self.retrievers["colpali"] = retriever
                
                task_manager.update_task_progress(
                    task_id,
                    current_step=f"ColPali index ready: {result['total_pages']} pages",
                    progress=1.0
                )
                
                return {
                    "index_name": index_name,
                    "status": "success",
                    "new_docs": result['new_docs'],
                    "new_pages": result['new_pages'],
                    "total_pages": result['total_pages']
                }
            else:
                raise Exception(result.get("message", "Failed to build index"))
                
        except Exception as e:
            # Re-raise the exception so TaskManager marks it as failed
            raise

    def _handle_build_indices(
        self,
        index_type: str,
        index_name_suffix: str
    ) -> Generator[str, None, None]:
        """Handle index building (single index at a time)."""
        try:
            status = f"🔧 Building/Updating {index_type.upper()} Index...\n\n"
            suffix = index_name_suffix.strip() or "default"
            
            # Get all documents
            docs = self.store.list_documents()
            if not docs:
                yield "❌ Error: No documents found. Please ingest documents first."
                return
            
            # Filter documents for OCR variants
            if index_type in ["ocr-bm25", "ocr-dense"]:
                ocr_docs = [doc for doc in docs if getattr(doc, 'use_ocr', False)]
                status += f"📚 Found {len(docs)} document(s), {len(ocr_docs)} with OCR\n"
                if len(ocr_docs) == 0:
                    yield status + "\n❌ Error: No OCR documents found. Use non-OCR index or ingest documents with OCR enabled."
                    return
            else:
                status += f"📚 Found {len(docs)} document(s)\n"
            
            yield status
            
            # Initialize incremental index manager
            index_manager = IncrementalIndexManager(self.config, self.store)
            
            # Build the selected index type
            if index_type == "bm25":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ BM25 Index Update (All Documents)\n"
                status += "─" * 50 + "\n"
                try:
                    result = index_manager.update_bm25_index(
                        index_name=f"bm25_{suffix}"
                    )
                    
                    if result["status"] == "success":
                        # Reload retriever
                        retriever = BM25IndexerRetriever(self.store)
                        retriever.load(self.config, index_name=f"bm25_{suffix}")
                        self.retrievers["bm25"] = retriever
                        
                        status += f"✅ Success!\n"
                        status += f"   New documents: {result['new_docs']}\n"
                        status += f"   New units: {result['new_units']}\n"
                        status += f"   Total: {result['total_units']} units from {result['total_docs']} documents\n"
                    elif result["status"] == "no_update":
                        status += f"ℹ️  {result['message']}\n"
                        status += f"   All documents already indexed\n"
                    else:
                        status += f"❌ {result['message']}\n"
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            # OCR-BM25 variant
            elif index_type == "ocr-bm25":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ BM25 Index Update (OCR Documents Only)\n"
                status += "─" * 50 + "\n"
                try:
                    result = index_manager.update_bm25_index(
                        index_name="bm25_ocr",
                        filter_ocr=True
                    )
                    
                    if result["status"] == "success":
                        # Reload retriever
                        retriever = BM25IndexerRetriever(self.store)
                        retriever.load(self.config, index_name="bm25_ocr")
                        self.retrievers["ocr-bm25"] = retriever
                        
                        status += f"✅ Success!\n"
                        status += f"   New documents: {result['new_docs']}\n"
                        status += f"   New units: {result['new_units']}\n"
                        status += f"   Total: {result['total_units']} units from {result['total_docs']} documents\n"
                    elif result["status"] == "no_update":
                        status += f"ℹ️  {result['message']}\n"
                        status += f"   All OCR documents already indexed\n"
                    else:
                        status += f"❌ {result['message']}\n"
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            # Dense incremental build/update
            elif index_type == "dense":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ Dense Index Update (All Documents)\n"
                status += "─" * 50 + "\n"
                try:
                    result = index_manager.update_dense_index(
                        index_name=f"dense_{suffix}"
                    )
                    
                    if result["status"] == "success":
                        # Reload retriever with updated index
                        embedder = VLLMEmbedder(
                            endpoint=self.config.dense["endpoint"],
                            model=self.config.dense["model"],
                            batch_size=self.config.dense.get("batch_size", 32)
                        )
                        index_dir = Path(self.config.indices_dir) / f"dense_{suffix}"
                        retriever = DenseIndexerRetriever.load(index_dir, embedder)
                        self.retrievers["dense"] = retriever
                        
                        status += f"✅ Success!\n"
                        status += f"   New documents: {result['new_docs']}\n"
                        status += f"   New units: {result['new_units']}\n"
                        status += f"   Total: {result['total_units']} units\n"
                        status += f"   vLLM endpoint: {self.config.dense['endpoint']}\n"
                    elif result["status"] == "no_update":
                        status += f"ℹ️  {result['message']}\n"
                        status += f"   All documents already indexed\n"
                    else:
                        status += f"❌ {result['message']}\n"
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            # OCR-Dense variant
            elif index_type == "ocr-dense":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ Dense Index Update (OCR Documents Only)\n"
                status += "─" * 50 + "\n"
                try:
                    result = index_manager.update_dense_index(
                        index_name="dense_ocr",
                        filter_ocr=True
                    )
                    
                    if result["status"] == "success":
                        # Reload retriever with updated index
                        embedder = VLLMEmbedder(
                            endpoint=self.config.dense["endpoint"],
                            model=self.config.dense["model"],
                            batch_size=self.config.dense.get("batch_size", 32)
                        )
                        index_dir = Path(self.config.indices_dir) / "dense_ocr"
                        retriever = DenseIndexerRetriever.load(index_dir, embedder)
                        self.retrievers["ocr-dense"] = retriever
                        
                        status += f"✅ Success!\n"
                        status += f"   New documents: {result['new_docs']}\n"
                        status += f"   New units: {result['new_units']}\n"
                        status += f"   Total: {result['total_units']} units\n"
                        status += f"   vLLM endpoint: {self.config.dense['endpoint']}\n"
                    elif result["status"] == "no_update":
                        status += f"ℹ️  {result['message']}\n"
                        status += f"   All OCR documents already indexed\n"
                    else:
                        status += f"❌ {result['message']}\n"
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            # Dense-VL incremental build/update (background task)
            elif index_type == "dense_vl":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ Dense-VL Index Update (Background Task)\n"
                status += "─" * 50 + "\n"
                try:
                    import uuid
                    from impl.index_tracker import IndexTracker
                    
                    # Build index name with suffix
                    index_name = f"dense_vl_{suffix}"
                    gpu_id = self.config.dense_vl.get("gpu", 1)
                    
                    status += f"📝 Index name: {index_name}\n"
                    status += f"📍 Using GPU: {gpu_id}\n\n"
                    
                    # Check current index status
                    index_dir = Path(self.config.indices_dir) / index_name
                    tracker = IndexTracker(index_dir)
                    indexed_docs = tracker.get_indexed_docs()
                    
                    # Get all documents and calculate what needs processing
                    docs = self.store.list_documents()
                    total_docs = len(docs)
                    total_pages = sum(doc.page_count for doc in docs)
                    
                    already_indexed_count = len(indexed_docs)
                    to_index_count = total_docs - already_indexed_count
                    
                    # Calculate pages
                    indexed_pages = sum(
                        tracker.indexed_docs[doc.doc_id].get('page_count', 0)
                        for doc in docs if doc.doc_id in indexed_docs
                    )
                    pages_to_process = total_pages - indexed_pages
                    
                    # Display statistics
                    status += f"📊 Current Status:\n"
                    status += f"   Total documents: {total_docs} ({total_pages:,} pages)\n"
                    status += f"   Already indexed: {already_indexed_count} docs ({indexed_pages:,} pages) ✓\n"
                    status += f"   To be processed: {to_index_count} docs ({pages_to_process:,} pages)\n\n"
                    
                    if to_index_count == 0:
                        status += "✅ All documents already indexed!\n"
                        status += "   No new documents to process.\n"
                        yield status
                        return
                    
                    # Submit as background task
                    task_id = f"index_dense_vl_{uuid.uuid4().hex[:8]}"
                    
                    task = self.task_manager.submit_task(
                        task_id=task_id,
                        task_type="index_dense_vl",
                        func=self._run_dense_vl_indexing,
                        args=(index_name, gpu_id),
                        kwargs={},
                        total_items=total_pages,  # Use total for progress calculation
                        description=f"Building Dense-VL index: {index_name}"
                    )
                    
                    status += f"✅ Task submitted: {task_id}\n"
                    status += f"   Check 'Background Tasks' tab for progress\n\n"
                    status += f"💡 Processing {pages_to_process:,} pages...\n"
                    status += f"   You can close this window and check progress later\n"
                    
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            # ColPali incremental build/update (background task)
            elif index_type == "colpali":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ ColPali Index Update (Background Task)\n"
                status += "─" * 50 + "\n"
                try:
                    import uuid
                    from impl.index_tracker import IndexTracker
                    
                    # Build index name with suffix
                    index_name = f"colpali_{suffix}"
                    gpu_id = self.config.colpali.get("gpu", 0)
                    device = f"cuda:{gpu_id}"
                    
                    status += f"📝 Index name: {index_name}\n"
                    status += f"📍 Using GPU: {gpu_id} ({device})\n\n"
                    
                    # Check current index status
                    index_dir = Path(self.config.indices_dir) / index_name
                    tracker = IndexTracker(index_dir)
                    indexed_docs = tracker.get_indexed_docs()
                    
                    # Get all documents and calculate what needs processing
                    docs = self.store.list_documents()
                    total_docs = len(docs)
                    total_pages = sum(doc.page_count for doc in docs)
                    
                    already_indexed_count = len(indexed_docs)
                    to_index_count = total_docs - already_indexed_count
                    
                    # Calculate pages
                    indexed_pages = sum(
                        tracker.indexed_docs[doc.doc_id].get('page_count', 0)
                        for doc in docs if doc.doc_id in indexed_docs
                    )
                    pages_to_process = total_pages - indexed_pages
                    
                    # Display statistics
                    status += f"📊 Current Status:\n"
                    status += f"   Total documents: {total_docs} ({total_pages:,} pages)\n"
                    status += f"   Already indexed: {already_indexed_count} docs ({indexed_pages:,} pages) ✓\n"
                    status += f"   To be processed: {to_index_count} docs ({pages_to_process:,} pages)\n\n"
                    
                    if to_index_count == 0:
                        status += "✅ All documents already indexed!\n"
                        status += "   No new documents to process.\n"
                        yield status
                        return
                    
                    # Submit as background task
                    task_id = f"index_colpali_{uuid.uuid4().hex[:8]}"
                    
                    # Get num_workers from config
                    num_workers = self.config.colpali.get("num_workers", 1)
                    
                    task = self.task_manager.submit_task(
                        task_id=task_id,
                        task_type="index_colpali",
                        func=self._run_colpali_indexing,
                        args=(index_name, device, suffix, num_workers),
                        kwargs={},
                        total_items=total_pages,  # Use total for progress calculation
                        description=f"Building ColPali index: {index_name}"
                    )
                    
                    status += f"✅ Task submitted: {task_id}\n"
                    status += f"   Workers: {num_workers}\n"
                    status += f"   Check 'Background Tasks' tab for progress\n\n"
                    status += f"💡 Processing {pages_to_process:,} pages...\n"
                    status += f"   You can close this window and check progress later\n"
                    status += f"   You can close this window and check progress later\n"
                    
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            status += "\n" + "=" * 50 + "\n"
            status += "🎉 Index Building Complete!\n"
            status += "=" * 50 + "\n"
            status += f"\nAvailable retrieval modes: {list(self.retrievers.keys())}\n"
            status += "\nℹ️  Incremental indexing:\n"
            status += "   • Only new documents are indexed\n"
            status += "   • Existing indices are preserved and updated\n"
            status += "   • No need to rebuild everything when adding docs\n"
            status += "\nYou can now use the 'Query & Answer' tab.\n"
            
            yield status
            
        except Exception as e:
            import traceback
            yield f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

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
        retrieval_mode: str,
        evidence_mode: str = "text",
        fusion_method: str = "rrf",
        retriever_1: str = "bm25",
        retriever_2: str = "dense",
        weight_1: float = 0.5
    ) -> Tuple[str, List, str, List]:
        """Handle query with selected retrieval mode and evidence format."""
        try:
            if not question:
                return "Please enter a question", [], "", []
            
            # Handle Hybrid mode - dynamically create retriever
            if retrieval_mode == "hybrid":
                from impl.retriever_hybrid import HybridRetriever
                
                # Validate: two different retrievers
                if retriever_1 == retriever_2:
                    return f"⚠️ 请选择两个不同的检索器！当前都选择了 {retriever_1}", [], "", [], []
                
                # Load retrievers (including lazy-loaded ColPali and Dense-VL)
                retriever_objs = {}
                for ret_name in [retriever_1, retriever_2]:
                    if ret_name == "colpali" and self.retrievers.get("colpali") is None:
                        # Lazy load ColPali on first use
                        if hasattr(self, "_colpali_config"):
                            try:
                                print(f"⏳ 首次使用 ColPali，正在加载模型...")
                                colpali_retriever = ColPaliRetriever.load(
                                    self._colpali_config["index_dir"],
                                    model_name=self._colpali_config["model_name"],
                                    device=self._colpali_config["device"],
                                    max_image_size=self._colpali_config.get("max_image_size", 1024)
                                )
                                self.retrievers["colpali"] = colpali_retriever
                                print(f"✅ ColPali 模型加载完成")
                            except Exception as e:
                                return f"❌ ColPali 加载失败: {e}", [], "", []
                        else:
                            return f"❌ ColPali 未配置", [], "", []
                    
                    if ret_name == "dense_vl" and self.retrievers.get("dense_vl") is None:
                        # Lazy load Dense-VL on first use
                        if self._dense_vl_config:
                            try:
                                print(f"⏳ 首次使用 Dense-VL，正在加载模型...")
                                from impl.index_dense_vl import DenseVLRetrieverLazy
                                dense_vl_retriever = DenseVLRetrieverLazy.load(
                                    index_dir=self._dense_vl_config["index_dir"],
                                    model_path=self._dense_vl_config["model_path"],
                                    gpu=self._dense_vl_config["gpu"],
                                    gpu_memory=self._dense_vl_config["gpu_memory"],
                                    batch_size=self._dense_vl_config["batch_size"],
                                    max_image_size=self._dense_vl_config.get("max_image_size", 1024)
                                )
                                self.retrievers["dense_vl"] = dense_vl_retriever
                                print(f"✅ Dense-VL 模型加载完成")
                            except Exception as e:
                                return f"❌ Dense-VL 加载失败: {e}", [], "", []
                        else:
                            return f"❌ Dense-VL 未配置或索引不存在", [], "", []
                    
                    if ret_name not in self.retrievers or self.retrievers[ret_name] is None:
                        return f"⚠️ 检索器 '{ret_name}' 未找到，请先构建索引", [], "", []
                    
                    retriever_objs[ret_name] = self.retrievers[ret_name]
                
                # Calculate weights
                weight_2 = 1.0 - weight_1
                weights = {retriever_1: weight_1, retriever_2: weight_2}
                
                # Create Hybrid retriever
                retriever = HybridRetriever(
                    retrievers=retriever_objs,
                    weights=weights,
                    fusion_method=fusion_method
                )
                
                print(f"🔄 Hybrid: {retriever_1}({weight_1:.2f}) + {retriever_2}({weight_2:.2f}), fusion={fusion_method}")
            
            # Handle single retrieval modes
            else:
                retriever = self.retrievers.get(retrieval_mode)
                
                # Lazy load ColPali if needed
                if retrieval_mode == "colpali" and retriever is None:
                    if hasattr(self, "_colpali_config"):
                        try:
                            print(f"⏳ 首次使用 ColPali，正在加载模型...")
                            retriever = ColPaliRetriever.load(
                                self._colpali_config["index_dir"],
                                model_name=self._colpali_config["model_name"],
                                device=self._colpali_config["device"],
                                max_image_size=self._colpali_config.get("max_image_size", 1024)
                            )
                            self.retrievers["colpali"] = retriever
                            print(f"✅ ColPali 模型加载完成")
                        except Exception as e:
                            return f"❌ ColPali 加载失败: {e}", [], "", []
                    else:
                        return "❌ ColPali 未配置", [], "", []
                
                # Lazy load Dense-VL if needed
                if retrieval_mode == "dense_vl" and retriever is None:
                    if self._dense_vl_config:
                        try:
                            print(f"⏳ 首次使用 Dense-VL，正在加载模型...")
                            from impl.index_dense_vl import DenseVLRetrieverLazy
                            retriever = DenseVLRetrieverLazy.load(
                                index_dir=self._dense_vl_config["index_dir"],
                                model_path=self._dense_vl_config["model_path"],
                                gpu=self._dense_vl_config["gpu"],
                                gpu_memory=self._dense_vl_config["gpu_memory"],
                                batch_size=self._dense_vl_config["batch_size"],
                                max_image_size=self._dense_vl_config.get("max_image_size", 1024)
                            )
                            self.retrievers["dense_vl"] = retriever
                            print(f"✅ Dense-VL 模型加载完成")
                        except Exception as e:
                            return f"❌ Dense-VL 加载失败: {e}", [], "", []
                    else:
                        return "❌ Dense-VL 未配置或索引不存在", [], "", []
                
                if retriever is None:
                    return f"❌ 检索器 '{retrieval_mode}' 不可用，请先构建索引", [], "", []

            
            # Switch generator based on evidence mode
            if evidence_mode == "image":
                # Use image-based generator
                try:
                    from impl.generator_qwen_vl import QwenVLGenerator
                    generator = QwenVLGenerator(self.config, use_images=True, store=self.store)
                    print(f"🖼️  Using image-based generation")
                except Exception as e:
                    return f"Failed to load image generator: {e}", [], "", [], []
            else:
                # Use existing text-based generator (or create new one with store)
                try:
                    from impl.generator_qwen_vl import QwenVLGenerator
                    generator = QwenVLGenerator(self.config, use_images=False, store=self.store)
                    print(f"📝 Using text-based generation with context assembly")
                except Exception as e:
                    # Fallback to original generator
                    generator = self.generator
                    print(f"📝 Using text-based generation")
            
            # Create temporary pipeline with selected generator
            from core.pipeline import Pipeline
            pipeline = Pipeline(
                retriever=retriever,
                selector=self.selector,
                generator=generator,
                logger=self.logger,
                store=self.store
            )
            
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
            result = pipeline.answer(query_input, self.config)
            
            # Format evidence table
            evidence_rows = []
            if result.evidence and result.evidence.evidence:
                for i, ev in enumerate(result.evidence.evidence, 1):
                    # For image mode, show page info instead of snippet
                    if evidence_mode == "image":
                        snippet_text = f"[Image Mode] Page {ev.page_id}"
                    else:
                        snippet_text = ev.snippet[:100] + "..." if len(ev.snippet) > 100 else ev.snippet
                    
                    evidence_rows.append([
                        i,
                        f"{retrieval_mode} + {evidence_mode}",  # Show both modes
                        ev.doc_id,
                        ev.page_id,
                        f"{ev.score:.4f}",
                        snippet_text
                    ])
            
            answer = result.generation.output.answer if result.generation else "No answer generated."
            
            # Debug: check what's in the result
            if not result.generation:
                print(f"⚠️  Warning: result.generation is None")
            elif not result.generation.output:
                print(f"⚠️  Warning: result.generation.output is None")
            elif not result.generation.output.answer:
                print(f"⚠️  Warning: answer is empty")
            else:
                print(f"✅ Generated answer: {answer[:100]}...")
            
            # Load page images if in image mode
            page_image_paths = []
            if evidence_mode == "image" and result.evidence and result.evidence.evidence:
                for ev in result.evidence.evidence:
                    # Load page artifact to get image path
                    artifact = self.store.load_page_artifact(ev.doc_id, ev.page_id)
                    if artifact and artifact.image_path:
                        image_path = Path(artifact.image_path)
                        if image_path.exists():
                            page_image_paths.append((str(image_path), f"{ev.doc_id} - Page {ev.page_id}"))
                        else:
                            print(f"⚠️ Image not found: {image_path}")
                    else:
                        print(f"⚠️ No image for {ev.doc_id} page {ev.page_id}")
            
            return answer, evidence_rows, query_input.query_id, page_image_paths
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return error_msg, [], "", []

    def _handle_eval(
        self,
        eval_file,
        eval_mode: str,
        eval_evidence_mode: str = "text",
        enable_vl_scoring: bool = True,
        fusion_method: str = "rrf",
        retriever_1: str = "bm25",
        retriever_2: str = "dense",
        weight_1: float = 0.5
    ) -> Tuple[str, dict, Optional[str], Optional[str]]:
        """Handle batch evaluation with custom hybrid configuration and VL scoring."""
        try:
            if eval_file is None:
                return "Error: No evaluation file uploaded", {}, None, None
            
            # Handle Hybrid mode - dynamically create retriever
            if eval_mode == "hybrid":
                from impl.retriever_hybrid import HybridRetriever
                
                # Validate: two different retrievers
                if retriever_1 == retriever_2:
                    return f"⚠️ 请选择两个不同的检索器！当前都选择了 {retriever_1}", {}, None, None
                
                # Ensure retrievers are loaded (including lazy ColPali and Dense-VL)
                retriever_objs = {}
                for ret_name in [retriever_1, retriever_2]:
                    if ret_name == "colpali" and self.retrievers.get("colpali") is None:
                        if hasattr(self, "_colpali_config"):
                            try:
                                print(f"⏳ Loading ColPali for evaluation...")
                                from impl.index_colpali import ColPaliRetriever
                                colpali_retriever = ColPaliRetriever.load(
                                    self._colpali_config["index_dir"],
                                    model_name=self._colpali_config["model_name"],
                                    device=self._colpali_config["device"],
                                    max_image_size=self._colpali_config.get("max_image_size", 1024)
                                )
                                self.retrievers["colpali"] = colpali_retriever
                                print(f"✅ ColPali loaded")
                            except Exception as e:
                                return f"❌ ColPali 加载失败: {e}", {}, None, None
                        else:
                            return f"❌ ColPali 未配置", {}, None, None
                    
                    if ret_name == "dense_vl" and self.retrievers.get("dense_vl") is None:
                        if self._dense_vl_config:
                            try:
                                print(f"⏳ Loading Dense-VL for evaluation...")
                                from impl.index_dense_vl import DenseVLRetrieverLazy
                                dense_vl_retriever = DenseVLRetrieverLazy.load(
                                    index_dir=self._dense_vl_config["index_dir"],
                                    model_path=self._dense_vl_config["model_path"],
                                    gpu=self._dense_vl_config["gpu"],
                                    gpu_memory=self._dense_vl_config["gpu_memory"],
                                    batch_size=self._dense_vl_config["batch_size"],
                                    max_image_size=self._dense_vl_config.get("max_image_size", 1024)
                                )
                                self.retrievers["dense_vl"] = dense_vl_retriever
                                print(f"✅ Dense-VL loaded")
                            except Exception as e:
                                return f"❌ Dense-VL 加载失败: {e}", {}, None, None
                        else:
                            return f"❌ Dense-VL 未配置或索引不存在", {}, None, None
                    
                    if ret_name not in self.retrievers or self.retrievers[ret_name] is None:
                        return f"⚠️ 检索器 '{ret_name}' 未找到，请先构建索引", {}, None, None
                    
                    retriever_objs[ret_name] = self.retrievers[ret_name]
                
                # Calculate weights
                weight_2 = 1.0 - weight_1
                weights = {retriever_1: weight_1, retriever_2: weight_2}
                
                # Create hybrid retriever
                retriever = HybridRetriever(
                    retrievers=retriever_objs,
                    weights=weights,
                    fusion_method=fusion_method
                )
                
                print(f"📊 Evaluation Hybrid: {retriever_1}({weight_1:.2f}) + {retriever_2}({weight_2:.2f}), fusion={fusion_method}")
            
            # Handle single retrieval modes
            else:
                retriever = self.retrievers.get(eval_mode)
                
                # Lazy load ColPali if needed (same as in _handle_query)
                if eval_mode == "colpali" and retriever is None:
                    if hasattr(self, "_colpali_config"):
                        try:
                            print(f"⏳ 首次使用 ColPali (评估模式)，正在加载模型...")
                            from impl.index_colpali import ColPaliRetriever
                            retriever = ColPaliRetriever.load(
                                self._colpali_config["index_dir"],
                                model_name=self._colpali_config["model_name"],
                                device=self._colpali_config["device"],
                                max_image_size=self._colpali_config.get("max_image_size", 1024)
                            )
                            self.retrievers["colpali"] = retriever
                            print(f"✅ ColPali 模型加载完成")
                        except Exception as e:
                            return f"❌ ColPali 加载失败: {e}", {}, None, None
                    else:
                        return "❌ ColPali 未配置", {}, None, None
                
                # Lazy load Dense-VL if needed
                if eval_mode == "dense_vl" and retriever is None:
                    if self._dense_vl_config:
                        try:
                            print(f"⏳ 首次使用 Dense-VL (评估模式)，正在加载模型...")
                            from impl.index_dense_vl import DenseVLRetrieverLazy
                            retriever = DenseVLRetrieverLazy.load(
                                index_dir=self._dense_vl_config["index_dir"],
                                model_path=self._dense_vl_config["model_path"],
                                gpu=self._dense_vl_config["gpu"],
                                gpu_memory=self._dense_vl_config["gpu_memory"],
                                batch_size=self._dense_vl_config["batch_size"],
                                max_image_size=self._dense_vl_config.get("max_image_size", 1024)
                            )
                            self.retrievers["dense_vl"] = retriever
                            print(f"✅ Dense-VL 模型加载完成")
                        except Exception as e:
                            return f"❌ Dense-VL 加载失败: {e}", {}, None, None
                    else:
                        return "❌ Dense-VL 未配置或索引不存在", {}, None, None
                
                if retriever is None:
                    return f"❌ 检索器 '{eval_mode}' 不可用，请先构建索引", {}, None, None
            
            # Switch generator based on evidence mode
            if eval_evidence_mode == "image":
                try:
                    from impl.generator_qwen_vl import QwenVLGenerator
                    generator = QwenVLGenerator(self.config, use_images=True, store=self.store)
                    print(f"🖼️ Evaluation using image-based generation")
                except Exception as e:
                    return f"Failed to load image generator: {e}", {}, None, None
            else:
                try:
                    from impl.generator_qwen_vl import QwenVLGenerator
                    generator = QwenVLGenerator(self.config, use_images=False, store=self.store)
                    print(f"📝 Evaluation using text-based generation")
                except Exception as e:
                    generator = self.generator
                    print(f"📝 Evaluation using default text-based generation")
            
            # Update pipeline with selected retriever and generator
            self.pipeline.retriever = retriever
            self.pipeline.generator = generator
            
            # Create eval runner with VL scoring if enabled
            from impl.eval_runner import EvalRunner, load_dataset_from_csv, load_dataset_from_json
            
            eval_runner = EvalRunner(
                pipeline=self.pipeline,
                enable_metrics=True,
                enable_vl_scoring=enable_vl_scoring,
                config=self.config
            )
            
            if eval_file.name.endswith('.csv'):
                dataset = load_dataset_from_csv(eval_file.name)
            else:
                dataset = load_dataset_from_json(eval_file.name)
            
            # Run evaluation (use 'evaluate' method, not 'run')
            report = eval_runner.evaluate(dataset, self.config)
            
            # Extract artifact paths from report
            csv_path = report.artifact_paths.get("predictions_csv")
            json_path = report.artifact_paths.get("report")
            
            status = f"✅ Evaluation complete\n"
            status += f"Retrieval Mode: {eval_mode}\n"
            status += f"Evidence Mode: {eval_evidence_mode}\n"
            if enable_vl_scoring:
                status += f"VL Scoring: Enabled\n"
            if "hybrid" in eval_mode:
                weight_2 = 1.0 - weight_1
                status += f"Hybrid Config: {retriever_1}({weight_1:.2f}) + {retriever_2}({weight_2:.2f}), {fusion_method}\n"
            status += f"Samples: {len(dataset.items)}\n"
            
            # Add VL metrics to status if available
            if enable_vl_scoring and "vl_avg_score" in report.metrics.extra:
                status += f"\n📊 VL Evaluation:\n"
                status += f"  Average Score: {report.metrics.extra['vl_avg_score']:.2f}/10\n"
                status += f"  Correct: {report.metrics.extra['vl_correct_rate']:.1%}\n"
                status += f"  Partial: {report.metrics.extra['vl_partial_rate']:.1%}\n"
            
            status += f"\nResults saved to: {csv_path}"
            
            return status, report.metrics.extra, csv_path, json_path
            
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
        """Get list of documents (legacy, for compatibility)."""
        try:
            docs = self.store.list_documents()
            
            # Determine available index types (include ocr variants)
            available_indices = []
            if "bm25" in self.retrievers:
                available_indices.append("bm25")
            if "ocr-bm25" in self.retrievers:
                available_indices.append("ocr-bm25")
            if "dense" in self.retrievers or self.config.dense.get("enabled"):
                available_indices.append("dense")
            if "ocr-dense" in self.retrievers or self.config.dense.get("enabled"):
                available_indices.append("ocr-dense")
            if "colpali" in self.retrievers or hasattr(self, "_colpali_config"):
                available_indices.append("colpali")
            if "dense_vl" in self.retrievers or self._dense_vl_config:
                available_indices.append("dense_vl")
            
            rows = []
            for meta in docs:
                # OCR status
                ocr_status = "✓" if getattr(meta, 'use_ocr', False) else "✗"
                
                # Check index status
                index_status = []
                for idx_type in available_indices:
                    status = self._check_index_completeness(meta.doc_id, meta.page_count, idx_type)
                    index_status.append(f"{idx_type.upper()}:{status}")
                
                status_str = " | ".join(index_status) if index_status else "No indices"
                
                rows.append([
                    meta.doc_id,
                    meta.title,
                    meta.page_count,
                    ocr_status,
                    status_str,
                    meta.created_at
                ])
            return rows
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def _check_index_completeness(self, doc_id: str, page_count: int, index_type: str) -> str:
        """
        Check if a document's index is complete.
        
        Args:
            doc_id: Document ID
            page_count: Total pages in document
            index_type: Type of index (bm25, ocr-bm25, dense, ocr-dense, colpali, dense_vl)
            
        Returns:
            Status string: "✓" (complete), "✗" (missing), "⚠" (incomplete)
        """
        try:
            # Determine index directory based on type
            if index_type == "bm25":
                index_dir = Path(self.config.indices_dir) / "bm25_default"
            elif index_type == "ocr-bm25":
                index_dir = Path(self.config.indices_dir) / "bm25_ocr"
            elif index_type == "dense":
                index_dir = Path(self.config.indices_dir) / "dense_default"
            elif index_type == "ocr-dense":
                index_dir = Path(self.config.indices_dir) / "dense_ocr"
            elif index_type == "colpali":
                index_dir = Path(self.config.indices_dir) / "colpali_default"
            elif index_type == "dense_vl":
                index_dir = Path(self.config.indices_dir) / "dense_vl_default"
            else:
                return "✗"
            
            # Check if index exists
            if not index_dir.exists():
                return "✗"
            
            # Load tracker to check document status
            from impl.index_tracker import IndexTracker
            tracker = IndexTracker(index_dir)
            
            if doc_id not in tracker.indexed_docs:
                return "✗"
            
            # Check if page count matches (all pages indexed)
            indexed_info = tracker.indexed_docs[doc_id]
            indexed_page_count = indexed_info.get("page_count", 0)
            
            if indexed_page_count == page_count:
                return "✓"
            else:
                return "⚠"  # Incomplete - not all pages indexed
                
        except Exception as e:
            return "✗"
    
    def _get_doc_list_paginated(self, page: int, page_size: int) -> Tuple[str, List, str]:
        """Get paginated list of uploaded files with ingestion status."""
        try:
            from datetime import datetime
            import os
            
            # Get all PDF files from uploads directory
            uploads_dir = Path("data/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            
            pdf_files = list(uploads_dir.glob("*.pdf")) + list(uploads_dir.glob("*.PDF"))
            pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Sort by upload time
            
            total = len(pdf_files)
            
            # Get ingested documents
            ingested_docs = {doc.doc_id: doc for doc in self.store.list_documents()}
            
            # Calculate pagination
            page = max(1, int(page))
            page_size = int(page_size)
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total)
            
            # Get page slice
            page_files = pdf_files[start_idx:end_idx]
            
            # Determine available index types
            available_indices = []
            if "bm25" in self.retrievers:
                available_indices.append("bm25")
            if "ocr-bm25" in self.retrievers:
                available_indices.append("ocr-bm25")
            if "dense" in self.retrievers or self.config.dense.get("enabled"):
                available_indices.append("dense")
            if "ocr-dense" in self.retrievers or self.config.dense.get("enabled"):
                available_indices.append("ocr-dense")
            if "colpali" in self.retrievers or hasattr(self, "_colpali_config"):
                available_indices.append("colpali")
            if "dense_vl" in self.retrievers or self._dense_vl_config:
                available_indices.append("dense_vl")
            
            # Build rows
            rows = []
            for pdf_file in page_files:
                filename = pdf_file.name
                doc_id = pdf_file.stem.replace(" ", "_").lower()
                
                # Check if ingested
                if doc_id in ingested_docs:
                    meta = ingested_docs[doc_id]
                    ingested_status = "✓"
                    pages = meta.page_count
                    ocr_status = "✓" if getattr(meta, 'use_ocr', False) else "✗"
                    
                    # Check index status
                    index_status = []
                    for idx_type in available_indices:
                        status = self._check_index_completeness(doc_id, pages, idx_type)
                        index_status.append(f"{idx_type.upper()}:{status}")
                    status_str = " | ".join(index_status) if index_status else "-"
                else:
                    ingested_status = "✗"
                    pages = "-"
                    ocr_status = "-"
                    status_str = "Not ingested"
                
                # Get upload time
                upload_time = datetime.fromtimestamp(pdf_file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                
                rows.append([
                    filename,
                    doc_id,
                    ingested_status,
                    pages,
                    ocr_status,
                    status_str,
                    upload_time
                ])
            
            # Statistics with index completion info for all index types
            ingested_count = len(ingested_docs)
            not_ingested = total - ingested_count
            
            # Count index completeness for all available index types
            from impl.index_tracker import IndexTracker
            index_stats = {}
            
            for idx_type in available_indices:
                # Map index type to directory name
                if idx_type == "bm25":
                    index_dir = Path(self.config.indices_dir) / "bm25_default"
                elif idx_type == "ocr-bm25":
                    index_dir = Path(self.config.indices_dir) / "bm25_ocr"
                elif idx_type == "dense":
                    index_dir = Path(self.config.indices_dir) / "dense_default"
                elif idx_type == "ocr-dense":
                    index_dir = Path(self.config.indices_dir) / "dense_ocr"
                elif idx_type == "colpali":
                    index_dir = Path(self.config.indices_dir) / "colpali_default"
                elif idx_type == "dense_vl":
                    index_dir = Path(self.config.indices_dir) / "dense_vl_default"
                else:
                    continue
                
                if index_dir.exists():
                    try:
                        tracker = IndexTracker(index_dir)
                        complete = 0
                        for doc_id, doc in ingested_docs.items():
                            if doc_id in tracker.indexed_docs:
                                indexed_info = tracker.indexed_docs[doc_id]
                                if indexed_info.get('page_count', 0) == doc.page_count:
                                    complete += 1
                        index_stats[idx_type] = (complete, ingested_count)
                    except:
                        pass
            
            stats = f"**Total Files**: {total} | **Ingested**: {ingested_count} | **Not Ingested**: {not_ingested}"
            
            # Add index statistics
            if index_stats:
                stats += "\n\n**Index Completion**: "
                index_parts = []
                for idx_type in available_indices:
                    if idx_type in index_stats:
                        complete, total_docs = index_stats[idx_type]
                        pct = (complete / total_docs * 100) if total_docs > 0 else 0
                        index_parts.append(f"{idx_type.upper()}: {complete}/{total_docs} ({pct:.0f}%)")
                stats += " | ".join(index_parts)
            
            # Page info
            if total == 0:
                page_info = "No files in uploads folder"
            else:
                page_info = f"Showing {start_idx + 1}-{end_idx} of {total} (Page {page})"
            
            return stats, rows, page_info
            
        except Exception as e:
            print(f"Error listing files: {e}")
            import traceback
            traceback.print_exc()
            return "**Total Files**: 0", [], "Error loading files"
    
    def _handle_file_selection(self, files):
        """Handle file selection and display file info with page counts."""
        if not files:
            return [], gr.update(choices=[], value=None)
        
        if not isinstance(files, list):
            files = [files]
        
        rows = []
        filenames = []
        for f in files:
            try:
                import pdfplumber
                filename = Path(f.name).name
                filenames.append(filename)
                size_mb = Path(f.name).stat().st_size / (1024 * 1024)
                
                # Get page count
                with pdfplumber.open(f.name) as pdf:
                    page_count = len(pdf.pages)
                
                rows.append([filename, f"{size_mb:.2f}", page_count])
            except Exception as e:
                filename = Path(f.name).name
                filenames.append(filename)
                rows.append([filename, "N/A", "Error"])
        
        # Update file selector dropdown
        return rows, gr.update(choices=filenames, value=filenames[0] if filenames else None)
    
    def _handle_preview_pdf(self, files, selected_file, page_num: int, zoom: float):
        """Preview a page from selected PDF and open accordion."""
        if not files:
            return None, gr.update(), "No file", gr.update(open=False)
        
        if not isinstance(files, list):
            files = [files]
        
        if len(files) == 0:
            return None, gr.update(), "No file", gr.update(open=False)
        
        try:
            import fitz  # PyMuPDF
            from PIL import Image
            
            # Find the selected file
            pdf_path = None
            if selected_file:
                for f in files:
                    if Path(f.name).name == selected_file:
                        pdf_path = f.name
                        break
            
            if not pdf_path:
                pdf_path = files[0].name
            
            doc = fitz.open(pdf_path)
            max_pages = len(doc)
            
            # Clamp page number
            page_num = max(1, min(int(page_num), max_pages))
            
            # Render page with zoom (maintain aspect ratio)
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            
            # Convert to PIL Image to ensure proper aspect ratio handling
            img_data = pix.tobytes("png")
            import io
            img = Image.open(io.BytesIO(img_data))
            
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name, format="PNG")
                preview_path = tmp.name
            
            doc.close()
            
            page_info = f"**Page {page_num} / {max_pages}**"
            
            # Update slider and open accordion
            return preview_path, gr.update(maximum=max_pages, value=page_num), page_info, gr.update(open=True)
            
        except Exception as e:
            print(f"Preview error: {e}")
            return None, gr.update(), "Error loading preview", gr.update(open=False)
    
    def _handle_preview_update(self, files, selected_file, page_num: int, zoom: float):
        """Update preview when page or zoom changes."""
        if not files:
            return None, "No file"
        
        try:
            import fitz
            from PIL import Image
            
            if not isinstance(files, list):
                files = [files]
            
            # Find the selected file
            pdf_path = None
            if selected_file:
                for f in files:
                    if Path(f.name).name == selected_file:
                        pdf_path = f.name
                        break
            
            if not pdf_path:
                pdf_path = files[0].name
            
            doc = fitz.open(pdf_path)
            max_pages = len(doc)
            page_num = max(1, min(int(page_num), max_pages))
            
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            
            # Convert to PIL Image to maintain aspect ratio
            img_data = pix.tobytes("png")
            import io
            img = Image.open(io.BytesIO(img_data))
            
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp.name, format="PNG")
                preview_path = tmp.name
            
            doc.close()
            
            page_info = f"**Page {page_num} / {max_pages}**"
            return preview_path, page_info
            
        except Exception as e:
            print(f"Preview update error: {e}")
            return None, "Error"
    
    def _handle_prev_page(self, files, selected_file, page_num: int, zoom: float):
        """Go to previous page."""
        new_page = max(1, int(page_num) - 1)
        img, page_info = self._handle_preview_update(files, selected_file, new_page, zoom)
        return img, gr.update(value=new_page), page_info
    
    def _handle_next_page(self, files, selected_file, page_num: int, zoom: float):
        """Go to next page."""
        new_page = int(page_num) + 1
        img, page_info = self._handle_preview_update(files, selected_file, new_page, zoom)
        return img, gr.update(value=new_page), page_info
    
    def _handle_file_selector_change(self, files, selected_file, zoom: float):
        """Handle file selector change - reset to page 1."""
        preview_path, page_info = self._handle_preview_update(files, selected_file, 1, zoom)
        
        # Get max pages for the new file
        if files and selected_file:
            try:
                import fitz
                if not isinstance(files, list):
                    files = [files]
                
                for f in files:
                    if Path(f.name).name == selected_file:
                        doc = fitz.open(f.name)
                        max_pages = len(doc)
                        doc.close()
                        return preview_path, gr.update(maximum=max_pages, value=1), page_info
            except:
                pass
        
        return preview_path, gr.update(value=1), page_info
    
    def _handle_refresh_docs_paginated(self, page: int, page_size: int) -> Tuple[str, List, str]:
        """Refresh document list with pagination."""
        return self._get_doc_list_paginated(page, page_size)

    def _handle_delete_doc(self, doc_id: str, page: int = 1, page_size: int = 20) -> Tuple[str, str, List, str]:
        """Delete a document and return updated paginated list."""
        try:
            if not doc_id:
                stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
                return "Error: No doc_id provided", stats, rows, page_info
            
            self.store.delete_document(doc_id)
            stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
            return f"✅ Deleted: {doc_id}", stats, rows, page_info
            
        except Exception as e:
            stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
            return f"Error: {str(e)}", stats, rows, page_info
    
    def _handle_ocr_selected(self, doc_ids_str: str, page: int = 1, page_size: int = 20) -> Tuple[str, str, List, str]:
        """Re-process selected documents with OCR."""
        try:
            if not doc_ids_str or not doc_ids_str.strip():
                stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
                return "❌ Error: No document IDs provided", stats, rows, page_info
            
            # Parse doc IDs
            doc_ids = [d.strip() for d in doc_ids_str.split(",") if d.strip()]
            
            if not doc_ids:
                stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
                return "❌ Error: No valid document IDs", stats, rows, page_info
            
            status = f"🔍 OCR Processing {len(doc_ids)} document(s)...\n"
            status += "=" * 50 + "\n\n"
            
            success_count = 0
            failed_count = 0
            skipped_count = 0
            
            for doc_id in doc_ids:
                try:
                    # Get existing document metadata
                    existing_meta = self.store.get_document(doc_id)
                    if not existing_meta:
                        status += f"❌ {doc_id}: Not found\n"
                        failed_count += 1
                        continue
                    
                    # Check if already OCR processed
                    if getattr(existing_meta, 'use_ocr', False):
                        status += f"⏭️  {doc_id}: Already OCR processed, skipping\n"
                        skipped_count += 1
                        continue
                    
                    # Re-ingest with OCR
                    from impl.ingest_pdf_v1 import PDFIngestorV1
                    ingestor = PDFIngestorV1(
                        config=self.config,
                        store=self.store,
                        use_ocr=True
                    )
                    
                    status += f"⏳ Processing {doc_id}...\n"
                    meta = ingestor.ingest(existing_meta.source_path, doc_id=doc_id)
                    status += f"✅ {doc_id}: OCR completed ({meta.page_count} pages)\n"
                    success_count += 1
                    
                except Exception as e:
                    status += f"❌ {doc_id}: {str(e)}\n"
                    failed_count += 1
            
            status += "\n" + "=" * 50 + "\n"
            status += f"📊 Summary: ✅ {success_count} succeeded | ⏭️ {skipped_count} skipped | ❌ {failed_count} failed\n"
            if success_count > 0:
                status += "💡 Next: Rebuild indices to use OCR text"
            
            stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
            return status, stats, rows, page_info
            
        except Exception as e:
            import traceback
            stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
            return f"❌ Error: {str(e)}\n{traceback.format_exc()}", stats, rows, page_info
    
    def _handle_ocr_all_non_ocr(self, page: int = 1, page_size: int = 20) -> Tuple[str, str, List, str]:
        """Process all non-OCR documents with OCR."""
        try:
            # Find all documents without OCR
            all_docs = self.store.list_documents()
            non_ocr_docs = [doc for doc in all_docs if not getattr(doc, 'use_ocr', False)]
            
            if not non_ocr_docs:
                stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
                return "ℹ️  No non-OCR documents found. All documents already processed with OCR.", stats, rows, page_info
            
            status = f"🔍 OCR Processing {len(non_ocr_docs)} non-OCR document(s)...\n"
            status += "=" * 50 + "\n\n"
            
            success_count = 0
            failed_count = 0
            
            from impl.ingest_pdf_v1 import PDFIngestorV1
            
            for doc in non_ocr_docs:
                try:
                    # Re-ingest with OCR
                    ingestor = PDFIngestorV1(
                        config=self.config,
                        store=self.store,
                        use_ocr=True
                    )
                    
                    status += f"⏳ Processing {doc.doc_id}...\n"
                    meta = ingestor.ingest(doc.source_path, doc_id=doc.doc_id)
                    status += f"✅ {doc.doc_id}: OCR completed ({meta.page_count} pages)\n"
                    success_count += 1
                    
                except Exception as e:
                    status += f"❌ {doc.doc_id}: {str(e)}\n"
                    failed_count += 1
            
            status += "\n" + "=" * 50 + "\n"
            status += f"📊 Summary: ✅ {success_count} succeeded | ❌ {failed_count} failed\n"
            status += f"💡 Total non-OCR documents: {len(non_ocr_docs)}\n"
            status += "⚠️  Next Step: Rebuild indices to use OCR text"
            
            stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
            return status, stats, rows, page_info
            
        except Exception as e:
            import traceback
            stats, rows, page_info = self._get_doc_list_paginated(page, page_size)
            return f"❌ Error: {str(e)}\n{traceback.format_exc()}", stats, rows, page_info


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
