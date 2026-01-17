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
        
        # Initialize generator based on config
        generator_type = self.config.generator.get("type", "template")
        if generator_type == "qwen3_vl":
            try:
                from impl.generator_qwen_llm import QwenLLMGenerator
                self.generator = QwenLLMGenerator(self.config)
                print(f"✅ Using QwenLLMGenerator")
            except Exception as e:
                print(f"⚠️  Failed to load QwenLLMGenerator: {e}, falling back to template")
                from impl.generator_template import TemplateGenerator
                self.generator = TemplateGenerator(mode="summary")
        else:
            from impl.generator_template import TemplateGenerator
            self.generator = TemplateGenerator(mode="summary")
            print(f"✅ Using TemplateGenerator")
        
        # Create pipeline (default retriever) with store for hit normalization
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
            print(f"✅ Loaded BM25 index: {len(retriever.units)} units")
        except Exception as e:
            print(f"❌ Failed to load BM25 index: {e}")
        
        # Dense (vLLM embedding)
        if self.config.dense.get("enabled"):
            dense_index_name = "dense_default"
            dense_index_dir = indices_dir / dense_index_name
            if dense_index_dir.exists():
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
        
        # ColPali (vision embedding on GPU 2) - 延迟加载，只记录配置
        if self.config.colpali.get("enabled"):
            colpali_index_name = "colpali_default"
            colpali_index_dir = indices_dir / colpali_index_name
            if colpali_index_dir.exists():
                # 不立即加载模型，只注册可用性
                self.retrievers["colpali"] = None  # Placeholder，延迟加载
                self._colpali_config = {
                    "index_dir": colpali_index_dir,
                    "model_name": self.config.colpali["model"],
                    "device": self.config.colpali.get("device", "cuda:2")
                }
                print(f"✅ ColPali index available (延迟加载模式)")
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
                pdf_files = gr.File(
                    label="Upload PDF(s)", 
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath"
                )
                use_ocr = gr.Checkbox(label="Use OCR (slower, better quality)", value=False)
                upload_btn = gr.Button("📤 Ingest Document(s)", variant="primary")
                upload_status = gr.Textbox(label="Ingestion Status", lines=10, interactive=False)

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
        gr.Markdown("⚠️ **Note**: Build one index at a time to avoid GPU OOM")
        
        with gr.Row():
            with gr.Column():
                index_type = gr.Radio(
                    choices=[
                        "bm25",
                        "dense",
                        "dense_vl",
                        "colpali"
                    ],
                    value="bm25",
                    label="Select Index Type"
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
                    lines=10,
                    interactive=False,
                    placeholder="Status will appear here..."
                )

        # Event handlers
        upload_btn.click(
            fn=self._handle_batch_upload,
            inputs=[pdf_files, use_ocr],
            outputs=[upload_status, doc_list],
            show_progress=True
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
                for mode in ["bm25", "dense", "colpali", "dense_vl"]:
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
                    info="单一检索 or Hybrid（混合检索，可在下方配置）"
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
                        choices=["bm25", "dense", "colpali", "dense_vl"],
                        value="bm25",
                        label="检索器 1",
                        info="第一个检索器"
                    )
                    
                    retriever_2 = gr.Dropdown(
                        choices=["bm25", "dense", "colpali", "dense_vl"],
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
                for mode in ["bm25", "dense", "colpali", "dense_vl"]:
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
                    info="单一检索 or Hybrid（混合检索，可在下方配置）"
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
                                choices=["bm25", "dense", "colpali", "dense_vl"],
                                value="bm25",
                                label="检索器 1"
                            )
                            
                            eval_retriever_2 = gr.Dropdown(
                                choices=["bm25", "dense", "colpali", "dense_vl"],
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

    def _handle_batch_upload(self, pdf_files, use_ocr: bool) -> Generator[Tuple[str, List], None, None]:
        """Handle batch PDF upload and ingestion with streaming progress."""
        try:
            print(f"[DEBUG] _handle_batch_upload called with {len(pdf_files) if pdf_files else 0} files")
            
            if pdf_files is None or len(pdf_files) == 0:
                print("[DEBUG] No files uploaded")
                yield "❌ Error: No files uploaded", self._get_doc_list()
                return
            
            # Handle single file or multiple files
            if not isinstance(pdf_files, list):
                pdf_files = [pdf_files]
            
            total_files = len(pdf_files)
            
            # Warning for large batches
            if total_files > 100:
                status = f"⚠️ Large batch detected: {total_files} files\n"
                status += "📊 Processing in batches to avoid memory issues...\n"
                status += "=" * 50 + "\n\n"
                yield status, self._get_doc_list()
            else:
                status = f"📦 Batch Upload: {total_files} file(s)\n"
                status += "=" * 50 + "\n"
                if use_ocr:
                    status += "⚙️ OCR enabled - processing may take time...\n"
                status += "\n"
                yield status, self._get_doc_list()
            
            # Process in batches to avoid memory issues
            BATCH_SIZE = 50  # Process 50 files at a time
            success_count = 0
            failed_count = 0
            ingested_docs = []
            
            import time
            start_time = time.time()
            
            for batch_idx in range(0, total_files, BATCH_SIZE):
                batch_end = min(batch_idx + BATCH_SIZE, total_files)
                batch_files = pdf_files[batch_idx:batch_end]
                
                if total_files > BATCH_SIZE:
                    status += f"\n🔄 Processing batch {batch_idx//BATCH_SIZE + 1}/{(total_files + BATCH_SIZE - 1)//BATCH_SIZE}\n"
                    status += f"   Files {batch_idx + 1}-{batch_end} of {total_files}\n\n"
                    yield status, self._get_doc_list()
                
                # Process each file in the batch
                for idx, pdf_file in enumerate(batch_files, batch_idx + 1):
                    try:
                        filename = Path(pdf_file.name).name
                        status += f"[{idx}/{total_files}] Processing: {filename}\n"
                        yield status, self._get_doc_list()
                        
                        # Ingest with V1 ingestor
                        ingestor = PDFIngestorV1(
                            config=self.config,
                            store=self.store,
                            use_ocr=use_ocr
                        )
                        
                        meta = ingestor.ingest(pdf_file.name)
                        
                        status += f"  ✅ Success: {meta.doc_id} ({meta.page_count} pages)\n"
                        ingested_docs.append(meta.doc_id)
                        success_count += 1
                        
                        # Show progress every 10 files
                        if idx % 10 == 0:
                            elapsed = time.time() - start_time
                            avg_time = elapsed / idx
                            remaining = (total_files - idx) * avg_time
                            status += f"\n📊 Progress: {idx}/{total_files} ({idx*100//total_files}%)"
                            status += f" | Elapsed: {elapsed:.0f}s | ETA: {remaining:.0f}s\n\n"
                        
                        yield status, self._get_doc_list()
                        
                    except Exception as e:
                        status += f"  ❌ Failed: {str(e)}\n"
                        failed_count += 1
                        yield status, self._get_doc_list()
                    
                    status += "\n"
                
                # Clean up batch resources
                import gc
                gc.collect()
            
            # Final summary
            elapsed_total = time.time() - start_time
            status += "=" * 50 + "\n"
            status += f"📊 Final Summary:\n"
            status += f"  ✅ Success: {success_count}/{total_files}\n"
            status += f"  ❌ Failed: {failed_count}/{total_files}\n"
            status += f"  ⏱️ Total time: {elapsed_total:.0f}s ({elapsed_total/total_files:.1f}s per file)\n"
            
            if success_count > 0:
                status += f"\n  First 10 IDs: {', '.join(ingested_docs[:10])}"
                if len(ingested_docs) > 10:
                    status += f" ... and {len(ingested_docs) - 10} more"
                status += "\n\n⚠️ Next Step: Build indices below to enable retrieval"
            
            yield status, self._get_doc_list()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Batch Upload Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            yield error_msg, self._get_doc_list()
    
    def _handle_upload(self, pdf_file, use_ocr: bool) -> Generator[Tuple[str, List], None, None]:
        """Handle single PDF upload (legacy, kept for compatibility)."""
        # Redirect to batch handler
        for result in self._handle_batch_upload([pdf_file] if pdf_file else None, use_ocr):
            yield result

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
            
            status += f"📚 Found {len(docs)} document(s)\n"
            yield status
            
            # Initialize incremental index manager
            index_manager = IncrementalIndexManager(self.config, self.store)
            
            # Build the selected index type
            if index_type == "bm25":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ BM25 Index Update\n"
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
            
            # Dense incremental build/update
            elif index_type == "dense":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ Dense Index Update\n"
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
            
            # Dense-VL incremental build/update
            elif index_type == "dense_vl":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ Dense-VL Index Update\n"
                status += "─" * 50 + "\n"
                try:
                    # Use offline script to build Dense-VL index
                    import subprocess
                    import re
                    script_path = Path(__file__).parent.parent.parent / "scripts" / "build_dense_vl_index.py"
                    
                    # Prepare environment with GPU setting
                    env = os.environ.copy()
                    gpu_id = self.config.dense_vl.get("gpu", 1)
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    status += f"📍 Using GPU: {gpu_id}\n"
                    
                    # Run offline build script with streaming output
                    process = subprocess.Popen(
                        ["python", "-u", str(script_path)],  # -u for unbuffered output
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        bufsize=1,  # Line buffered
                        universal_newlines=True
                    )
                    
                    # Filter function to show only important lines
                    def should_show_line(line: str) -> bool:
                        """Filter out technical details and keep only important status."""
                        line = line.strip()
                        if not line:
                            return False
                        
                        # Skip these patterns
                        skip_patterns = [
                            "Imported Qwen3VLEmbedder",
                            "Loading Qwen3VL Embedding model",
                            "Original GPU ID",
                            "Image resize:",
                            "Using device",
                            "CUDA_VISIBLE_DEVICES",
                            "Using Flash Attention",
                            "Model loaded on",
                            "Calling offline build script",
                            "Script:",
                            "Each worker loads",
                            "Expected GPU usage",
                            "Split into",
                            "Progress (each worker",
                            "Worker ",  # Skip individual worker progress lines
                            "[A",  # Skip ANSI escape sequences
                        ]
                        
                        for pattern in skip_patterns:
                            if pattern in line:
                                return False
                        
                        # Show these important patterns
                        show_patterns = [
                            "Found",
                            "document",
                            "New documents:",
                            "pages",
                            "Total pages to embed:",
                            "Set CUDA_VISIBLE_DEVICES",
                            "Using",
                            "parallel workers",
                            "Embeddings shape:",
                            "Index saved",
                            "Total",
                            "Embedding dim:",
                            "✅",
                            "❌",
                            "⚠️",
                            "🔨",
                            "📚",
                            "📝",
                            "⚡",
                        ]
                        
                        for pattern in show_patterns:
                            if pattern in line:
                                return True
                        
                        return False
                    
                    # Stream filtered output in real-time
                    output_lines = []
                    for line in process.stdout:
                        if should_show_line(line):
                            # Remove ANSI escape sequences
                            clean_line = re.sub(r'\x1b\[[0-9;]*[mGKHJA]', '', line)
                            status += clean_line
                            output_lines.append(clean_line.rstrip())
                            yield status  # Update UI in real-time
                    
                    process.wait()
                    
                    if process.returncode == 0:
                        # Reset retriever to None (will lazy load on next use)
                        self.retrievers["dense_vl"] = None
                        
                        # Update config to point to new index
                        dense_vl_index_dir = Path(self.config.indices_dir) / f"dense_vl_{suffix}"
                        if dense_vl_index_dir.exists():
                            self._dense_vl_config = {
                                "index_dir": dense_vl_index_dir,
                                "model_path": self.config.dense_vl["model_path"],
                                "gpu": self.config.dense_vl.get("gpu", 1),
                                "gpu_memory": self.config.dense_vl.get("gpu_memory", 0.45),
                                "batch_size": self.config.dense_vl.get("batch_size", 8)
                            }
                        
                        status += f"\n✅ Dense-VL index built successfully (offline mode)\n"
                        status += f"   Index will lazy-load on first query\n"
                    else:
                        status += f"\n❌ Build script failed with exit code {process.returncode}\n"
                        
                except Exception as e:
                    import traceback
                    status += f"❌ Error: {str(e)}\n"
                    status += f"{traceback.format_exc()}\n"
                yield status
            
            # ColPali incremental build/update
            elif index_type == "colpali":
                status += "\n" + "─" * 50 + "\n"
                status += "⏳ ColPali Index Update\n"
                status += "─" * 50 + "\n"
                try:
                    result = index_manager.update_colpali_index(
                        index_name=f"colpali_{suffix}"
                    )
                    
                    if result["status"] == "success":
                        # Reload retriever
                        device = self.config.colpali.get("device", "cuda:2")
                        retriever = ColPaliRetriever(
                            model_name=self.config.colpali["model"],
                            device=device,
                            max_global_pool_pages=self.config.colpali.get("max_global_pool", 100)
                        )
                        index_dir = Path(self.config.indices_dir) / f"colpali_{suffix}"
                        retriever.load_instance(index_dir)
                        self.retrievers["colpali"] = retriever
                        
                        status += f"✅ Success!\n"
                        status += f"   New documents: {result['new_docs']}\n"
                        status += f"   New pages: {result['new_pages']}\n"
                        status += f"   Total: {result['total_pages']} pages\n"
                        status += f"   Device: {device}\n"
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
                                    device=self._colpali_config["device"]
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
                                device=self._colpali_config["device"]
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
                                    device=self._colpali_config["device"]
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
                                device=self._colpali_config["device"]
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
