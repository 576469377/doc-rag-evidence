"""
Enhanced Gradio UI V1 for doc-rag-evidence system.
Supports multiple retrieval modes: BM25, Dense, ColPali, Hybrid.
"""
from __future__ import annotations

import os
import uuid
import yaml
from pathlib import Path
from typing import Optional, List, Tuple
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
        
        # Initialize hybrid retrievers if multiple methods available
        self._init_hybrid_retrievers()

    def _init_hybrid_retrievers(self):
        """Initialize hybrid retrieval combinations."""
        available = [k for k, v in self.retrievers.items() if v is not None or k == "colpali"]
        
        # Dense + ColPali hybrid
        if "dense" in available and "colpali" in available:
            from impl.retriever_hybrid import HybridRetriever
            # Placeholder for lazy initialization
            self.retrievers["hybrid_dense_colpali"] = "lazy"
            print(f"✅ Hybrid (Dense+ColPali) available (延迟加载)")
        
        # BM25 + Dense hybrid  
        if "bm25" in available and "dense" in available:
            from impl.retriever_hybrid import HybridRetriever
            self.retrievers["hybrid_bm25_dense"] = HybridRetriever(
                retrievers={"bm25": self.retrievers["bm25"], "dense": self.retrievers["dense"]},
                weights={"bm25": 0.4, "dense": 0.6},
                fusion_method="weighted_sum"
            )
            print(f"✅ Hybrid (BM25+Dense) initialized")

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
        except Exception as e:
            print(f"❌ Failed to launch Gradio: {e}")
            # Try alternative port
            print("⚠️  Trying alternative port 7861...")
            demo.launch(
                share=share,
                server_name="127.0.0.1",
                server_port=7861,
                show_error=True
            )

    def _build_document_tab(self):
        """Build document management tab."""
        gr.Markdown("## Upload and Manage Documents")

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
                upload_status = gr.Textbox(label="Ingestion Status", lines=8, interactive=False)

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
            fn=self._handle_batch_upload,
            inputs=[pdf_files, use_ocr],
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
        
        # Evidence format selector (outside columns for better visibility)
        evidence_mode = gr.Radio(
            choices=["text", "image"],
            value="text",
            label="Evidence Format",
            info="text: 使用文本snippet | image: 使用完整页面图片（更准确，适合VL模型）"
        )
        
        # Hybrid fusion settings (collapsible)
        with gr.Accordion("⚙️ Hybrid Fusion Settings (仅对 Hybrid 模式生效)", open=False):
            gr.Markdown("### 自定义混合检索配置")
            
            with gr.Row():
                with gr.Column():
                    fusion_method = gr.Radio(
                        choices=["weighted_sum", "rrf"],
                        value="weighted_sum",
                        label="Fusion Method (融合方法)",
                        info="weighted_sum: 加权分数融合 | rrf: 倒数排名融合"
                    )
                    
                    gr.Markdown("""
                    #### 融合方法说明
                    - **Weighted Sum**: 将各检索器的分数归一化后加权求和
                      - 考虑分数大小，高分文档优势明显
                      - 适用：检索器分数有明确物理意义
                    
                    - **RRF (Reciprocal Rank Fusion)**: 只考虑排名位置
                      - 公式：score = sum(1 / (60 + rank))
                      - 对分数尺度不敏感，更鲁棒
                      - 适用：多个检索器分数范围差异大时
                    """)
                
                with gr.Column():
                    gr.Markdown("#### 检索器选择与权重")
                    
                    retriever_1 = gr.Dropdown(
                        choices=["bm25", "dense", "colpali"],
                        value="bm25",
                        label="First Retriever (检索器1)",
                        info="选择第一个检索器"
                    )
                    
                    retriever_2 = gr.Dropdown(
                        choices=["bm25", "dense", "colpali"],
                        value="dense",
                        label="Second Retriever (检索器2)",
                        info="选择第二个检索器"
                    )
                    
                    weight_1 = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.4,
                        step=0.05,
                        label="Weight of First Retriever",
                        info="第一个检索器的权重（第二个自动为 1 - weight_1）"
                    )
                    
                    weight_display = gr.Markdown(
                        "**当前权重**: 检索器1 = 0.40, 检索器2 = 0.60"
                    )
            
            gr.Markdown("---")
            gr.Markdown("💡 **快速配置**: 选择两个不同的检索器，调整权重滑块，点击 'Ask Question' 应用配置")
            
            # Update weight display when slider changes
            def update_weight_display(w1):
                w2 = 1.0 - w1
                return f"**当前权重**: 检索器1 = {w1:.2f}, 检索器2 = {w2:.2f}"
            
            weight_1.change(
                fn=update_weight_display,
                inputs=[weight_1],
                outputs=[weight_display]
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
                
                # Hybrid fusion settings for evaluation
                with gr.Accordion("⚙️ Hybrid Fusion Settings (仅对 Hybrid 模式生效)", open=False):
                    gr.Markdown("### 评估中的混合检索配置")
                    
                    with gr.Row():
                        with gr.Column():
                            eval_fusion_method = gr.Radio(
                                choices=["weighted_sum", "rrf"],
                                value="weighted_sum",
                                label="Fusion Method",
                                info="weighted_sum: 加权分数 | rrf: 倒数排名"
                            )
                        
                        with gr.Column():
                            eval_retriever_1 = gr.Dropdown(
                                choices=["bm25", "dense", "colpali"],
                                value="bm25",
                                label="First Retriever",
                                info="选择第一个检索器"
                            )
                            
                            eval_retriever_2 = gr.Dropdown(
                                choices=["bm25", "dense", "colpali"],
                                value="dense",
                                label="Second Retriever",
                                info="选择第二个检索器"
                            )
                            
                            eval_weight_1 = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.4,
                                step=0.05,
                                label="Weight of First Retriever",
                                info="第一个检索器的权重"
                            )
                            
                            eval_weight_display = gr.Markdown(
                                "**当前权重**: 检索器1 = 0.40, 检索器2 = 0.60"
                            )
                    
                    gr.Markdown("💡 Hybrid 模式下，将使用以上配置进行批量评估")
                    
                    # Update weight display
                    def update_eval_weight_display(w1):
                        w2 = 1.0 - w1
                        return f"**当前权重**: 检索器1 = {w1:.2f}, 检索器2 = {w2:.2f}"
                    
                    eval_weight_1.change(
                        fn=update_eval_weight_display,
                        inputs=[eval_weight_1],
                        outputs=[eval_weight_display]
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
                eval_fusion_method,
                eval_retriever_1,
                eval_retriever_2,
                eval_weight_1
            ],
            outputs=[eval_status, eval_metrics, download_csv, download_json]
        )

    # ========== Event Handlers ==========

    def _handle_batch_upload(self, pdf_files, use_ocr: bool) -> Tuple[str, List]:
        """Handle batch PDF upload and ingestion."""
        try:
            if pdf_files is None or len(pdf_files) == 0:
                return "❌ Error: No files uploaded", self._get_doc_list()
            
            # Handle single file or multiple files
            if not isinstance(pdf_files, list):
                pdf_files = [pdf_files]
            
            total_files = len(pdf_files)
            status_lines = []
            status_lines.append(f"📦 Batch Upload: {total_files} file(s)")
            status_lines.append("=" * 50)
            
            if use_ocr:
                status_lines.append("⚙️ OCR enabled - processing may take time...")
            status_lines.append("")
            
            # Process each file
            success_count = 0
            failed_count = 0
            ingested_docs = []
            
            for idx, pdf_file in enumerate(pdf_files, 1):
                try:
                    filename = Path(pdf_file.name).name
                    status_lines.append(f"[{idx}/{total_files}] Processing: {filename}")
                    
                    # Ingest with V1 ingestor
                    ingestor = PDFIngestorV1(
                        config=self.config,
                        store=self.store,
                        use_ocr=use_ocr
                    )
                    
                    meta = ingestor.ingest(pdf_file.name)
                    
                    status_lines.append(f"  ✅ Success: {meta.doc_id} ({meta.page_count} pages)")
                    ingested_docs.append(meta.doc_id)
                    success_count += 1
                    
                except Exception as e:
                    status_lines.append(f"  ❌ Failed: {str(e)}")
                    failed_count += 1
                
                status_lines.append("")  # Blank line between files
            
            # Summary
            status_lines.append("=" * 50)
            status_lines.append(f"📊 Summary:")
            status_lines.append(f"  ✅ Success: {success_count}/{total_files}")
            status_lines.append(f"  ❌ Failed: {failed_count}/{total_files}")
            
            if success_count > 0:
                status_lines.append(f"\n  Ingested IDs: {', '.join(ingested_docs)}")
                status_lines.append("\n⚠️ Next Step: Build indices below to enable retrieval")
            
            return "\n".join(status_lines), self._get_doc_list()
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Batch Upload Error: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            return error_msg, self._get_doc_list()
    
    def _handle_upload(self, pdf_file, use_ocr: bool) -> Tuple[str, List]:
        """Handle single PDF upload (legacy, kept for compatibility)."""
        # Redirect to batch handler
        return self._handle_batch_upload([pdf_file] if pdf_file else None, use_ocr)

    def _handle_build_indices(
        self,
        build_bm25: bool,
        build_dense: bool,
        build_colpali: bool,
        index_name_suffix: str
    ) -> str:
        """Handle index building (now with incremental updates)."""
        try:
            if not any([build_bm25, build_dense, build_colpali]):
                return "❌ Error: Please select at least one index type to build"

            status = "🔧 Building/Updating Indices (Incremental Mode)...\n\n"
            suffix = index_name_suffix.strip() or "default"
            
            # Get all documents
            docs = self.store.list_documents()
            if not docs:
                return "❌ Error: No documents found. Please ingest documents first."
            
            status += f"📚 Found {len(docs)} document(s)\n"
            
            # Initialize incremental index manager
            index_manager = IncrementalIndexManager(self.config, self.store)
            
            # BM25 incremental build/update
            if build_bm25:
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
            
            # Dense incremental build/update
            if build_dense:
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
            
            # ColPali incremental build/update
            if build_colpali:
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
            
            status += "\n" + "=" * 50 + "\n"
            status += "🎉 Index Building Complete!\n"
            status += "=" * 50 + "\n"
            status += f"\nAvailable retrieval modes: {list(self.retrievers.keys())}\n"
            status += "\nℹ️  Incremental indexing:\n"
            status += "   • Only new documents are indexed\n"
            status += "   • Existing indices are preserved and updated\n"
            status += "   • No need to rebuild everything when adding docs\n"
            status += "\nYou can now use the 'Query & Answer' tab.\n"
            
            return status
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"

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
        fusion_method: str = "weighted_sum",
        retriever_1: str = "bm25",
        retriever_2: str = "dense",
        weight_1: float = 0.4
    ) -> Tuple[str, List, str]:
        """Handle query with selected retrieval mode and evidence format."""
        try:
            if not question:
                return "Please enter a question", [], ""
            
            # Switch retriever (延迟加载ColPali和Hybrid)
            retriever = self.retrievers.get(retrieval_mode)
            
            # 如果是ColPali且还未加载，现在加载
            if retrieval_mode == "colpali" and retriever is None:
                if hasattr(self, "_colpali_config"):
                    try:
                        print(f"⏳ 首次使用ColPali，正在加载模型...")
                        retriever = ColPaliRetriever.load(
                            self._colpali_config["index_dir"],
                            model_name=self._colpali_config["model_name"],
                            device=self._colpali_config["device"]
                        )
                        self.retrievers["colpali"] = retriever
                        print(f"✅ ColPali模型加载完成")
                    except Exception as e:
                        return f"Failed to load ColPali: {e}", [], ""
                else:
                    return "ColPali not configured.", [], ""
            
            # 动态重建 Hybrid retriever（使用用户指定的检索器组合、权重和融合方法）
            if "hybrid" in retrieval_mode:
                from impl.retriever_hybrid import HybridRetriever
                
                # 验证用户选择的两个检索器不同
                if retriever_1 == retriever_2:
                    return f"⚠️ 请选择两个不同的检索器！当前都选择了 {retriever_1}", [], ""
                
                # 确保所需的检索器已加载
                retriever_objs = {}
                for ret_name in [retriever_1, retriever_2]:
                    if ret_name == "colpali" and self.retrievers.get("colpali") is None:
                        # 动态加载ColPali
                        if hasattr(self, "_colpali_config"):
                            try:
                                print(f"⏳ 首次使用ColPali，正在加载模型...")
                                colpali_retriever = ColPaliRetriever.load(
                                    self._colpali_config["index_dir"],
                                    model_name=self._colpali_config["model_name"],
                                    device=self._colpali_config["device"]
                                )
                                self.retrievers["colpali"] = colpali_retriever
                                print(f"✅ ColPali模型加载完成")
                            except Exception as e:
                                return f"Failed to load ColPali: {e}", [], ""
                        else:
                            return f"ColPali not configured.", [], ""
                    
                    if ret_name not in self.retrievers:
                        return f"⚠️ 检索器 '{ret_name}' 未找到，请先构建索引", [], ""
                    
                    retriever_objs[ret_name] = self.retrievers[ret_name]
                
                # 计算归一化权重
                weight_2 = 1.0 - weight_1
                weights = {retriever_1: weight_1, retriever_2: weight_2}
                
                # 创建 Hybrid retriever
                retriever = HybridRetriever(
                    retrievers=retriever_objs,
                    weights=weights,
                    fusion_method=fusion_method
                )
                
                print(f"🔄 Custom Hybrid ({retriever_1}+{retriever_2}) with {fusion_method}")
                print(f"   Weights: {retriever_1}={weight_1:.2f}, {retriever_2}={weight_2:.2f}")
            
            if retriever is None:
                return f"Retriever '{retrieval_mode}' not available. Please build indices first.", [], ""

            
            # Switch generator based on evidence mode
            if evidence_mode == "image":
                # Use image-based generator
                try:
                    from impl.generator_qwen_vl import QwenVLGenerator
                    generator = QwenVLGenerator(self.config, use_images=True, store=self.store)
                    print(f"🖼️  Using image-based generation")
                except Exception as e:
                    return f"Failed to load image generator: {e}", [], ""
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
            
            return answer, evidence_rows, query_input.query_id
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            return error_msg, [], ""

    def _handle_eval(
        self,
        eval_file,
        eval_mode: str,
        fusion_method: str = "weighted_sum",
        retriever_1: str = "bm25",
        retriever_2: str = "dense",
        weight_1: float = 0.4
    ) -> Tuple[str, dict, Optional[str], Optional[str]]:
        """Handle batch evaluation with custom hybrid configuration."""
        try:
            if eval_file is None:
                return "Error: No evaluation file uploaded", {}, None, None
            
            # Get or create retriever with custom hybrid config
            retriever = self.retrievers.get(eval_mode)
            
            # For hybrid modes, create custom configuration
            if "hybrid" in eval_mode:
                from impl.retriever_hybrid import HybridRetriever
                
                # Validate different retrievers
                if retriever_1 == retriever_2:
                    return f"⚠️ 请选择两个不同的检索器！当前都选择了 {retriever_1}", {}, None, None
                
                # Ensure retrievers are loaded
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
                                return f"Failed to load ColPali: {e}", {}, None, None
                        else:
                            return f"ColPali not configured.", {}, None, None
                    
                    if ret_name not in self.retrievers:
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
                
                print(f"📊 Evaluation Hybrid Config:")
                print(f"   {retriever_1} ({weight_1:.2f}) + {retriever_2} ({weight_2:.2f})")
                print(f"   Fusion: {fusion_method}")
            
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
            if "hybrid" in eval_mode:
                status += f"Config: {retriever_1}({weight_1:.2f}) + {retriever_2}({weight_2:.2f}), {fusion_method}\n"
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
