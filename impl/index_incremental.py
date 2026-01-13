"""
Incremental indexing support - add documents to existing indices.
Avoids full index rebuilds when adding new documents.
"""
from typing import List, Set
from pathlib import Path

from core.schemas import AppConfig, IndexUnit
from infra.store_local import DocumentStoreLocal
from impl.index_bm25 import BM25IndexerRetriever
from impl.index_dense import DenseIndexerRetriever, VLLMEmbedder
from impl.index_colpali import ColPaliRetriever
from impl.index_tracker import IndexTracker


class IncrementalIndexManager:
    """
    Manages incremental index updates.
    Only indexes new/modified documents instead of rebuilding everything.
    """
    
    def __init__(self, config: AppConfig, store: DocumentStoreLocal):
        """
        Initialize incremental index manager.
        
        Args:
            config: Application config
            store: Document store
        """
        self.config = config
        self.store = store
    
    def get_new_documents(self, index_name: str) -> List[str]:
        """
        Get list of documents that need indexing.
        
        Args:
            index_name: Name of the index (e.g., "bm25_default")
            
        Returns:
            List of document IDs that are not yet indexed
        """
        # Load tracker
        index_dir = Path(self.config.indices_dir) / index_name
        tracker = IndexTracker(index_dir)
        
        # Get all documents in store
        all_docs = self.store.list_documents()
        all_doc_ids = {doc.doc_id for doc in all_docs}
        
        # Get already indexed documents
        indexed_doc_ids = tracker.get_indexed_docs()
        
        # Find new documents
        new_doc_ids = all_doc_ids - indexed_doc_ids
        
        return sorted(list(new_doc_ids))
    
    def update_bm25_index(
        self,
        index_name: str = "bm25_default",
        doc_ids: List[str] = None,
        force_rebuild: bool = False
    ) -> dict:
        """
        Incrementally update BM25 index with new documents.
        
        Args:
            index_name: Name of the index
            doc_ids: Specific documents to add (None = auto-detect new docs)
            force_rebuild: If True, rebuild entire index
            
        Returns:
            Dict with update statistics
        """
        index_dir = Path(self.config.indices_dir) / index_name
        tracker = IndexTracker(index_dir)
        
        # Auto-detect new documents if not specified
        if doc_ids is None:
            doc_ids = self.get_new_documents(index_name)
        
        if not doc_ids and not force_rebuild:
            return {
                "status": "no_update",
                "message": "No new documents to index",
                "new_docs": 0,
                "new_units": 0
            }
        
        # Initialize retriever
        retriever = BM25IndexerRetriever(self.store)
        
        # Load existing index if not rebuilding
        if not force_rebuild and index_dir.exists():
            retriever.load(self.config, index_name=index_name)
            existing_units = retriever.units.copy()
            print(f"📚 Loaded existing index: {len(existing_units)} units")
        else:
            existing_units = []
            if force_rebuild:
                print("🔄 Force rebuild: starting from scratch")
        
        # Build units for new documents
        new_units = []
        for doc_id in doc_ids:
            units = retriever.build_units(doc_id, self.config)
            new_units.extend(units)
            
            # Update tracker
            doc = self.store.get_document(doc_id)
            tracker.mark_indexed(doc_id, doc.page_count, len(units))
            print(f"  + {doc_id}: {len(units)} units")
        
        # Combine old and new units
        all_units = existing_units + new_units
        
        if not all_units:
            return {
                "status": "error",
                "message": "No units to index (documents have no text)",
                "new_docs": len(doc_ids),
                "new_units": 0
            }
        
        # Rebuild index with combined units
        print(f"🔨 Building index with {len(all_units)} total units...")
        stats = retriever.build_index(all_units, self.config)
        
        # Save
        retriever.persist(self.config, index_name=index_name)
        tracker.save()
        
        return {
            "status": "success",
            "message": f"Added {len(doc_ids)} documents",
            "new_docs": len(doc_ids),
            "new_units": len(new_units),
            "total_docs": stats.doc_count,
            "total_units": stats.unit_count,
            "index_name": index_name
        }
    
    def update_dense_index(
        self,
        index_name: str = "dense_default",
        doc_ids: List[str] = None,
        force_rebuild: bool = False
    ) -> dict:
        """
        Incrementally update Dense index with new documents.
        
        Args:
            index_name: Name of the index
            doc_ids: Specific documents to add (None = auto-detect new docs)
            force_rebuild: If True, rebuild entire index
            
        Returns:
            Dict with update statistics
        """
        index_dir = Path(self.config.indices_dir) / index_name
        tracker = IndexTracker(index_dir)
        
        # Auto-detect new documents if not specified
        if doc_ids is None:
            doc_ids = self.get_new_documents(index_name)
        
        if not doc_ids and not force_rebuild:
            return {
                "status": "no_update",
                "message": "No new documents to index",
                "new_docs": 0,
                "new_units": 0
            }
        
        # Initialize embedder and retriever
        # Create embedder (needed for both load and build)
        embedder = VLLMEmbedder(
            endpoint=self.config.dense["endpoint"],
            model=self.config.dense["model"],
            batch_size=self.config.dense.get("batch_size", 32)
        )
        
        # Load existing index if not rebuilding
        if not force_rebuild and index_dir.exists():
            retriever = DenseIndexerRetriever.load(index_dir, embedder)
            existing_units = retriever.units.copy()
            print(f"📚 Loaded existing index: {len(existing_units)} units")
        else:
            # Create new retriever instance
            retriever = DenseIndexerRetriever(
                embedder=embedder,
                index_type=self.config.dense.get("index_type", "Flat"),
                nlist=self.config.dense.get("nlist", 100),
                nprobe=self.config.dense.get("nprobe", 10)
            )
            existing_units = []
            if force_rebuild:
                print("🔄 Force rebuild: starting from scratch")
        
        # Build units for new documents
        bm25_retriever = BM25IndexerRetriever(self.store)
        new_units = []
        for doc_id in doc_ids:
            units = bm25_retriever.build_units(doc_id, self.config)
            new_units.extend(units)
            
            # Update tracker
            doc = self.store.get_document(doc_id)
            tracker.mark_indexed(doc_id, doc.page_count, len(units))
            print(f"  + {doc_id}: {len(units)} units")
        
        # Combine old and new units
        all_units = existing_units + new_units
        
        if not all_units:
            return {
                "status": "error",
                "message": "No units to index (documents have no text)",
                "new_docs": len(doc_ids),
                "new_units": 0
            }
        
        # Rebuild index with combined units (will call vLLM API)
        print(f"🔨 Building Dense index with {len(all_units)} total units...")
        print(f"   Calling vLLM embedding API: {self.config.dense['endpoint']}")
        retriever.build_index(all_units, self.config)
        
        # Save
        retriever.save(index_dir)
        tracker.save()
        
        return {
            "status": "success",
            "message": f"Added {len(doc_ids)} documents",
            "new_docs": len(doc_ids),
            "new_units": len(new_units),
            "total_units": len(all_units),
            "index_name": index_name
        }
    
    def update_colpali_index(
        self,
        index_name: str = "colpali_default",
        doc_ids: List[str] = None,
        force_rebuild: bool = False
    ) -> dict:
        """
        Incrementally update ColPali index with new documents.
        
        Args:
            index_name: Name of the index
            doc_ids: Specific documents to add (None = auto-detect new docs)
            force_rebuild: If True, rebuild entire index
            
        Returns:
            Dict with update statistics
        """
        index_dir = Path(self.config.indices_dir) / index_name
        tracker = IndexTracker(index_dir)
        
        # Auto-detect new documents if not specified
        if doc_ids is None:
            doc_ids = self.get_new_documents(index_name)
        
        if not doc_ids and not force_rebuild:
            return {
                "status": "no_update",
                "message": "No new documents to index",
                "new_docs": 0,
                "new_pages": 0
            }
        
        # Initialize retriever
        device = self.config.colpali.get("device", "cuda:2")
        retriever = ColPaliRetriever(
            model_name=self.config.colpali["model"],
            device=device,
            max_global_pool_pages=self.config.colpali.get("max_global_pool", 100)
        )
        
        # Load existing index if not rebuilding
        if not force_rebuild and index_dir.exists():
            retriever.load_instance(index_dir)
            existing_pages = len(retriever.store.page_ids)
            print(f"📚 Loaded existing index: {existing_pages} pages")
        else:
            existing_pages = 0
            if force_rebuild:
                print("🔄 Force rebuild: starting from scratch")
        
        # Build page list for new documents
        new_page_list = []
        for doc_id in doc_ids:
            doc = self.store.get_document(doc_id)
            for page_id in range(doc.page_count):
                new_page_list.append((doc_id, page_id))
            
            # Update tracker
            tracker.mark_indexed(doc_id, doc.page_count, doc.page_count)
            print(f"  + {doc_id}: {doc.page_count} pages")
        
        if not new_page_list:
            return {
                "status": "error",
                "message": "No pages to index",
                "new_docs": len(doc_ids),
                "new_pages": 0
            }
        
        # Add new pages to index
        print(f"🔨 Encoding {len(new_page_list)} new pages on {device}...")
        retriever.add_pages(new_page_list, self.config)
        
        # Save
        retriever.save(index_dir)
        tracker.save()
        
        return {
            "status": "success",
            "message": f"Added {len(doc_ids)} documents",
            "new_docs": len(doc_ids),
            "new_pages": len(new_page_list),
            "total_pages": len(retriever.store.page_ids),
            "index_name": index_name
        }
