# impl/metrics.py
"""Evaluation metrics for RAG system."""

from typing import List, Dict, Set, Tuple, Optional
import re
from collections import Counter


class RetrievalMetrics:
    """
    Retrieval-level metrics.
    
    Measures how well the retrieval system finds relevant documents/blocks.
    """
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
        """
        Recall@K: fraction of relevant items in top-K results.
        
        Args:
            retrieved: List of retrieved unit_ids (ordered by rank)
            relevant: Set of ground truth relevant unit_ids
            k: Cutoff
        
        Returns:
            Recall@K in [0, 1]
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = retrieved_at_k & relevant
        
        return len(hits) / len(relevant)
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: Set[str], k: int = 10) -> float:
        """
        Precision@K: fraction of top-K results that are relevant.
        
        Args:
            retrieved: List of retrieved unit_ids
            relevant: Set of ground truth relevant unit_ids
            k: Cutoff
        
        Returns:
            Precision@K in [0, 1]
        """
        if k == 0:
            return 0.0
        
        retrieved_at_k = set(retrieved[:k])
        hits = retrieved_at_k & relevant
        
        return len(hits) / min(k, len(retrieved))
    
    @staticmethod
    def mrr(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Mean Reciprocal Rank: 1 / (rank of first relevant item).
        
        Args:
            retrieved: List of retrieved unit_ids (ordered)
            relevant: Set of relevant unit_ids
        
        Returns:
            MRR in [0, 1]
        """
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                return 1.0 / rank
        
        return 0.0
    
    @staticmethod
    def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
        """
        Average Precision: mean of precision values at each relevant position.
        
        Args:
            retrieved: List of retrieved unit_ids
            relevant: Set of relevant unit_ids
        
        Returns:
            AP in [0, 1]
        """
        if not relevant:
            return 0.0
        
        num_hits = 0
        sum_precisions = 0.0
        
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                num_hits += 1
                precision_at_rank = num_hits / rank
                sum_precisions += precision_at_rank
        
        if num_hits == 0:
            return 0.0
        
        return sum_precisions / len(relevant)
    
    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int = 10
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@K.
        
        Args:
            retrieved: List of retrieved unit_ids
            relevance_scores: Dict mapping unit_id to relevance score (e.g., 0-3)
            k: Cutoff
        
        Returns:
            NDCG@K in [0, 1]
        """
        def dcg(scores: List[float]) -> float:
            """Compute DCG."""
            return sum(
                (2 ** score - 1) / (i + 2) ** 0.5  # log2(i+2) in denominator
                for i, score in enumerate(scores)
            )
        
        # Actual DCG
        retrieved_at_k = retrieved[:k]
        actual_scores = [relevance_scores.get(item, 0.0) for item in retrieved_at_k]
        actual_dcg = dcg(actual_scores)
        
        # Ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal_scores)
        
        if ideal_dcg == 0:
            return 0.0
        
        return actual_dcg / ideal_dcg


class EvidenceMetrics:
    """
    Evidence-level metrics.
    
    Measures whether selected evidence contains information needed for answer.
    """
    
    @staticmethod
    def evidence_hit_rate(
        selected_evidence: List[str],
        ground_truth_evidence: Set[str]
    ) -> float:
        """
        Evidence Hit Rate: fraction of GT evidence found in selected evidence.
        
        Args:
            selected_evidence: List of selected unit_ids
            ground_truth_evidence: Set of GT unit_ids that contain answer
        
        Returns:
            Hit rate in [0, 1]
        """
        if not ground_truth_evidence:
            return 1.0  # No GT evidence required
        
        selected_set = set(selected_evidence)
        hits = selected_set & ground_truth_evidence
        
        return len(hits) / len(ground_truth_evidence)
    
    @staticmethod
    def evidence_precision(
        selected_evidence: List[str],
        ground_truth_evidence: Set[str]
    ) -> float:
        """
        Evidence Precision: fraction of selected evidence that is relevant.
        
        Args:
            selected_evidence: List of selected unit_ids
            ground_truth_evidence: Set of relevant unit_ids
        
        Returns:
            Precision in [0, 1]
        """
        if not selected_evidence:
            return 0.0
        
        selected_set = set(selected_evidence)
        hits = selected_set & ground_truth_evidence
        
        return len(hits) / len(selected_set)
    
    @staticmethod
    def citation_accuracy(
        generated_citations: List[str],
        ground_truth_evidence: Set[str]
    ) -> float:
        """
        Citation Accuracy: fraction of cited evidence that is truly relevant.
        
        Penalizes hallucinated citations.
        
        Args:
            generated_citations: List of unit_ids cited in answer
            ground_truth_evidence: Set of relevant unit_ids
        
        Returns:
            Accuracy in [0, 1]
        """
        if not generated_citations:
            return 0.0
        
        correct_citations = sum(1 for cit in generated_citations if cit in ground_truth_evidence)
        
        return correct_citations / len(generated_citations)


class GenerationMetrics:
    """
    Generation-level metrics.
    
    Measures quality of generated answer.
    """
    
    @staticmethod
    def exact_match(predicted: str, gold: str, ignore_case: bool = True) -> float:
        """
        Exact Match: 1.0 if predicted == gold, else 0.0.
        
        Args:
            predicted: Generated answer
            gold: Ground truth answer
            ignore_case: Whether to ignore case
        
        Returns:
            EM score (0.0 or 1.0)
        """
        pred = predicted.strip()
        gold_text = gold.strip()
        
        if ignore_case:
            pred = pred.lower()
            gold_text = gold_text.lower()
        
        return 1.0 if pred == gold_text else 0.0
    
    @staticmethod
    def f1_score(predicted: str, gold: str) -> float:
        """
        Token-level F1 score.
        
        Args:
            predicted: Generated answer
            gold: Ground truth answer
        
        Returns:
            F1 score in [0, 1]
        """
        pred_tokens = GenerationMetrics._normalize_text(predicted).split()
        gold_tokens = GenerationMetrics._normalize_text(gold).split()
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for token-level comparison."""
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def keyphrase_match(
        predicted: str,
        gold_keyphrases: List[str],
        partial: bool = True
    ) -> float:
        """
        Keyphrase Match: fraction of gold keyphrases found in prediction.
        
        Useful for domain-specific answers (e.g., "碳酸氢钠" in chemistry).
        
        Args:
            predicted: Generated answer
            gold_keyphrases: List of required keyphrases
            partial: Allow partial match (substring)
        
        Returns:
            Match rate in [0, 1]
        """
        if not gold_keyphrases:
            return 1.0
        
        pred_normalized = GenerationMetrics._normalize_text(predicted)
        
        matches = 0
        for phrase in gold_keyphrases:
            phrase_normalized = GenerationMetrics._normalize_text(phrase)
            
            if partial:
                if phrase_normalized in pred_normalized:
                    matches += 1
            else:
                if phrase_normalized == pred_normalized:
                    matches += 1
        
        return matches / len(gold_keyphrases)
    
    @staticmethod
    def citation_coverage(
        answer: str,
        citation_pattern: str = r'\[(\d+)\]'
    ) -> Tuple[int, List[int]]:
        """
        Extract citations from answer.
        
        Args:
            answer: Generated answer with citations
            citation_pattern: Regex pattern for citations
        
        Returns:
            (num_citations, list_of_citation_numbers)
        """
        citations = re.findall(citation_pattern, answer)
        citation_numbers = [int(c) for c in citations]
        
        return len(citation_numbers), citation_numbers


class RAGMetrics:
    """
    Combined RAG metrics evaluator.
    """
    
    def __init__(self):
        self.retrieval = RetrievalMetrics()
        self.evidence = EvidenceMetrics()
        self.generation = GenerationMetrics()
    
    def evaluate_full_pipeline(
        self,
        query_result: "QueryRunRecord",
        ground_truth: Dict
    ) -> Dict[str, float]:
        """
        Evaluate full RAG pipeline with ground truth.
        
        Args:
            query_result: QueryRunRecord from pipeline
            ground_truth: Dict with keys:
                - relevant_units: Set[str] (relevant unit_ids)
                - evidence_units: Set[str] (units containing answer)
                - answer: str (gold answer)
                - keyphrases: List[str] (optional)
        
        Returns:
            Dict of metric_name -> score
        """
        metrics = {}
        
        # Retrieval metrics
        if query_result.retrieval and ground_truth.get("relevant_units"):
            retrieved_ids = [h.unit_id for h in query_result.retrieval.hits]
            relevant = ground_truth["relevant_units"]
            
            metrics["retrieval_recall@5"] = self.retrieval.recall_at_k(retrieved_ids, relevant, k=5)
            metrics["retrieval_recall@10"] = self.retrieval.recall_at_k(retrieved_ids, relevant, k=10)
            metrics["retrieval_precision@5"] = self.retrieval.precision_at_k(retrieved_ids, relevant, k=5)
            metrics["retrieval_mrr"] = self.retrieval.mrr(retrieved_ids, relevant)
        
        # Evidence metrics
        if query_result.evidence and ground_truth.get("evidence_units"):
            selected_ids = [ev.unit_id for ev in query_result.evidence.items]
            evidence_gt = ground_truth["evidence_units"]
            
            metrics["evidence_hit_rate"] = self.evidence.evidence_hit_rate(selected_ids, evidence_gt)
            metrics["evidence_precision"] = self.evidence.evidence_precision(selected_ids, evidence_gt)
        
        # Generation metrics
        if query_result.generation and ground_truth.get("answer"):
            predicted = query_result.generation.output.answer
            gold = ground_truth["answer"]
            
            metrics["generation_f1"] = self.generation.f1_score(predicted, gold)
            metrics["generation_em"] = self.generation.exact_match(predicted, gold)
            
            if ground_truth.get("keyphrases"):
                metrics["keyphrase_match"] = self.generation.keyphrase_match(
                    predicted,
                    ground_truth["keyphrases"]
                )
            
            # Citation analysis
            num_cit, cit_list = self.generation.citation_coverage(predicted)
            metrics["num_citations"] = num_cit
            
            if ground_truth.get("evidence_units"):
                # Map citation numbers to unit_ids
                cited_units = query_result.generation.output.cited_units
                evidence_gt = ground_truth["evidence_units"]
                
                if cited_units:
                    metrics["citation_accuracy"] = self.evidence.citation_accuracy(
                        cited_units,
                        evidence_gt
                    )
        
        return metrics
    
    def aggregate_metrics(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate metrics across multiple queries.
        
        Args:
            all_metrics: List of metric dicts (one per query)
        
        Returns:
            Dict of metric_name -> mean_score
        """
        if not all_metrics:
            return {}
        
        # Collect all metric names
        all_keys = set()
        for m in all_metrics:
            all_keys.update(m.keys())
        
        # Compute mean for each metric
        aggregated = {}
        for key in all_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
