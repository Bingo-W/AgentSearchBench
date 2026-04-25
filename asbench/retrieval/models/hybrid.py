from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Literal, Type, Any
import numpy as np
import logging

from asbench.retrieval.base import BaseRetriever
from asbench.retrieval.models.bm25 import BM25Retriever
from asbench.retrieval.models.sentence_transformer import DenseRetriever

logger = logging.getLogger(__name__)


RETRIEVER_REGISTRY: Dict[str, Type[BaseRetriever]] = {
    "bm25": BM25Retriever,
    "bge": DenseRetriever,
    "toolret": DenseRetriever,
}

@dataclass
class RetrieverConfig:
    retriever_type: str
    weight: float
    db_path: str
    index_config: Literal["naive", "v1", "artifacts", "queries"]
    kwargs: Dict[str, Any] = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, configs: List[RetrieverConfig]):
        if not configs:
            raise ValueError("HybridRetriever requires at least one RetrieverConfig.")

        self._validate_configs(configs)
        self.retrievers: List[BaseRetriever] = []
        self.weights: List[float] = []

        cache: Dict[Tuple, BaseRetriever] = {}

        for cfg in configs:
            cache_key = (cfg.retriever_type, cfg.db_path, cfg.index_config)
            if cache_key not in cache:
                cache[cache_key] = self._instantiate(cfg)
            self.retrievers.append(cache[cache_key])
            self.weights.append(cfg.weight)

        self._normalise_weights()

    @classmethod
    def from_loaded(
        cls,
        retrievers: List[BaseRetriever],
        weights: List[float],
    ) -> "HybridRetriever":
        """
        Construct a HybridRetriever from already-instantiated retrievers.
        Skips model loading entirely — only weights change between calls.
        """
        if not retrievers:
            raise ValueError("HybridRetriever requires at least one retriever.")
        if len(retrievers) != len(weights):
            raise ValueError(
                f"Number of retrievers ({len(retrievers)}) must match "
                f"number of weights ({len(weights)})."
            )

        instance = cls.__new__(cls)  # bypass __init__
        instance.retrievers = retrievers
        instance.weights = weights
        instance._normalise_weights()
        return instance

    def update_weights(self, weights: List[float]) -> None:
        """Swap weights in-place without rebuilding the retriever."""
        if len(weights) != len(self.retrievers):
            raise ValueError(
                f"Expected {len(self.retrievers)} weights, got {len(weights)}."
            )
        self.weights = weights
        self._normalise_weights()

    def _normalise_weights(self) -> None:
        total = sum(self.weights)
        if total <= 0:
            raise ValueError(f"Weights must sum to a positive value; got {total}.")
        self.norm_weights = [w / total for w in self.weights]

    @classmethod
    def preload(cls, configs: List[RetrieverConfig]) -> List[BaseRetriever]:
        """Load and cache retrievers without constructing a HybridRetriever."""
        cls._validate_configs(configs)
        cache: Dict[Tuple, BaseRetriever] = {}
        retrievers = []
        for cfg in configs:
            key = (cfg.retriever_type, cfg.db_path, cfg.index_config)
            if key not in cache:
                cache[key] = cls._instantiate(cfg)
            retrievers.append(cache[key])
        return retrievers

    @staticmethod
    def _validate_configs(configs: List[RetrieverConfig]) -> None:
        for cfg in configs:
            if cfg.retriever_type not in RETRIEVER_REGISTRY:
                raise ValueError(
                    f"Unknown retriever type '{cfg.retriever_type}'. "
                    f"Registered types: {list(RETRIEVER_REGISTRY)}"
                )

    @staticmethod
    def _instantiate(cfg: RetrieverConfig) -> BaseRetriever:
        cls = RETRIEVER_REGISTRY[cfg.retriever_type]
        return cls(db_path=cfg.db_path, index_config=cfg.index_config, **cfg.kwargs)

    @staticmethod
    def _minmax_normalise(scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return scores
        values = np.array(list(scores.values()), dtype=float)
        lo, hi = values.min(), values.max()
        if hi == lo: # identical scores, map to 1
            return {k: 1.0 for k in scores}
        return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}

    def _fuse(
        self,
        all_results: List[List[Tuple[str, float]]],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        """
        Weighted score fusion across all component retrievers.
        """
        fused: Dict[str, float] = {}

        for norm_weight, results in zip(self.norm_weights, all_results):
            score_map = dict(results)
            norm_map = self._minmax_normalise(score_map)
            for agent_id, norm_score in norm_map.items(): # 0 if missing document
                fused[agent_id] = fused.get(agent_id, 0.0) + norm_weight * norm_score

        sorted_results = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k] if top_k is not None else sorted_results
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text using the first retriever that supports encoding."""
        for retriever in self.retrievers:
            if hasattr(retriever, "model") and hasattr(retriever.model, "encode"):
                return retriever.model.encode(text, show_progress_bar=False)
        raise NotImplementedError("No component retriever supports encoding.")

    def sim(self, d1: str, d2: str) -> float:
        total_weight = sum(self.norm_weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero.")
        sim_sum = 0.0
        for norm_weight, retriever in zip(self.norm_weights, self.retrievers):
            sim_sum += norm_weight * retriever.sim(d1, d2)
        return sim_sum / total_weight

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        all_results: List[List[Tuple[str, float]]] = [
            r.retrieve_all(query) for r in self.retrievers
        ]
        return self._fuse(all_results, top_k)

    def batch_retrieve(
        self, queries: Dict[str, str], top_k: int = 10
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Retrieve for multiple queries (for evaluation)."""
        results = {}
        for qid, query in queries.items():
            results[qid] = self.retrieve(query, top_k)
        return results
    
    def batch_retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
        results = {}
        for qid, query in queries.items():
            results[qid] = self.retrieve(query, top_k=None)  # retrieve all results for fusion
        return results
