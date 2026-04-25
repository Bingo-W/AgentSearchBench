from abc import ABC, abstractmethod
from typing import Literal
from functools import partial
from tqdm import tqdm

from asbench.retrieval.utils import load_documents, agentbase_indexing, artifact_indexing, query_indexing


class BaseRetriever(ABC):
    """
    Abstract base class for AgentBase retrievers.
    """

    def __init__(self, db_path: str, index_config: Literal["naive", "agentbase", "artifacts", "queries"]):
        self.db_path = db_path
        self.agent_ids = []
        self.corpus = []
        self.index_config = index_config
        self.indexing_func = {
            "naive": partial(load_documents, self.db_path),
            "agentbase": partial(agentbase_indexing, self.db_path),
            "artifacts": partial(artifact_indexing, self.db_path),
            "queries": partial(query_indexing, self.db_path)
        }

    @abstractmethod
    def build_index(self) -> None:
        """Build retrieval index from database."""
        pass

    @abstractmethod
    def sim(self, d1: str, d2: str) -> float:
        """Calculate similarity between two documents."""
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Retrieve top-k agents for a single query.

        Returns:
            List of (agent_id, score) tuples
        """
        pass

    @abstractmethod
    def retrieve_all(self, query: str) -> list[tuple[str, float]]:
        pass

    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve for multiple queries (for evaluation)."""
        results = {}
        for qid, query in tqdm(queries.items(), desc=f"Batch Retrieval"):
            results[qid] = self.retrieve(query, top_k)
        return results

    def batch_retrieve_all(self, queries: dict[str, str]) -> dict[str, list[tuple[str, float]]]:
        results = {}
        for qid, query in queries.items():
            results[qid] = self.retrieve_all(query)
        return results