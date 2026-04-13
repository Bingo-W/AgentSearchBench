from typing import List, Tuple, Dict, Literal
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from asbench.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retrieval using sentence transformers."""

    def __init__(
        self, model_name: str, db_path: str, index_config: Literal["naive", "v1", "artifacts", "queries"]
    ):
        super().__init__(db_path, index_config)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.embeddings_path = self._default_embeddings_path()
        self.load_index() if self._embeddings_exist() else self.build_index()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def _default_embeddings_path(self) -> str:
        model_safe = self.model_name.replace("/", "_")
        payload = {
            "model": self.model_name,
            "filename": str(self.db_path).split("/")[-1],
            "columns": self.index_config,
            "rows": len(pd.read_csv(self.db_path)) if self.index_config != "queries" else "N/A",
        }
        # stable SHA-256 hash based on payload
        hash_str = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return f"asbench/data/embeddings/{model_safe}_{hash_str}.npz"

    def _embeddings_exist(self) -> bool:
        return Path(self.embeddings_path).exists()

    def _store_index(self):
        path = Path(self.embeddings_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.corpus_embeddings,
            agent_ids=np.array(self.agent_ids),
            model_name=self.model_name,
            index_config=self.index_config,
        )

    def load_index(self):
        """
        Use precomputed embeddings.
        """
        data = np.load(self.embeddings_path, allow_pickle=True)
        self.corpus_embeddings = data["embeddings"]
        self.agent_ids = data["agent_ids"].tolist()

        # verify metadata
        stored_model = str(data["model_name"])
        stored_index_config = data["index_config"].tolist()
        if stored_model != self.model_name:
            print(
                f"WARNING: Loaded embeddings from {stored_model}, but using {self.model_name}"
            )
        if stored_index_config != self.index_config:
            raise ValueError(
                f"Index configuration mismatch! Stored: {stored_index_config}, Expected: {self.index_config}"
            )

    def build_index(self):
        """
        Build your embeddings.
        """
        self.agent_ids, self.corpus = self.indexing_func[self.index_config]()
        self.corpus_embeddings = self.model.encode(
            self.corpus,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        self._store_index()  # avoid re-building

    def sim(self, d1: str, d2: str) -> float:
        emb1 = self.model.encode(d1, convert_to_numpy=True)
        emb2 = self.model.encode(d2, convert_to_numpy=True)
        return float(np.dot(emb1, emb2))

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve using dot product.
        """
        # NOTE: .encode() internally normalises our vectors
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        scores = np.dot(self.corpus_embeddings, query_embedding)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.agent_ids[idx], float(scores[idx])) for idx in top_indices]
    
    def retrieve_all(self, query: str):
        query_embedding = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        scores = np.dot(self.corpus_embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1]
        return [(self.agent_ids[idx], float(scores[idx])) for idx in top_indices]
    
    def batch_retrieve(
        self, queries: dict[str, str], top_k: int = 10
    ) -> dict[str, list[tuple[str, float]]]:
        """Retrieve for multiple queries (for evaluation)."""
        query_embeddings = self.model.encode(
            list(queries.values()), convert_to_numpy=True, show_progress_bar=True
        )
        results = {}
        for qid, query_embedding in zip(queries.keys(), query_embeddings):
            scores = np.dot(self.corpus_embeddings, query_embedding)
            top_indices = np.argsort(scores)[-top_k:][::-1]
            results[qid] = [(self.agent_ids[idx], float(scores[idx])) for idx in top_indices]
        return results
