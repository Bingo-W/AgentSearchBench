from typing import List, Tuple, Dict, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from asbench.retrieval.base import BaseRetriever
from asbench.retrieval.utils import tokenise


class BM25Retriever(BaseRetriever):
    """BM25 sparse retrieval for AgentBase."""

    def __init__(
        self, db_path: str, index_config: Literal["naive", "v1", "artifacts", "queries"], **bm25_params
    ):
        super().__init__(db_path, index_config)
        self.bm25_params = bm25_params
        self.index = None
        self.build_index()

    def build_index(self):
        """
        Load documents and build BM25 index.
        """
        self.agent_ids, self.corpus = self.indexing_func[self.index_config]()
        tokenised_docs = [tokenise(doc) for doc in self.corpus]
        self.index = BM25Okapi(tokenised_docs, **self.bm25_params)

    def sim(self, d1: str, d2: str) -> float:
        tokenised_d1 = tokenise(d1)
        tokenised_d2 = tokenise(d2)
        self.index = BM25Okapi([tokenised_d1], **self.bm25_params)
        return self.index.get_scores(tokenised_d2)

    def retrieve(self, query: str, top_k: int = 10):
        """
        Retrieve top-k agents using BM25.
        """
        tokenized_query = tokenise(query)
        scores = self.index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(self.agent_ids[idx], float(scores[idx])) for idx in top_indices]

    def retrieve_all(self, query: str):
        tokenized_query = tokenise(query)
        scores = self.index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1]
        return [(self.agent_ids[idx], float(scores[idx])) for idx in top_indices]