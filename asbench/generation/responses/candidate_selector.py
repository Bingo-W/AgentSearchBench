# Proxy candidate selection for response generation

import random
import itertools
from typing import Callable

from asbench.retrieval.base import BaseRetriever
from asbench.retrieval.models.sentence_transformer import DenseRetriever
from asbench.retrieval.models.hybrid import HybridRetriever, RetrieverConfig


def select_candidates( # unified interface
    retriever: HybridRetriever,
    tasks: dict[str, tuple],
    top_k: int,
    single_agent: bool
) -> dict[str, list[str]] | dict[str, list[list[str]]]:
    
    if single_agent:
        return _select_single(retriever, tasks, top_k)
    else:
        return _select_multi(retriever, tasks, top_k)

def _select_single(
    retriever: HybridRetriever, tasks: dict[str, tuple], top_k: int
) -> dict[str, list[str]]:
    identified_candidates = {}
    
    for qid, (query_text, *_) in tasks.items():
        retrieval_results = retriever.retrieve(query_text, top_k)
        agent_ids, _ = zip(*retrieval_results)  # unzip list of (agent_id, score)
        identified_candidates[qid] = list(agent_ids)
    return identified_candidates

def _select_multi(
    retriever: HybridRetriever, tasks: dict[str, tuple], top_k: int
) -> dict[str, dict[str, list[str]]]:
    identified_candidates = {}

    for qid, (_, _, subtasks, *_) in tasks.items():
        identified_candidates[qid] = {}
        for i, subtask in enumerate(subtasks):
            retrieval_results = retriever.retrieve(subtask, top_k * 5)
            agent_ids, _ = zip(*retrieval_results)  # unzip list of (agent_id, score)
            identified_candidates[qid][f"t:{i}"] = list(agent_ids)
    return identified_candidates


# ==================== OLD VERSIONS (for reference, to be deleted) ====================
CandidateSelector = Callable[
    [
        str,
        dict[str, tuple],
        int,
        float | None,
    ],  # (data_file, queries, top_k, threshold)
    dict[str, list[str]],
]

CandidateSelectorMulti = Callable[
    [
        str,
        dict[str, tuple],
        int,
        int,
        float | None,
    ],  # (data_file, subtasks, top_k, top_l, threshold)
    dict[str, list[list[str]]],
]

def _select_candidates_single_old(
    data_file: str, queries: dict[str, tuple], top_k: int, threshold: float | None
) -> dict[str, list[str]]:
    retriever = DenseRetriever("BAAI/bge-large-en-v1.5", data_file, index_config="v1")

    identified_candidates = {}
    for qid, (query_text, _) in queries.items():
        # NOTE small manipulation to achieve top_k given a threshold
        target_top_k = top_k * 5
        retrieval_results = retriever.retrieve(query_text, target_top_k)

        # list[tuple[str, float]], (agent_id, score)
        # filter by >threshold for cos sim
        filtered_ids = [
            agent_id
            for agent_id, score in retrieval_results
            if (score >= threshold if threshold else True)
        ][:top_k]
        identified_candidates[qid] = filtered_ids

    return identified_candidates

def _select_candidates_multi_old(
        data_file: str, queries: dict[str, tuple], top_k: int, threshold: float | None
) -> dict[str, dict[str, list[str]]]:
    # given m subtasks (that belong to one query q), retrieve top_k candidates per each subtask.
    retriever = DenseRetriever("BAAI/bge-large-en-v1.5", data_file, index_config="v1")

    identified_candidates = {}
    for qid, (_, _, subtasks, _) in queries.items():
        identified_candidates[qid] = {}
        for i, subtask in enumerate(subtasks):
            retrieval_results = retriever.retrieve(subtask, top_k * 5)

            filtered_ids = [
                agent_id
                for agent_id, score in retrieval_results
                if (score >= threshold if threshold else True)
            ][:top_k]
            identified_candidates[qid][f"t:{i}"] = filtered_ids
    return identified_candidates

def _select_candidates_casual_multi(
    data_file: str, queries: dict[str, tuple], top_k: int, top_l: int, threshold: float | None
) -> dict[str, list[list[str]]]:
    # given a m subtasks (that belong to one query q), retrieve l candidates per subtaks, and uniformly sample combinations from l^m.
    retriever = DenseRetriever("BAAI/bge-large-en-v1.5", data_file, index_config="v1")

    identified_combinations = {}
    for qid, (_, _, subtasks, _) in queries.items():
        retrieved_candidates = []
        for subtask in subtasks:
            retrieval_results = retriever.retrieve(subtask, top_l * 5)

            filtered_ids = [
                agent_id
                for agent_id, score in retrieval_results
                if (score >= threshold if threshold else True)
            ][:top_l]
            retrieved_candidates.append(filtered_ids)

        # uniformly sample top_k combinations from the space of l^m combinations
        all_combinations = list(itertools.product(*retrieved_candidates))
        identified_combinations[qid] = [
            list(combo)
            for combo in random.sample(all_combinations, min(top_k, len(all_combinations)))
        ]

    return identified_combinations

def select_candidates_experimental(
    data_file: str, queries: dict[str, tuple], top_k: int, threshold: float | None = None
) -> dict[str, list[tuple[str, float]]]:
    retriever = DenseRetriever("BAAI/bge-large-en-v1.5", data_file, index_config="v1")

    identified_candidates = {}
    for qid, (query_text, _) in queries.items():
        target_top_k = top_k * 5
        retrieval_results = retriever.retrieve(query_text, target_top_k)

        filtered_ids = [
            (agent_id, score)
            for agent_id, score in retrieval_results
            if (score >= threshold if threshold else True)
        ][:top_k]
        identified_candidates[qid] = filtered_ids

    return identified_candidates

def select_candidates_hybrid_experimental(
        retriever_configs: list[RetrieverConfig], 
        queries: dict[str, tuple],
        top_k: int
) -> dict[str, list[tuple[str, float]]]:
    retriever = HybridRetriever(configs=retriever_configs)
    
    identified_candidates = {}
    for qid, (query_text, _) in queries.items():
        retrieval_results = retriever.retrieve(query_text, top_k)
        identified_candidates[qid] = retrieval_results
    return identified_candidates