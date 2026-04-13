from typing import Dict, List, Tuple
from pathlib import Path
import json
import os
import pandas as pd
import logging

from asbench.generation.utils import generic_jsonl_load

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def query_indexing(db_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Indexing configuration for queries.
    source --> JSONL files {query_id: query_text}.
    
    :returns: ids and prepared documents (queries)
    """
    queries_dict = generic_jsonl_load(db_path)
    query_ids = list(queries_dict.keys())
    query_texts = [query_text for query_text, *_ in queries_dict.values()]
    return query_ids, query_texts

def artifact_indexing(db_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Indexing configuration for artifacts (e.g., videos, images, websites).
    source --> JSONL files {url: metadata}.
    
    :returns: ids (urls) and prepared documents (metadata)
    """
    artifacts_dict = pd.read_csv(db_path)
    urls = artifacts_dict["url"].tolist()
    metadata = artifacts_dict["metadata"].tolist()
    return urls, metadata

def agentbase_indexing(db_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    We follow ToolBench's indexing approach for AgentBase
        1. Simple flat string to capture important fields (removed ids and misc)
        2. Reorder columns so that high-priority fields (name, description, category) appear first.
        3. Use field names as prefixes to more context to the model in subsequent (lower priority) fields.

    :returns: ids and prepared documents
    """
    agents_df = pd.read_csv(db_path)
    agent_ids = agents_df["agent_id"]
    agents_df.drop(columns=["agent_id", "platform_id", "misc"], inplace=True)

    high_priority_cols = ["agent_name", "agent_description", "agent_category"]
    low_priority_cols = [col for col in agents_df.columns if col not in high_priority_cols]

    documents = agents_df.apply(
        lambda row: ", ".join(
            [f"{row[col]}" for col in high_priority_cols if pd.notna(row[col])] +
            [f"{col}: {row[col]}" for col in low_priority_cols if pd.notna(row[col])]
        ),
        axis=1,
    ).tolist()
    return agent_ids, documents


# ---------------------------------------------------------------------------
# Prepare documents
# ---------------------------------------------------------------------------

def load_documents( # aka naive indexing (description-only)
    db_path: str, columns=["agent_name", "agent_description"]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Loads documents (for sparse and dense models) by concatenating all column fields
    :returns: ids and prepared documents
    """
    agents_df = pd.read_csv(db_path)
    agent_ids = agents_df["agent_id"]  # keep agent IDs (mapping back after retrieval)
    documents = agents_df[columns].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    return agent_ids, documents

def load_queries(queries_path: str) -> Dict[str, str]:
    with open(queries_path) as json_file:
        data = json.load(json_file)
    return data

def tokenise(doc: str) -> List[str]:
    return doc.lower().split()
