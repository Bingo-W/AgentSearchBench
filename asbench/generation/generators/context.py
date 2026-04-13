# Task Context Generator

import logging
from pathlib import Path
import pandas as pd
from typing import List

from asbench.retrieval.models.bm25 import BM25Retriever
from asbench.generation.inference import LLMInference
from asbench.generation.prompts import CONTEXT_PROMPT
from asbench.generation.utils import extract_json


logger = logging.getLogger(__name__)

class ContextGenerator:
    """
    Modifies tasks to include necessary context, making them answerable without personal/private references.

    a) false flag or needs additional data, b) needs real reference (link)
    For a) we prompt the LLM generate the necessary additional synthetic data/context (if any)
    For b) we sample relevant public urls/links (e.g., websites, videos, images) and prompt the LLM to modify the task to include one as a reference.
    """

    def __init__(
        self,
        llm: LLMInference
    ):
        self.llm = llm
        self.llm_model = llm.model_name
        self.artifact_types: List[str] = ["video", "image", "web"]
        self.base_dir = Path(__file__).resolve().parent.parent.parent.parent / "asbench/data/artifacts"
        assert self._check_artifact_files(self.artifact_types, self.base_dir), f"Artifact files missing in {self.base_dir} for types: {self.artifact_types}"
        self._prep_retrievers()

    def gen_tasks_w_context(self, tasks: dict[str, tuple]) -> dict[str, tuple[str, str]]:
        """
        returns: {qid: (modified_task_text, ref_agent_id)}
        """
        tasks_w_context = {}
        for qid, (task_text, agent_id) in tasks.items():
            prompt = CONTEXT_PROMPT.get_messages(task=task_text, retrieved_artifacts=self._get_artifact(task_text, top_k=1))
            llm_response = self.llm.invoke(prompt).content
            modified_task = extract_json(llm_response).get("task", task_text)
            tasks_w_context[qid] = (modified_task, agent_id)
            logger.info(f"Generated context for task {qid}")
        return tasks_w_context
    
    def _get_artifact(self, task: str, top_k: int=1) -> dict[str, list[tuple[str, str]]]:
        retrievers = [getattr(self, f"{atype}_retriever", None) for atype in self.artifact_types]
        links = [retriever.retrieve(task, top_k) if retriever else [] for retriever in retrievers]
        metadata = [self._get_metadata(atype, link) for atype, link in zip(self.artifact_types, links)]
        return {atype: links for atype, links in zip(self.artifact_types, metadata)}
    
    def _prep_retrievers(self):
        # load paths
        paths: list[str] = [str(self.base_dir / f"{atype}.csv") for atype in self.artifact_types]
        for atype, path in zip(self.artifact_types, paths):
            setattr(self, f"{atype}_links", pd.read_csv(path))

        # load retrievers (default: sparse)
        for atype, path in zip(self.artifact_types, paths):
            retriever = BM25Retriever(path, index_config="artifacts")
            setattr(self, f"{atype}_retriever", retriever)

    def _get_metadata(self, link_type: str, link: str) -> str:
        attr = getattr(self, f"{link_type}_links", None)
        if attr is None:
            logger.warning(f"No links found for type: {link_type}, must be one of {self.artifact_types}")
            return None
        
        # get metadata of given link
        metadata = attr[attr["url"] == link[0][0]]["metadata"].values
        return metadata[0] if len(metadata) > 0 else None

    @staticmethod
    def _check_artifact_files(artifact_types: List[str], base_dir) -> bool:
        for atype in artifact_types:
            path = base_dir / f"{atype}.csv"
            if not path.exists():
                return False
        return True
