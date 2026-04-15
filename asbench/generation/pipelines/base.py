# Abstract base class for task pipelines

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from asbench.generation.utils import (
    Tasks, TaskLabels,
    TaskCandidates, TaskJudgedLabels, TaskResponses,
    generic_jsonl_save
)

logger = logging.getLogger(__name__)


class BasePipeline(ABC):
    @abstractmethod
    def run(self) -> tuple[
        Tasks, TaskLabels,
        TaskCandidates, TaskJudgedLabels, TaskResponses
    ]:
        """
        returns: tasks and their labels 
        """
        pass

    @staticmethod
    def save(
        tasks: Tasks,
        task_labels: TaskLabels,
        experiment_name: str,
    ):
        output_dir = Path(__file__).parent.parent.parent.parent / f"outputs/{experiment_name}"
        generic_jsonl_save(tasks, f"{output_dir}/tasks.jsonl")
        generic_jsonl_save(task_labels, f"{output_dir}/task_labels.jsonl")

    @staticmethod
    def save_debug(
        task_rank_labels: TaskLabels,
        task_labels_multi: dict[str, TaskLabels],
        candidates: TaskCandidates,
        judged_labels: TaskJudgedLabels,
        responses: TaskResponses,
        experiment_name: str,
    ):
        output_dir = Path(__file__).parent.parent.parent.parent / f"outputs/{experiment_name}/debug"
        
        if task_rank_labels:
            generic_jsonl_save(task_rank_labels, f"{output_dir}/task_rank_labels.jsonl")
        if task_labels_multi:
            generic_jsonl_save(task_labels_multi, f"{output_dir}/task_labels_multi.jsonl")
        if candidates:
            generic_jsonl_save(candidates, f"{output_dir}/candidates.jsonl")
        if judged_labels:
            generic_jsonl_save(judged_labels, f"{output_dir}/judged_labels.jsonl")
        if responses:
            generic_jsonl_save(responses, f"{output_dir}/responses.jsonl")