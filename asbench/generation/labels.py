# Query Relevance Judgments (label) Generators

from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional
from collections import defaultdict
import logging

from asbench.generation.utils import (
    Tasks, TaskLabels,
    TaskJudgedLabels, TaskResponses,
)

from asbench.generation.judges.base import BaseLLMJudge
from asbench.retrieval.models.hybrid import HybridRetriever

logger = logging.getLogger(__name__)


class BaseLabelGenerator(ABC):
    def __init__(self):
        self.labels = {}  # dict[str, dict[str, int]], {"query_id": {"agent_id": relevance_score}}

    @abstractmethod
    def generate(self) -> TaskLabels:
        pass


class BinaryLabelGenerator(BaseLabelGenerator):
    """Binary Label Generator"""

    def __init__(self):
        self.labels = {}

    def generate(
        self, 
        judged_labels: TaskJudgedLabels,
        threshold: int
    ) -> TaskLabels:
        self.labels = {}
        for qid, labels in judged_labels.items():
            self.labels[qid] = {
                agent_id: 1
                for agent_id, (score, _) in labels.items()
                if score >= threshold
            }
        return self.labels

class MultiBinaryLabelGenerator(BaseLabelGenerator):
    def generate(
        self,
        judged_labels: dict[str, TaskJudgedLabels],
        threshold: int
    ) -> dict[str, TaskLabels]:
        self.labels = {}
        for qid, subtasks in judged_labels.items():
            self.labels[qid] = {}
            for subtask_id, labels in subtasks.items():
                self.labels[qid][subtask_id] = {
                    agent_id: 1
                    for agent_id, (score, _) in labels.items()
                    if score >= threshold
                }
        return self.labels


class RankingLabelGenerator:
    def __init__(self):
        self.labels = {}  # dict[str, dict[str, int]], {"query_id": {"agent_id": relevance_score}}
    
    @abstractmethod
    def generate(
        self,
        multi_judged_labels: dict[str, TaskJudgedLabels],
        tasks: Optional[Tasks] = None,
        multi_responses: Optional[dict[str, TaskResponses]] = None,
    ) -> TaskLabels:
        pass

class FineGrainRankingLabelGenerator(RankingLabelGenerator):
    def __init__(
            self, 
            n_candidates: int,
            relevance_threshold: int, 
    ):
        super().__init__()
        self.n_candidates = n_candidates
        self.relevance_threshold = relevance_threshold 

    """ Fine-grain ranking label generator that assigns relevance labels based on subtask completion rate """
    def generate(
        self,
        multi_judged_labels: dict[str, TaskJudgedLabels],
        tasks: Optional[Tasks] = None,
        multi_responses: Optional[dict[str, TaskResponses]] = None,
    ) -> TaskLabels:
        self.labels = {}

        # calculate agent scores
        agent_scores = {}

        for iid, subtasks in multi_judged_labels.items():
            agent_scores[iid] = defaultdict(int)
            for _, labels in subtasks.items():
                for agent_id, (score, _) in labels.items():
                    agent_scores[iid][agent_id] += 1 if score >= self.relevance_threshold else 0

        # assign labels
        for iid, subtasks in multi_judged_labels.items():
            self.labels[iid] = {
                agent_id: score
                for agent_id, score in agent_scores[iid].items() if score > 0
            }
            # sort by score
            self.labels[iid] = dict(sorted(self.labels[iid].items(), key=lambda item: item[1], reverse=True))

        # select top k agents from final sorted list
        for iid, agent_scores in self.labels.items():
            self.labels[iid] = dict(list(agent_scores.items())[:self.n_candidates])
        return self.labels

class GoldenRankingLabelGenerator(FineGrainRankingLabelGenerator):
    """ Golden ranking label generator extends fine-grain labels by discounting the relevance score for semantically misaligned documents """
    def __init__(
            self,
            n_candidates: int,
            relevance_threshold: int,
            agentbase: pd.DataFrame,
            consistency_judge: BaseLLMJudge,
    ):
        self.labels = {}
        self.agentbase = agentbase
        self.consistency_judge = consistency_judge
        self.finegrain_generator = FineGrainRankingLabelGenerator(n_candidates=n_candidates, relevance_threshold=relevance_threshold)

    def generate(
            self,
            multi_judged_labels: dict[str, TaskJudgedLabels],
            tasks: Tasks,
            multi_responses: dict[str, TaskResponses],
    ) -> TaskLabels:
        self.labels = {}

        # 1. get fine-grain labels
        finegrain_labels = self.finegrain_generator.generate(tasks=tasks, multi_judged_labels=multi_judged_labels)

        # 2. calculate consistency scores (binary)
        agent_docs = { # dict[str, dict[str, str]] of {intent_id: {agent_id: agent_description}}
            iid: {
                agent_id: self.agentbase.loc[self.agentbase["agent_id"] == agent_id, "agent_description"].values[0]
                for agent_id in labels.keys()
            } for iid, labels in finegrain_labels.items()
        }
        agent_responses = { # dict[str, dict[str, list[str]]] of {intent_id: {agent_id: [responses]}}
            iid: {
                agent_id: [labels[agent_id] for labels in multi_responses[iid].values() if labels.get(agent_id, None)]
                for agent_id in labels.keys()
            } for iid, labels in finegrain_labels.items()
        }
        consistency_scores = self.consistency_judge.judge(agent_docs, agent_responses)

        # 3. discount fine-grain scores based on consistency (half per score)
        discount = 0.5
        for iid, labels in finegrain_labels.items():
            self.labels[iid] = {
                agent_id: round(score - discount, 3) if consistency_scores[iid].get(agent_id, [0])[0] == 0 else score
                for agent_id, score in labels.items()
            }
            # sort by score
            self.labels[iid] = dict(sorted(self.labels[iid].items(), key=lambda item: item[1], reverse=True))

        # select top k agents from final sorted list
        for iid, agent_scores in self.labels.items():
            self.labels[iid] = dict(list(agent_scores.items())[:self.n_candidates])
        return self.labels
