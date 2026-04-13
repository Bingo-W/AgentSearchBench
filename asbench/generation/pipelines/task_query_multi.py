# Multi-Agent Task Query Generator

import logging
from typing import Optional
import random
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as transformers_logging

from asbench.generation.utils import (
    Tasks, TaskLabels,
    TaskCandidates, TaskJudgedLabels, TaskResponses,
    LLMParams, RetrieverParams, ExecutorParams,
    generic_jsonl_load, extract_json
)

from asbench.generation.pipelines.base import BasePipeline

from asbench.generation.inference import LLMInference
from asbench.generation.generators.multi_doc import MultiDocGenerator
from asbench.generation.generators.context import ContextGenerator
from asbench.generation.judges.task_judge import TaskJudge
from asbench.retrieval.models.hybrid import HybridRetriever, RetrieverConfig
from asbench.retrieval.models.bm25 import BM25Retriever
from asbench.generation.labels import FineGrainRankingLabelGenerator, GoldenRankingLabelGenerator
from asbench.generation.judges.consistency_judge import DocConsistencyJudge
from asbench.generation.prompts import MULTI_AGENT_PROMPT, JUDGE_DOC_CONSISTENCY_PROMPT

logger = logging.getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
transformers_logging.set_verbosity_error()


class TaskQueryMulti(BasePipeline):
    def __init__(
        self,
        gen_model: str,
        nli_model: str,
        n_tasks: int,
        n_agents: list[int],
        n_candidates: int,
        selector_models: dict[str, float],
        active_platforms: list[str],
        judge_models: list[str],
        use_golden_labels: bool,
        judge_token_limit: int,
        qrel_type: str,
        rank_label_type: str,
        relevance_threshold: int,
        agentbase: str,
        llm_params: LLMParams,
        retriever_params: RetrieverParams,
        executor_params: ExecutorParams,
        generate_labels: bool,
        single_agent_paths: dict[str, str],
        **kwargs,
    ):
        self.gen_model = gen_model
        self.nli_model = nli_model
        self.n_tasks = n_tasks
        self.n_candidates = n_candidates
        self.selector_models = selector_models
        self.activate_platforms = active_platforms
        self.judge_models = judge_models
        self.use_golden_labels = use_golden_labels
        self.judge_token_limit = judge_token_limit
        self.qrel_type = qrel_type
        self.rank_label_type = rank_label_type
        self.relevance_threshold = relevance_threshold
        self.agentbase = agentbase
        self.generate_labels = generate_labels
        self.n_subtasks: list[int] = n_agents
        # needed for re-construction
        self.single_agent_tasks: Tasks = generic_jsonl_load(single_agent_paths["tasks"])
        self.single_agent_task_labels: TaskLabels = generic_jsonl_load(single_agent_paths["labels"])
        self.single_agent_responses: TaskResponses = generic_jsonl_load(single_agent_paths["responses"])
        self.single_agent_judged_labels: TaskJudgedLabels = generic_jsonl_load(single_agent_paths["judged_labels"])

        # setup
        self.nli_tokeniser = AutoTokenizer.from_pretrained(nli_model)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model).to(device)
        self.agentbase_df = pd.read_csv(self.agentbase)
        self.gen_llm = LLMInference(
            model_name=llm_params[gen_model][0],
            api_key=llm_params[gen_model][1],
            temperature=llm_params[gen_model][2],
        )
        task_selector_configs = [
            RetrieverConfig(model_name, weight=float(weight), db_path=single_agent_paths["tasks"],
                            index_config="queries", 
                            kwargs={"model_name": retriever_params[model_name][0]} 
                            if model_name not in ["bm25"] else {})
            for model_name, weight in selector_models.items()
        ]
        self.task_selector = HybridRetriever(task_selector_configs)
        self.task_selector = BM25Retriever(db_path=single_agent_paths["tasks"], index_config="queries")  # ablation with BM25 retriever only
        self.task_generator = MultiDocGenerator(
            corpus=self.agentbase_df,
            llm=self.gen_llm,
            generic_retriever=self.task_selector,
        )
        self.task_judge = TaskJudge(llm=self.gen_llm)
        self.context_generator = ContextGenerator(llm=self.gen_llm)
        if rank_label_type == "finegrain":
            self.rank_label_generator = FineGrainRankingLabelGenerator(
                n_candidates=n_candidates,
                relevance_threshold=relevance_threshold,
            )
        elif rank_label_type == "golden":
            doc_consistency_judge = DocConsistencyJudge(
                llm=self.gen_llm,
                token_limit=judge_token_limit,
                prompt=JUDGE_DOC_CONSISTENCY_PROMPT,
            )
            self.rank_label_generator = GoldenRankingLabelGenerator(
                n_candidates=n_candidates,
                relevance_threshold=relevance_threshold,
                agentbase=self.agentbase_df,
                consistency_judge=doc_consistency_judge,
            )
        else:
            raise ValueError(f"Unsupported rank label type: {rank_label_type}")

    def run(self) -> tuple[
        Tasks, TaskLabels, TaskLabels, dict[str, TaskLabels]
    ]:
        tasks: Tasks = {}
        task_labels: TaskLabels = {}
        task_rank_labels: TaskLabels = {}
        task_labels_multi: dict[str, TaskLabels] = {}
        task_responses_multi: dict[str, TaskResponses] = {}
        task_judged_labels_multi: dict[str, TaskJudgedLabels] = {}

        # 1. Task Generation
        while len(tasks) < self.n_tasks:
            subtasks: Tasks = self._sample_tasks()
            multi_agent_task = self._gen_multi_agent_task(subtasks)
            if multi_agent_task and self._is_grounded(multi_agent_task, subtasks):
                # construct tasks
                qid = f"q:{len(tasks)}"
                agent_ids = list(set(agent_id for _, (_, agent_ids) in subtasks.items() for agent_id in agent_ids))
                subtask_texts = [task for _, (task, *_) in subtasks.items()]
                subtask_qids = list(subtasks.keys())
                tasks[qid] = (multi_agent_task, agent_ids, subtask_texts, subtask_qids)

                # construct labels
                if self.generate_labels:
                    task_responses_multi[qid] = {}
                    task_judged_labels_multi[qid] = {}
                    task_labels[qid] = {}
                    task_labels_multi[qid] = {}
                    
                    for tid, _ in subtasks.items():
                        task_responses_multi[qid][tid] = self.single_agent_responses[tid]
                        task_judged_labels_multi[qid][tid] = self.single_agent_judged_labels[tid]
                        task_labels[qid].update(self.single_agent_task_labels[tid])
                        task_labels_multi[qid][tid] = self.single_agent_task_labels[tid]

                logger.info(f"Constructed multi-agent task query {qid} with {len(subtask_qids)} subtasks")
            else:
                logger.info(f"Discarded multi-agent task query due to failed grounding check")

        # 2. Label Generation (not needed, inherited from single-agent tasks)
        if self.generate_labels:
            task_rank_labels = self.rank_label_generator.generate(
                multi_judged_labels=task_judged_labels_multi,
                tasks=tasks,
                multi_responses=task_responses_multi,
            )
        return tasks, task_labels, task_rank_labels, task_labels_multi   

    def _sample_tasks(self) -> dict[str, tuple[str, list[str]]]:
        # sample one anchor task, then retrieve similar (but not redundant) tasks based on the anchor task
        n = random.choice(self.n_subtasks)
        anchor_tid = random.choice(list(self.single_agent_tasks.keys()))
        redundancy_k = 20 # skip top redundant k
        res = self.task_selector.retrieve(self.single_agent_tasks[anchor_tid][0], top_k=(redundancy_k+n)*5) # oversample
        similar_tids = [task_id for task_id, _ in res if task_id in self.single_agent_tasks and task_id != anchor_tid and task_id in self.single_agent_tasks]
        similar_tids = similar_tids[redundancy_k:redundancy_k + n]  # take top n non-redundant similar tasks
        sampled_tasks = {qid: self.single_agent_tasks[qid] for qid in similar_tids}
        return sampled_tasks

    def _gen_multi_agent_task(self, subtasks: Tasks) -> str:
        subtasks = [task for _, (task, *_) in subtasks.items()]
        prompt = MULTI_AGENT_PROMPT.get_messages(
            subtasks=subtasks,
        )

        try:
            response = self.gen_llm.invoke(prompt).content
            extracted = extract_json(response)
            query = extracted["query"]
            return query
        except Exception as e:
            logger.error(f"Error generating multi-agent query for subtasks {subtasks}: {e}")
            return None

    def _is_grounded(self, task: str, subtasks: dict[str, tuple[str, list[str]]]) -> bool:
        # Is each subtask grounded on the multi-agent tasks scope?
        # NOTE: subtasks must not deviate from the multi-agent tasks scope.
        subtasks = [task for _, (task, *_) in subtasks.items()]
        for subtask in subtasks:
            if not self._is_entailed(task, subtask):
                return False
        return True
    
    def _is_entailed(self, premise: str, hypothesis: str) -> bool:
        input = self.nli_tokeniser(premise, hypothesis, truncation=False, return_tensors="pt")
        output = self.nli_model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        return label_names[np.argmax(list(prediction.values()))] == "entailment"
