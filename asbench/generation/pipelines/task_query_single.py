# Single-Agent Task Query Generator

import logging
from typing import Optional
import pandas as pd

from asbench.generation.utils import (
    Tasks, TaskLabels, 
    TaskCandidates, TaskJudgedLabels, TaskResponses,
    LLMParams, RetrieverParams, ExecutorParams
)

from asbench.generation.pipelines.base import BasePipeline

from asbench.generation.inference import LLMInference
from asbench.generation.generators.multi_doc import MultiDocGenerator
from asbench.generation.generators.context import ContextGenerator
from asbench.generation.judges.task_judge import TaskJudge
from asbench.retrieval.models.hybrid import HybridRetriever, RetrieverConfig
from asbench.generation.responses.candidate_selector import select_candidates
from asbench.generation.responses.response_generator import ResponseGenerator
from asbench.generation.judges.single_judge import SingleAgentJudge
from asbench.generation.judges.multi_judge import MultiJudge
from asbench.generation.labels import BinaryLabelGenerator

logger = logging.getLogger(__name__)


class TaskQuerySingle(BasePipeline):
    def __init__(
        self,
        gen_model: str,
        n_tasks: int,
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
        **kwargs,
    ):
        self.gen_model = gen_model
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

        # setup
        self.agentbase_df = pd.read_csv(self.agentbase)
        self.gen_llm = LLMInference(
            model_name=llm_params[gen_model][0],
            api_key=llm_params[gen_model][1],
            temperature=llm_params[gen_model][2],
        )
        selector_configs = [
            RetrieverConfig(model_name, weight=float(weight), db_path=agentbase,
                            index_config=retriever_params[model_name][1], 
                            kwargs={"model_name": retriever_params[model_name][0]} 
                            if model_name not in ["bm25"] else {})
            for model_name, weight in selector_models.items()
        ]
        self.selector = HybridRetriever(selector_configs)
        self.task_generator = MultiDocGenerator(
            corpus=self.agentbase_df,
            llm=self.gen_llm,
            generic_retriever=self.selector,
        )
        self.task_judge = TaskJudge(llm=self.gen_llm)
        self.context_generator = ContextGenerator(llm=self.gen_llm)
        self.response_generator = ResponseGenerator(
            corpus=self.agentbase_df,
            executor_params=executor_params,
            response_path=None,
        )
        judges: list[SingleAgentJudge] = [
            SingleAgentJudge(
                llm=LLMInference(
                    model_name=llm_params[judge_model][0],
                    api_key=llm_params[judge_model][1],
                    temperature=llm_params[judge_model][2],
                ),
                token_limit=judge_token_limit,
                use_golden_labels=use_golden_labels,
            ) for judge_model in judge_models
        ]
        self.response_judge = MultiJudge(judges=judges)
        self.label_generator = BinaryLabelGenerator()

    def run(self) -> tuple[
        Tasks, TaskLabels,
        TaskCandidates, TaskJudgedLabels, TaskResponses
    ]:
        tasks: Tasks = {}
        task_labels: TaskLabels = {}
        candidates: TaskCandidates = {}
        judged_labels: TaskJudgedLabels = {}
        responses: TaskResponses = {}

        # 1. Task Generation
        tasks = self.task_generator.generate_batch(n_tasks=self.n_tasks)
        judgements = self.task_judge.judge(tasks=tasks)

        self_contained     = {tid: tasks[tid] for tid, eval in judgements.items() if eval["score"] == 1}
        requires_context   = {tid: tasks[tid] for tid, eval in judgements.items() if eval["score"] == 0}
        requires_context = self.context_generator.gen_tasks_w_context(tasks=requires_context)
        tasks = {**self_contained, **requires_context}

        if not self.generate_labels:
            return tasks, task_labels

        # 2. Label Generation
        candidates = select_candidates(
            retriever=self.selector,
            tasks=tasks,
            top_k=self.n_candidates,
            single_agent=True,
        )
        responses = self.response_generator.gen_responses(
            tasks=tasks,
            candidates=candidates
        )
        judged_labels = self.response_judge.judge(
            tasks=tasks,
            responses=responses
        )
        task_labels = self.label_generator.generate(
            judged_labels=judged_labels,
            threshold=self.relevance_threshold
        )

        return tasks, task_labels, candidates, judged_labels, responses