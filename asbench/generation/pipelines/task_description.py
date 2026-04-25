# Task Description Generator

import logging
from typing import Optional
import pandas as pd
import numpy as np

from asbench.generation.utils import (
    Tasks, TaskLabels, 
    TaskCandidates, TaskJudgedLabels, TaskResponses,
    LLMParams, RetrieverParams, ExecutorParams,
    extract_json
)
from asbench.generation.prompts import TASK_DESCRIPTION_PROMPT, JUDGE_DOC_CONSISTENCY_PROMPT

from asbench.generation.pipelines.base import BasePipeline

from asbench.generation.inference import LLMInference
from asbench.generation.generators.multi_doc import MultiDocGenerator
from asbench.generation.generators.context import ContextGenerator
from asbench.generation.judges.task_judge import TaskJudge
from asbench.generation.generators.rubric import DescriptionRubric
from asbench.retrieval.models.hybrid import HybridRetriever, RetrieverConfig
from asbench.generation.responses.candidate_selector import select_candidates
from asbench.generation.responses.response_generator import ResponseGenerator
from asbench.generation.judges.single_judge import SingleAgentJudge
from asbench.generation.judges.multi_judge import MultiJudge
from asbench.generation.labels import BinaryLabelGenerator, FineGrainRankingLabelGenerator, GoldenRankingLabelGenerator
from asbench.generation.judges.consistency_judge import DocConsistencyJudge

logger = logging.getLogger(__name__)


class TaskDescription(BasePipeline):
    def __init__(
        self,
        gen_model: str,
        n_tasks: int,
        n_subtasks: int,
        sample_pool_size: int,
        coherence_threshold: float,
        n_criteria: int,
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
        self.n_subtasks = n_subtasks
        self.sample_pool_size = sample_pool_size
        self.coherence_threshold = coherence_threshold
        self.n_criteria = n_criteria
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
        self.rubric_generator = DescriptionRubric(llm=self.gen_llm)
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
        Tasks, TaskLabels, TaskLabels, dict[str, TaskLabels],
        dict[str, TaskCandidates], dict[str, TaskJudgedLabels], dict[str, TaskResponses]
    ]:
        tasks: Tasks = {}
        task_labels: TaskLabels = {}
        task_rank_labels: TaskLabels = {}
        task_labels_multi: dict[str, TaskLabels] = {}
        candidates_multi: dict[str, TaskCandidates] = {}
        responses_multi: dict[str, TaskResponses] = {}
        judged_labels_multi: dict[str, TaskJudgedLabels] = {}

        # 1. Task Generation
        while len(tasks) < self.n_tasks:
            iid = f"d:{len(tasks)}"
            # A. Generate a Pool of Candidate Task Query
            sampled_agents = self._sample_sim_agents(n_agents=self.sample_pool_size)
            sampled_task_query = self._gen_task_query(ref_agents=sampled_agents)
            logger.info(f"Generated {len(sampled_task_query)} task query for description {iid}", extra={"importance": "high"})
            # B. Synthesize a Coherent Subset of Task Query
            coherent_task_query = self._coherent_subset(sampled_task_query, self.coherence_threshold)
            task_description = self._gen_task_description(coherent_task_query)
            # C. Evaluate and Score the Proxy Task Query using a Rubric
            rubric = self.rubric_generator.generate_rubric(task_description, n_criteria=self.n_criteria)
            assigned_labels = self.rubric_generator.evaluate_tasks(rubric, sampled_task_query)
            task_scores = self.rubric_generator._score_tasks(assigned_labels, sampled_task_query)
            selected_task_query = self.rubric_generator.select_tasks(self.n_subtasks, assigned_labels, task_scores, sampled_task_query) 
            logger.info(f"Selected {len(selected_task_query)} executable tasks for task {iid} based on rubric evaluation", extra={"importance": "high"})

            # assign
            debug_agent_ids = [agent["agent_id"] for agent in sampled_agents]
            debug_task_query = [task_query[0] for task_query in selected_task_query.values()]
            tasks[iid] = (task_description, debug_agent_ids, debug_task_query, rubric)

            # 2. Label Generation
            if self.generate_labels:
                task_query_candidates = select_candidates(
                    retriever=self.selector,
                    tasks=selected_task_query,
                    top_k=self.n_candidates,
                    single_agent=True,
                )
                task_query_responses = self.response_generator.gen_responses(
                    tasks=selected_task_query,
                    candidates=task_query_candidates
                )
                task_query_judged_labels = self.response_judge.judge(
                    tasks=selected_task_query,
                    responses=task_query_responses,
                    rubric=rubric,
                )
                task_query_labels: TaskLabels = self.label_generator.generate(
                    judged_labels=task_query_judged_labels,
                    threshold=self.relevance_threshold
                )

                # assign
                candidates_multi[iid] = task_query_candidates
                responses_multi[iid] = task_query_responses
                judged_labels_multi[iid] = task_query_judged_labels
                task_labels[iid] = {k: max(d.get(k, 0) for d in task_query_labels.values()) 
                                    for k in {k for d in task_query_labels.values() for k in d}}
                task_labels_multi[iid] = task_query_labels
        
        if self.generate_labels:
            task_rank_labels = self.rank_label_generator.generate(
                multi_judged_labels=judged_labels_multi,
                tasks=tasks,
                multi_responses=responses_multi,
            )
        return tasks, task_labels, task_rank_labels, candidates_multi, judged_labels_multi, responses_multi

    def _sample_sim_agents(self, n_agents: int) -> list[dict]:
        samples = []
        anchor = self.agentbase_df.sample(n=1, replace=False).to_dict(orient="records")[0]
        samples.append(anchor) # randomly sample the anchor agent

        # retrieve similar agents based on embedding similarity (retriever component)
        retrieved = self.selector.retrieve(anchor["agent_description"], top_k=n_agents)
        sampled_agents = [
            self.agentbase_df[self.agentbase_df["agent_id"] == sample_id].to_dict(orient="records")[0]
            for sample_id, _ in retrieved
            if sample_id != anchor["agent_id"]
        ]
        samples.extend(sampled_agents)
        return samples

    def _gen_task_query(self, ref_agents: list[dict]) -> Tasks:
        # Single Agent Task Query Helper
        tasks: Tasks = self.task_generator.generate_batch_from(ref_agents=ref_agents)
        judgements = self.task_judge.judge(tasks=tasks)
        self_contained     = {tid: tasks[tid] for tid, eval in judgements.items() if eval["score"] == 1}
        requires_context   = {tid: tasks[tid] for tid, eval in judgements.items() if eval["score"] == 0}
        requires_context = self.context_generator.gen_tasks_w_context(tasks=requires_context)
        tasks = {**self_contained, **requires_context}
        return tasks

    def _coherent_subset(self, candidate_task_query: Tasks, threshold: int) -> Tasks:
        # 1. get normalised query embeddings
        query_embeddings = {}
        query_texts = [query for query, _ in candidate_task_query.values()]
        for qid, query_text in zip(candidate_task_query.keys(), query_texts):
            query_embeddings[qid] = self.selector.encode(query_text)

        # 2. compute normalised centroid
        centroid = np.mean(list(query_embeddings.values()), axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # 3. compute cosine similarity to centroid and filter based on threshold
        coherent_task_query = {}
        for qid, embedding in query_embeddings.items():
            cosine_sim = np.dot(embedding, centroid)
            if cosine_sim >= threshold:
                coherent_task_query[qid] = candidate_task_query[qid]
        return coherent_task_query

    def _gen_task_description(self, exec_tasks: dict[str, tuple]) -> str:
        # generate an abstract task description around the given executable tasks
        prompt = TASK_DESCRIPTION_PROMPT.get_messages(specific_queries=exec_tasks)
        high_level_task = ""
        try:
            response = self.gen_llm.invoke(prompt).content
            extracted = extract_json(response)

            high_level_task = extracted.get("high_level_task", "")
            if not high_level_task:
                logger.warning(f"No task description found in LLM response")
            
        except Exception as e:
            logger.error(f"Error generating task description for {len(exec_tasks)} tasks: {e}")
        return high_level_task