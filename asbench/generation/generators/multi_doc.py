# Single-Agent Multi-Document (default) Task Query Generator

import logging
import numpy as np
import pandas as pd

from asbench.generation.inference import LLMInference
from asbench.generation.generators.base import BaseGenerator
from asbench.retrieval.base import BaseRetriever
from asbench.generation.prompts import MULTI_DOC_PROMPT
from asbench.generation.utils import extract_json


logger = logging.getLogger(__name__)

class MultiDocGenerator(BaseGenerator):
    def __init__(
        self,
        corpus: pd.DataFrame,
        llm: LLMInference,
        generic_retriever: BaseRetriever,
        tasks_per_call=1,
        sample_range=(3, 5),
    ):
        """
        sample_range (inclusive): range of number of documents to sample for multi-document query generation
        """
        super().__init__(corpus, llm, tasks_per_call)
        self.sample_values = np.arange(sample_range[0], sample_range[1] + 1)
        self.retriever = generic_retriever

    def generate(self, ref_agent: dict) -> list[dict]:
        tasks = {}
        # retriever-based sampling (randomly sample 1, then retrieve n_samples similar agents)
        target_agent = ref_agent
        n_samples = np.random.choice(self.sample_values)
        samples = self.retriever.retrieve(target_agent["agent_description"], top_k=n_samples)
        sampled_agents = [
            self.corpus[self.corpus["agent_id"] == sample_id].to_dict(orient="records")[0]
            for sample_id, _ in samples
        ]
        
        prompt = MULTI_DOC_PROMPT.get_messages(
            tasks_per_call=self.tasks_per_call,
            target_agent=target_agent,
            auxiliary_agents=sampled_agents
        )
        try:
            response = self.llm.invoke(prompt).content
            extracted = extract_json(response)
            extracted = {
                k: v
                for i, (k, v) in enumerate(extracted.items())
                if i < self.tasks_per_call
            }

            for i, query in enumerate(extracted.values()):
                tasks[f"t:{self.counter + i}"] = (query, [target_agent["agent_id"]])
            self.counter += len(extracted.values())
            logger.info(f"Generated {len(extracted.values())} multi-doc query(s) from {len(sampled_agents)} agents")

        except Exception as e:
            logger.error(f"Error generating query for agents {[agent['agent_id'] for agent in sampled_agents]}: {e}")
        return tasks
    
    def generate_batch(self, n_tasks: int):
        self.counter = 0
        tasks = {}
        reference_agents = self.corpus.sample(n=n_tasks, replace=False).to_dict(orient="records")
        
        for agent in reference_agents:
            tasks.update(self.generate(agent))
        return tasks

    def generate_batch_from(self, ref_agents: list[dict]) -> dict[str, tuple[str, list[str]]]:
        self.counter = 0
        tasks = {}

        for agent in ref_agents:
            tasks.update(self.generate(agent))
        return tasks
