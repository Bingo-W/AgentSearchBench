# Abstract base class for task generators

from abc import ABC, abstractmethod
import pandas as pd
from asbench.generation.inference import LLMInference


class BaseGenerator(ABC):
    def __init__(
        self,
        corpus: pd.DataFrame,
        llm: LLMInference,
        tasks_per_call=1,
    ):
        """
        Abstract base class for task generators.

        corpus: reference document corpus
        llm_model: generation model
        tasks_per_call: Number of tasks to generate per LLM call.
            **Experimental** — default 1, but can increase for efficiency.
        """
        self.corpus = corpus
        self.llm = llm
        self.llm_model = self.llm.model_name
        self.tasks_per_call = tasks_per_call
        self.counter = 0

    @abstractmethod
    def generate(self) -> dict:
        """
        Generate task(s) from a single reference.
        """
        pass

    @abstractmethod
    def generate_batch(self) -> dict:
        """
        Generates tasks from a batch of references.
        """
        pass
