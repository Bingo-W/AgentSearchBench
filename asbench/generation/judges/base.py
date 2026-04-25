# LLM-as-a-Judge for response quality evaluation

import logging
from abc import ABC, abstractmethod

from asbench.generation.inference import LLMInference
from asbench.generation.prompts.loader import Prompt
from asbench.generation.prompts import JUDGE_BASE_PROMPT
from asbench.configs.config import load_config
import tiktoken

logger = logging.getLogger(__name__)


class BaseLLMJudge(ABC):
    def __init__(
        self,
        llm: LLMInference,
        token_limit: int,
        use_golden_labels: bool = False,
        prompt: Prompt = JUDGE_BASE_PROMPT,
    ):
        self.llm_model = llm.model_name
        self.llm = llm
        self.prompt = prompt
        self.config = load_config() 
        self.enc = tiktoken.encoding_for_model("gpt-4o")  # cap response tokens (outliers)
        self.token_limit = token_limit
        self.use_golden_labels = use_golden_labels

    @abstractmethod
    def judge(self, queries: dict[str, tuple], responses: dict) -> dict:
        pass

    @abstractmethod
    def judge_batch(self, queries: dict[str, tuple], responses: dict) -> dict:
        pass
