# LLM-as-a-Judge for task quality evaluation

import logging

from asbench.generation.inference import LLMInference
from asbench.generation.prompts import JUDGE_TASK_PROMPT
from asbench.generation.utils import extract_json


logger = logging.getLogger(__name__)

class TaskJudge:
    def __init__(
        self,
        llm: LLMInference,
    ):
        self.llm = llm
        self.prompt = JUDGE_TASK_PROMPT

    def judge(self, tasks: dict[str, tuple]) -> dict[str, dict]:
        """
        tasks: dict[qid, (task_text, metadata)]
        returns: dict[qid, (score, reasoning)]
        """
        results = {}
        for qid, (task_text, _) in tasks.items():
            prompt = self.prompt.get_messages(query=task_text)
            llm_response = self.llm.invoke(prompt).content
            evaluation = extract_json(llm_response)
            results[qid] = self._normalise(evaluation)
            logger.info(f"Judged task {qid}")
        return results

    def _normalise(self, evaluation: dict) -> dict:
        score = int(evaluation.get("score", 0))
        reasoning = evaluation.get("reasoning", "")
        return {"score": score, "reasoning": reasoning}
