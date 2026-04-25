# Multi-LLM-as-a-Judge for response quality evaluation.

import logging
from asbench.generation.judges.base import BaseLLMJudge


logger = logging.getLogger(__name__)

class MultiJudge:
    def __init__(
        self,
        judges: list[BaseLLMJudge],
        weights: list[int] | None=None,
    ):
        weights = weights or [1] * len(judges)
        assert len(judges) == len(weights), "Number of judges must match number of weights"

        self.judges = judges
        total = sum(weights)
        self.norm_weights = [w / total for w in weights]

    def judge(
            self,
            tasks: dict[str, tuple],
            responses: dict[str, dict[str, tuple]]
    ) -> dict[str, dict[str, tuple[int, str]]]:
        # get scores from each judge
        all_judged_labels = []
        for judge in self.judges:
            judged_labels = judge.judge(tasks, responses)
            all_judged_labels.append(judged_labels)

        # aggregate scores
        final_judged_labels = {}
        for task_id in tasks.keys():
            final_judged_labels[task_id] = {}

            for agent_id in responses.get(task_id, {}).keys():
                # weighted and rounded average of scores
                weighted_score_sum = 0
                reasoning_list = []
                for i, judged_labels in enumerate(all_judged_labels):
                    score, reasoning = judged_labels.get(task_id, {}).get(agent_id, (0, ""))
                    weighted_score_sum += self.norm_weights[i] * score
                    reasoning_list.append(reasoning)
                
                final_judged_labels[task_id][agent_id] = (round(weighted_score_sum), "\n".join(reasoning_list))
                logger.info(f"Aggregated judged score for task {task_id}, agent {agent_id}: score={weighted_score_sum}")

        return final_judged_labels

    def judge_debug(
            self,
            tasks: dict[str, tuple],
            responses: dict[str, dict[str, tuple]],
    ) -> tuple[dict, dict]:
        # also return individual judge scores for debugging
        individual_judged_labels = {}
        for judge in self.judges:
            judged_labels = judge.judge(tasks, responses)
            individual_judged_labels[judge.llm_model] = judged_labels
        hybrid_judged_labels = {}

        for task_id in tasks.keys():
            hybrid_judged_labels[task_id] = {}
            for agent_id in responses.get(task_id, {}).keys():
                # weighted and rounded average of scores
                weighted_score_sum = 0
                reasoning_list = []
                for i, judged_labels in enumerate(list(individual_judged_labels.values())):
                    score, reasoning = judged_labels.get(task_id, {}).get(agent_id, (0, ""))
                    weighted_score_sum += self.norm_weights[i] * score
                    reasoning_list.append(reasoning)
                
                hybrid_judged_labels[task_id][agent_id] = (round(weighted_score_sum), "\n".join(reasoning_list))
                logger.info(f"Aggregated judged score for task {task_id}, agent {agent_id}: score={weighted_score_sum}")

        return hybrid_judged_labels, individual_judged_labels
