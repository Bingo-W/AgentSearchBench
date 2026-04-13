# Task Description Rubric Generation, Assignment, and Scoring

import logging
from asbench.generation.prompts import RUBRIC_GEN_PROMPT, RUBRIC_EVAL_PROMPT
from asbench.generation.utils import extract_json
from asbench.generation.inference import LLMInference


logger = logging.getLogger(__name__)


class DescriptionRubric:
    def __init__(
            self,
            llm: LLMInference,
    ):
        self.llm = llm
        self.gen_batch_size = 10  # number of specific queries passed in per batch for nugget generation

    def generate_rubric(self, user_intent: str, n_criteria: int) -> list[str]:
        """
        Iteratively generate a rubric based on the user intent.
        - user_intent: user intent that the rubric should address
        - return: list of rubric criteria
        """
        criteria = []

        messages = RUBRIC_GEN_PROMPT.get_messages(
            user_intent=user_intent,
            n_criteria=n_criteria,
        )
        response = self.llm.invoke(messages).content
        extracted = extract_json(response)

        criteria = extracted.get("rubric", [])
        if not isinstance(criteria, list):
            criteria = []
        if criteria == []:
            logger.warning(f"No rubric criteria extracted from response: {response}")
    
        return criteria
    
    def evaluate_tasks(self, rubric: list[str], specific_tasks: dict[str, tuple]) -> dict[str, dict[str, int]]:
        """
        Score each specific task with respect to each criterion in the rubric.
        - user_intent: user intent that the nuggets address
        - rubric: list of rubric criteria
        - specific_tasks: dict of specific tasks, {"query_id": ("query_text", [sample(s)])}

        - return: dict of task labels, {"query_id": [criterion1_label, criterion2_label, ...]}
        """
        task_labels = {}

        for criterion in rubric:
            task_labels[criterion] = self.evaluate_criterion(criterion, specific_tasks)
        return task_labels

    def evaluate_criterion(self, criterion: str, specific_tasks: dict[str, tuple]) -> dict[str, int]:
        """
        Score each specific task with respect to how well it meets the criterion.
        - user_intent: user intent that the nuggets address
        - criterion: rubric criterion to evaluate against
        - specific_tasks: dict of specific tasks, {"query_id": ("query_text", [sample(s)])}

        - return: dict of task scores, {"query_id": score}
        """

        # score each task individually
        labels = {}
        for task_id, (task_text, _) in specific_tasks.items():
            messages = RUBRIC_EVAL_PROMPT.get_messages(
                criterion=criterion,
                task=task_text,
            )
            response = self.llm.invoke(messages).content
            extracted = extract_json(response)

            label = extracted.get("label", None)
            if label is None:
                logger.warning(f"No label extracted for task {task_id} and criterion '{criterion}' from response: {response}")
                label = 0  # default to not satisfied
            elif isinstance(label, str):
                try:
                    label = int(label)
                except ValueError:
                    logger.warning(f"Label for task {task_id} and criterion '{criterion}' is not an integer in response: {response}")
                    label = 0  # default to not satisfied

            labels[task_id] = label
        return labels
    
    def _score_tasks(self, task_labels: dict[str, dict[str, int]], specific_tasks: dict[str, tuple]) -> dict[str, float]:
        """
        Score each task based on the number of criteria it satisfies.
        - task_scores: dict of task labels, {"query_id": [criterion1_label, criterion2_label, ...]}
        - specific_tasks: dict of specific tasks, {"query_id": ("query_text", [sample(s)])}

        - return: dict of task scores, {"query_id": score}
        """
        scored_tasks = {}

        for task_id in specific_tasks.keys():
            labels = [labels[task_id] for criterion, labels in task_labels.items() if task_id in labels]
            scored_tasks[task_id] = sum(labels) / len(labels) if labels else 0.0
        return scored_tasks
    
    def select_tasks(
            self,
            m: int,
            task_labels: dict[str, dict[str, int]],
            task_scores: dict[str, float],
            specific_tasks: dict[str, tuple]
    ) -> dict[str, tuple]:
        """
        :returns dict of selected tasks, {task_id: ("task_text", [sample(s)])}
        """

        selected_tasks = {}
        _lambda = 1.0
        current_budget = {criterion: 0 for criterion in task_labels.keys()} # b_j
        target_budget = m / len(list(task_labels.keys()))  # ideal even distribution
        sorted_scores = sorted(task_scores.items(), key=lambda x: x[1], reverse=True)

        for i in range(m): # select m tasks
            selection_scores = {}
            for task_id, score in sorted_scores:
                penalty = sum([max(0, current_budget[criterion] - target_budget) for criterion, labels in task_labels.items() if task_id in labels and labels[task_id] == 1])
                selection_scores[task_id] = score - _lambda * penalty

            best_task_id = max(selection_scores, key=selection_scores.get)
            selected_tasks[best_task_id] = specific_tasks[best_task_id]

            # update
            for criterion, labels in task_labels.items():
                if best_task_id in labels and labels[best_task_id] == 1:
                    current_budget[criterion] += 1
            sorted_scores = [(task_id, score) for task_id, score in sorted_scores if task_id != best_task_id]

        return selected_tasks
