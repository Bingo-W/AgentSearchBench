# loading prompt templates
from pathlib import Path
from asbench.generation.prompts.loader import PromptLoader

TEMPLATE_DIR = Path(__file__).parent / "templates"

# task generation
SINGLE_DOC_PROMPT = PromptLoader.load(TEMPLATE_DIR / "task_single_doc.txt")
MULTI_DOC_PROMPT = PromptLoader.load(TEMPLATE_DIR / "task_multi_doc.txt")
CONTEXT_PROMPT = PromptLoader.load(TEMPLATE_DIR / "task_context.txt")
RUBRIC_GEN_PROMPT = PromptLoader.load(TEMPLATE_DIR / "rubric_generate.txt")
RUBRIC_EVAL_PROMPT = PromptLoader.load(TEMPLATE_DIR / "rubric_evaluate.txt")
MULTI_AGENT_PROMPT = PromptLoader.load(TEMPLATE_DIR / "task_multi_agent.txt")
TASK_DESCRIPTION_PROMPT = PromptLoader.load(TEMPLATE_DIR / "task_description.txt")
TASK_PROBING_PROMPT = PromptLoader.load(TEMPLATE_DIR / "task_probing.txt")

# judges
JUDGE_BASE_PROMPT = PromptLoader.load(TEMPLATE_DIR / "judge_base.txt")
JUDGE_MULTI_PROMPT = PromptLoader.load(TEMPLATE_DIR / "judge_multi.txt")
JUDGE_TASK_PROMPT = PromptLoader.load(TEMPLATE_DIR / "judge_task.txt")
JUDGE_DOC_CONSISTENCY_PROMPT = PromptLoader.load(TEMPLATE_DIR / "judge_doc_consistency.txt")

__all__ = [
    "SINGLE_DOC_PROMPT", "MULTI_DOC_PROMPT", "CONTEXT_PROMPT",
    "RUBRIC_GEN_PROMPT", "RUBRIC_EVAL_PROMPT",
    "MULTI_AGENT_PROMPT", "TASK_DESCRIPTION_PROMPT", "TASK_PROBING_PROMPT",
    "JUDGE_BASE_PROMPT", "JUDGE_MULTI_PROMPT", "JUDGE_TASK_PROMPT", "JUDGE_DOC_CONSISTENCY_PROMPT",
]
