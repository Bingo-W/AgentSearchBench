import os
import json
import re
import logging
from pathlib import Path
import pandas as pd
import fnmatch

logger = logging.getLogger(__name__)


Tasks               = dict[str, tuple]                          # {task_id: (task_text, ...)}
Probes              = dict[str, tuple[str, list[str]]]          # {task_id: (task_text, [candidate1, candidate2, ...])}
TaskLabels          = dict[str, dict[str, int]]                 # {task_id: {agent_id: relevance}}

TaskCandidates      = dict[str, list[str]]                      # {task_id: [candidate1, candidate2, ...]} 
TaskResponses       = dict[str, dict[str, tuple[str, float]]]   # {task_id: {agent_id: (response_text, execution_time)}}
TaskJudgedLabels    = dict[str, dict[str, tuple[int, str]]]     # {task_id: {agent_id: (score, reasoning)}}

LLMParams           = dict[str, tuple[str, str, int]]           # {name: (model_name, model_api_key, temperature)}
RetrieverParams     = dict[str, tuple[str, str]]                # {name: (model_name, indexing_type)}
ExecutorParams      = dict[str, str]                            # {executor_name: credential_info}


def generic_json_save(data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    return

def generic_json_load(input_path: str) -> dict:
    with open(input_path, "r") as f:
        data = json.load(f)
    return data

def generic_jsonl_save(data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in data.items():
            json_line = json.dumps({entry[0]: entry[1]})
            f.write(json_line + "\n")
    return

def generic_jsonl_append(data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a") as f:
        for entry in data.items():
            json_line = json.dumps({entry[0]: entry[1]})
            f.write(json_line + "\n")
    return

def generic_jsonl_load(input_path: str) -> dict:
    data = {}
    with open(input_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                data.update(entry)
            except json.JSONDecodeError as e:
                print(f"Failed to parse line: {line[:10]}")
                raise e
    return data

def extract_json(llm_text: str) -> dict:
    # strip <think>...</think> blocks from reasoning models
    llm_text = re.sub(r"<think>.*?</think>", "", llm_text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", llm_text, re.DOTALL)
    if not match:
        logger.error(f"Failed to extract JSON from LLM response: {llm_text}")
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {e}\nExtracted: {match.group()}\nFull response: {llm_text}")
        return {}

def random_sample(agents: pd.DataFrame, category: str, num: int) -> list[dict]:
    filtered_agents = agents[agents["agent_category"] == category]
    return filtered_agents.sample(n=num).to_dict(orient="records")

def get_exp_dir(output_dir: Path, exp_target: str, exp_iteration: str) -> Path:
    exp_dir = output_dir / exp_target / exp_iteration
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def get_task_files(query_dir: str, prefix: str) -> dict:
    # loads queries with the given prefix (e.g., "single_doc" or "multi_doc") from the specified directory
    query_files = []
    for query_file in os.listdir(query_dir):
        if query_file == f"{prefix}_tasks.jsonl":
            query_files.append(query_file)
    return query_files

def get_qrel_files(qrel_dir: str, prefix: str, type: str) -> dict:
    # loads qrels with the given prefix (e.g., "single_doc" or "multi_doc") from the specified directory
    qrel_files = []
    for qrel_file in os.listdir(qrel_dir):
        if qrel_file == f"{prefix}_{type}_qrels.jsonl":
            qrel_files.append(qrel_file)
    return qrel_files

def get_qrel_ranking_files(qrel_dir: str, prefix: str) -> list[str]:
    qrel_files = []
    for qrel_file in os.listdir(qrel_dir):
        if fnmatch.fnmatch(qrel_file, f"{prefix}*ranking_labels.jsonl"):
            qrel_files.append(qrel_file)
    return qrel_files


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    dim = "\x1b[2m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    base_format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    def format(self, record):
        if record.levelno == logging.INFO:
            importance = getattr(record, "importance", "low")

            if importance == "high":
                log_fmt = self.grey + self.base_format + self.reset
            else:
                log_fmt = self.dim + self.grey + self.base_format + self.reset

        elif record.levelno == logging.WARNING:
            log_fmt = self.yellow + self.base_format + self.reset
        elif record.levelno == logging.ERROR:
            log_fmt = self.red + self.base_format + self.reset
        elif record.levelno == logging.CRITICAL:
            log_fmt = self.bold_red + self.base_format + self.reset
        else:
            log_fmt = self.grey + self.base_format + self.reset

        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
