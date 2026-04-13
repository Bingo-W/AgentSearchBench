import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional
import click
import pandas as pd

from asbench.configs.config import load_config
from asbench.generation.utils import (
    Tasks, TaskLabels,
    TaskCandidates, TaskJudgedLabels, TaskResponses,
    LLMParams, RetrieverParams, ExecutorParams,
    CustomFormatter
)

from asbench.generation.pipelines.base import BasePipeline
from asbench.generation.pipelines.task_query_single import TaskQuerySingle
from asbench.generation.pipelines.task_query_real import TaskQueryReal
from asbench.generation.pipelines.task_query_multi import TaskQueryMulti
from asbench.generation.pipelines.task_description import TaskDescription


@click.command()
@click.option(
    "--type",
    type=click.Choice(["single", "multi", "real", "description"]),
    required=True,
    help="The type of the task to generate.",
)
@click.option(
    "--agentbase",
    type=str,
    required=True,
    help="The path to the AgentBase dataset file.",
)
@click.option(
    "--source-tasks",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="The path to the source tasks file.",
)
@click.option(
    "--generate-labels",
    type=bool,
    required=False,
    default=False,
    help="Whether to generate labels for the tasks.",
)
@click.option(
    "--debug",
    type=bool,
    required=False,
    default=False,
    help="Whether to save debug information (candidates, judged labels, responses).",
)
@click.option(
    "--experiment-name",
    type=str,
    required=True,
    default="experiment",
    help="The name of the experiment.",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="The path to the configuration file.",
)
def main(
    type: str,
    agentbase: str,
    source_tasks: Optional[Path],
    generate_labels: bool,
    debug: bool,
    experiment_name: Optional[str],
    config_path: Optional[Path],
):
    """
    Generate tasks for AgentSearchBench.
    """
    # logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    # playwright (runs on first use)
    subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        check=True
    )

    # config
    config = load_config(config_path) if config_path else load_config()
    model_cfg = config["models"]
    gen_cfg = config["generation"]
    gen_cfg["generate_labels"] = generate_labels
    
    # AgentBase executable dataset
    active_platforms = gen_cfg["active_platforms"]
    agentbase_df = pd.read_csv(agentbase)
    agentbase_df = agentbase_df[agentbase_df["platform_name"].isin(active_platforms)]
    agentbase_executable = agentbase[:-4] + "_executable.csv"
    agentbase_df.to_csv(agentbase_executable, index=False)
    gen_cfg["agentbase"] = agentbase_executable

    # LLM params
    llm_params: LLMParams = {
        name: (
            details["model"],
            details["api_key"],
            details["temperature"]
        )
        for name, details in model_cfg["llm_models"].items()
    }
    gen_cfg["llm_params"] = llm_params
    assert all(model in gen_cfg["llm_params"] for model in gen_cfg["judge_models"] + [gen_cfg["gen_model"]]), "Missing LLM model configurations."

    # Retriever params
    retriever_params: RetrieverParams = {
        name: (
            details["model"],
            details["index"]
        )        for name, details in model_cfg["retrieval_models"].items()
    }
    gen_cfg["retriever_params"] = retriever_params
    assert all(model in gen_cfg["retriever_params"] for model in gen_cfg["selector_models"].keys()), "Missing retriever model configurations."

    # Executor params
    executor_params: ExecutorParams = {
        name: details["credential"]
        for name, details in model_cfg["executor_platforms"].items()
    }
    gen_cfg["executor_params"] = executor_params
    assert all(executor in gen_cfg["executor_params"] for executor in gen_cfg["active_platforms"]), "Missing executor model configurations."

    # Executor params
    gen_cfg["executor_params"] = {
        name: details["credential"]
        for name, details in model_cfg["executor_platforms"].items()
    }
    assert all(executor in gen_cfg["executor_params"] for executor in gen_cfg["active_platforms"]), "Missing executor model configurations."

    # Pipeline selection
    tasks: Tasks = {}
    task_labels: TaskLabels = {}
    task_rank_labels: TaskLabels = {}
    task_labels_multi: dict[str, TaskLabels] = {}
    candidates: TaskCandidates = {}
    judged_labels: TaskJudgedLabels = {}
    responses: TaskResponses = {}
    
    if type == "single":
        pipeline = TaskQuerySingle(
            **gen_cfg
        )
        tasks, task_labels, candidates, judged_labels, responses = pipeline.run()
    elif type == "real":
        if source_tasks is None:
            raise ValueError("Source tasks file must be provided for real task generation.")
        
        gen_cfg["real_tasks"] = source_tasks
        pipeline = TaskQueryReal(
            **gen_cfg,
        )
        tasks, task_labels, candidates, judged_labels, responses = pipeline.run()
    elif type == "multi":
        if source_tasks is None:
            raise ValueError("Source tasks file must be provided for multi-agent task generation.")

        gen_cfg["single_agent_paths"] = {
            "tasks": source_tasks,
            "labels": source_tasks.parent / (source_tasks.stem.replace("tasks", "labels") + source_tasks.suffix),
            "responses": source_tasks.parent / (source_tasks.stem.replace("tasks", "responses") + source_tasks.suffix),
            "judged_labels": source_tasks.parent / (source_tasks.stem.replace("tasks", "judged_labels") + source_tasks.suffix),
        }
        gen_cfg["nli_model"] = model_cfg["nli_model"]
        pipeline = TaskQueryMulti(
            **gen_cfg,
        )
        tasks, task_labels, task_rank_labels, task_labels_multi = pipeline.run()
    elif type == "description":
        pipeline = TaskDescription(
            **gen_cfg,
        )
        tasks, task_labels, task_rank_labels, candidates, judged_labels, responses = pipeline.run()
    else:
        raise NotImplementedError(f"Pipeline type {type} is not implemented.")
    
    # Save
    BasePipeline.save(
        tasks=tasks,
        task_labels=task_labels,
        experiment_name=experiment_name,
    )
    if debug:
        BasePipeline.save_debug(
            task_rank_labels=task_rank_labels,
            task_labels_multi=task_labels_multi,
            candidates=candidates,
            judged_labels=judged_labels,
            responses=responses,
            experiment_name=experiment_name,
        )

if __name__ == "__main__":
    main()