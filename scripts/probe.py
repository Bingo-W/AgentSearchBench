import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional
import click
import pandas as pd

from asbench.generation.responses.response_generator import ResponseGenerator
from asbench.configs.config import load_config
from asbench.generation.utils import (
    Probes, TaskResponses, ExecutorParams,
    CustomFormatter,
    generic_jsonl_load, generic_jsonl_save,
)


@click.command()
@click.option(
    "--agentbase",
    type=str,
    required=True,
    help="The path to the AgentBase dataset file.",
)
@click.option(
    "--probes",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The path to the target probe's file.",
)
@click.option(
    "--name",
    type=str,
    required=False,
    default="probe",
    help="The name of the probing run.",
)
@click.option(
    "--config-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="The path to the configuration file.",
)
def main(
    agentbase: str,
    probes: str,
    name: str,
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
    output_dir = Path(__file__).parent.parent / f"outputs/{name}"

    # Executor params
    probing_queries: Probes = generic_jsonl_load(probes)
    active_platforms = {
        agent.split(":")[1]
        for data in probing_queries.values()
        for agent in data[1]
    }

    executor_params: ExecutorParams = {
        name: details["credential"]
        for name, details in model_cfg["executor_platforms"].items()
    }
    assert all(executor in executor_params for executor in active_platforms), "Missing executor model configurations."

    # Response generator
    response_generator = ResponseGenerator(
        corpus=pd.read_csv(agentbase),
        executor_params=executor_params,
        response_path=output_dir,
    )

    # Continue probe generation
    candidates = {qid: data[1] for qid, data in probing_queries.items()}
    responses: TaskResponses = response_generator.gen_responses(probing_queries, candidates)
    generic_jsonl_save(responses, output_dir / "responses.jsonl")
    return True

if __name__ == "__main__":
    main()