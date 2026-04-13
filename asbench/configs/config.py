# configs/config.py
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import pandas as pd

from asbench.generation.utils import get_exp_dir


CONFIG_DIR = Path(__file__).parent

def load_config(config_path: Optional[Path] = None):
    config_path = config_path if config_path else CONFIG_DIR
    load_dotenv(config_path / ".env")

    def _load(name):
        with open(config_path / name) as f:
            return yaml.safe_load(f)

    config = {
        "models": _load("models.yaml"),
        "generation": _load("generation.yaml")
    }
    
    # resolve API keys
    for model_name, model_info in config["models"]["llm_models"].items():
        config["models"]["llm_models"][model_name]["api_key"] = os.getenv(model_info["api_key_env"])

    config["models"]["executor_platforms"]["agentainetwork"]["credential"] = os.getenv(config["models"]["executor_platforms"]["agentainetwork"]["credential"])
    config["models"]["hf_token"] = os.getenv(config["models"]["hf_token_env"])
    config["models"]["hf_api_key"] = os.getenv(config["models"]["hf_api_env"])

    # resolve paths
    config["models"]["executor_platforms"]["openaiagents"]["credential"] = config_path / config["models"]["executor_platforms"]["openaiagents"]["credential"]

    return config
