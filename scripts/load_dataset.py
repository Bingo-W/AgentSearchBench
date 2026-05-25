import pandas as pd
from datasets import load_dataset
from pathlib import Path


def load_agentbase():
    BASE_DIR = Path(__file__).parent.parent
    dataset = load_dataset("AgentSearch/AgentSearchBench-Agents")
    df = pd.DataFrame(dataset["agents"])
    df.to_csv(BASE_DIR / "asbench/data/agentbase.csv", index=False)
    return True

if __name__ == "__main__":
    load_agentbase()
