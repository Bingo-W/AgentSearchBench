<div align= "center">
    <h1> AgentSearchBench🤖</h1>
</div>

<div align="center">

  [![arxiv-link](https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red)]()
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

<div align="center">

![Dialogues](https://img.shields.io/badge/Tool\_Agents-9759-gree?style=flat-square)
![Dialogues](https://img.shields.io/badge/Total\_Tasks-3.2K-greene?style=flat-square)
![Dialogues](https://img.shields.io/badge/Total\_Agent\_Executions-66.7K-gree?style=flat-square)

</div>

**AgentSearchBench**: A Benchmark for AI Agent Search in the Wild

If you find this project useful, feel free to ⭐️ it and give it a [Citation](#Citation)!

## Overview

> **Abstract**
> The rapid growth of AI agent ecosystems is transforming how complex tasks are delegated and executed, creating a new challenge of identifying suitable agents for a given task. Unlike traditional tools, agent capabilities are often compositional and execution-dependent, making them difficult to assess from textual descriptions alone. However, existing research and benchmarks typically assume well-specified functionalities, controlled candidate pools, or only executable task queries, leaving realistic agent search scenarios insufficiently studied. We introduce AgentSearchBench, a large-scale benchmark for agent search in the wild, built from nearly 10,000 real-world agents across multiple providers. The benchmark formalizes agent search as retrieval and reranking problems under both executable task queries and high-level task descriptions, and evaluates relevance using execution-grounded performance signals. Experiments reveal a consistent gap between semantic similarity and actual agent performance, exposing the limitations of description-based retrieval and reranking methods. We further show that lightweight behavioral signals, including execution-aware probing, can substantially improve ranking quality, highlighting the importance of incorporating execution signals into agent discovery.
> 

## Data

Below is the statistics of the data:

| Total Agents | Total Tasks | Task Query | Task Description | Total Executions |
|-----------|----------|---------------|---------------|------------------|
| 9759 | 3211 | 2952  | 259  | 66740 |

We crawl ~10000 real-world AI Agents from [GPT Store](https://chatgpt.com/gpts), [Google Cloud Marketplace](https://cloud.google.com/marketplace), and [AgentAI Platform](https://agent.ai/).

### Data Release

Please download our dataset from [Google Drive]().

```
├── data/
│  ├── agentbase/
│  ├── tasks/
│  ├── labels/
│  ├── responses/
```

- `agentbase`: metadata and access links for scraped real-world agents.
- `tasks`: asbench tasks for task query and task description.
- `labels`: asbench task labels (relevant agents)
- `responses`: raw execution responses for each single-agent task query.


## Task Generation

Here is an overview of the benchmark construction:

<br>
<div align="center">
<img src="assets/pipeline.png" width="800px">
</div>
<br>

To start with, follow the [installation steps](#requirements). See [generation](asbench/generation/README.md) for step-by-step details on how to generate each task type.

### Quick Start

To generate the tasks only:

```bash
python -m scripts.generate \
    --type single \
    --agentbase asbench/data/agentbase-v1.1.csv
```

To generate the tasks with labels:

```bash
python -m scripts.generate \
    --type single \
    --agentbase asbench/data/agentbase-v1.1.csv \
    --generate-labels True \
    --debug True
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--type` | Task Type (single, multi, real, description) | Required |
| `--agentbase` | Path to the dataset | Required |
| `--source-tasks` | Path to the source tasks (required by types `real` and `multi`). | `None` |
| `--generate-labels` | Whether to also generate labels for the task | `False` |
| `--debug` | Whether to save debug information (candidates, judged labels, responses). | `False` |
| `--experiment-name` | Experiment name for output | `experiment` |
| `--config-path` | Path to the configuration directory | `asbench/configs/` |

Modify the configurations in [generation.yaml](asbench/configs/generation.yaml) and [models.yaml](asbench/configs/models.yaml) for finer-grained control.

## Probing

[Instructions to run probing queries and get agent response]


## Baseline Evaluations


## Requirements

Install dependencies:

```bash
git clone https://github.com/Bingo-W/AgentSearchBench.git
cd AgentSearchBench
uv sync
```

Set-up the configurations:

```bash
├── asbench/configs/
│  ├── config.py
│  ├── generation.yaml
│  ├── models.yaml
│  ├── .env
```

Check to ensure you have above configuration folder structure. 
Update the `.env` with api keys required by your `models.yaml` (depending on your set-up).

```bash
HF_TOKEN=
HF_API_KEY=
OPENAI_API_KEY=
AGENT_AI_NETWORK_API_KEY=
...
```


## Citation