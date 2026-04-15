# AgentSearchBench Task Generation

- [`generate.py`](../../scripts/generate.py) is used to launch generation tasks.
- [`generation.yaml`](#generation-configs) contains key pipeline configurations.
- [`models.yaml`](#models-configuration) contains details about used model configurations.

## Examples

We provide examples on how to run each task type using the full pipeline.

### Single-Agent Task Query

```bash
uv run python -m scripts.generate \
    --type single \
    --agentbase asbench/data/agentbase.csv \
    --generate_labels True \
    --debug True \
    --experiment-name tq-single-agent
```

### Real Task Query

First prepare a task file from an existing benchmark you want to use as part of AgentSearchBench. You can find an example [here](../data/examples/gaia.jsonl). You should follow the following format:
```json
{q:0: ["<insert_task_text>", "<insert_golden_label_if_any>"]}
...
```

Reference this filepath when running the pipeline:

```bash
uv run python -m scripts.generate \
    --type real \
    --agentbase asbench/data/agentbase.csv \
    --source-tasks asbench/data/examples/gaia.jsonl \
    --generate_labels True \
    --debug True \
    --experiment-name tq-real
```

### Multi-Agent Task Query

Multi-agent task query re-use existing single-agent task query. You should reference filepath to single-agent task query files. We recommend running single-agent pipeline first with `generate_labels` and `debug` set to `True`. An example is provided at [here](../data/examples).

```bash
uv run python -m scripts.generate \
    --type multi \
    --agentbase asbench/data/agentbase.csv \
    --source-tasks asbench/data/examples/task_query_single_tasks.jsonl \
    --generate_labels True \
    --debug True \
    --experiment-name tq-multi-agent
```

### Task Description

```bash
uv run python -m scripts.generate \
    --type description \
    --agentbase asbench/data/agentbase.csv \
    --generate_labels True \
    --debug True \
    --experiment-name td
```

Note that for our default runs we set `n_subtasks=10` and `n_candidates=20`, totalling 200 agents executions required per generated task description. This makes task description generation at-scale time-consuming and costly.

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--type` | Task Type (single, multi, real, description) | Required |
| `--agentbase` | Path to the dataset | Required |
| `--source-tasks` | Path to the source tasks (required by types `real` and `multi`). | `None` |
| `--generate-labels` | Whether to also generate labels for the task | `False` |
| `--debug` | Whether to save debug information (candidates, judged labels, responses). | `False` |
| `--experiment-name` | Experiment name for output | `experiment` |
| `--config-path` | Path to the configuration directory | `asbench/configs/` |


## Generation Configs

The generation pipeline is configured via [`generation.yaml`](../configs/generation.yaml). We provide default settings used during AgentSearchBench construction.

### Task Generation

| Parameter | Default | Description |
|---|---|---|
| `gen_model` | `gpt-5.2` | Model used to generate tasks (defined in `models.yaml`) |
| `n_tasks` | `100` | Number of tasks to generate |
| `n_agents` | `[2, 3, 4]` | Agent counts to use per multi-agent task query |
| `n_subtasks` | `10` | Number of subtasks per task description |
| `sample_pool_size` | `100` | Candidate pool size for subtask sampling |
| `coherence_threshold` | `0.8` | Minimum coherence score to accept a generated task |
| `n_criteria` | `5` | Evaluation criteria per task (recommended: `n_subtasks / 2`) |

### Candidate Selection

| Parameter | Default | Description |
|---|---|---|
| `n_candidates` | `5` | Number of candidate documents to retrieve per query |
| `selector_models` | — | Retrieval models and their weights (defined in `models.yaml`) |

Default selector weights:

```yaml
selector_models:
  bm25: 1.0
  bge: 1.0
  toolret: 1.0
```

### Response Generation

| Parameter | Default | Description |
|---|---|---|
| `active_platforms` | `[agentainetwork, openaiagents]` | Platforms used to generate responses (comment/uncomment to toggle) |

### Judging

| Parameter | Default | Description |
|---|---|---|
| `judge_models` | `[gpt-5.2, gemini-2.5-pro, qwen-3.5-397B_openrouter]` | Models used for relevance judging (defined in `models.yaml`) |
| `use_golden_labels` | `false` | Use ground-truth labels instead of model judgements |
| `judge_token_limit` | `4000` | Max tokens per judging request |

### Labels

| Parameter | Default | Description |
|---|---|---|
| `qrel_type` | `binary` | Relevance label format: `binary` or `graded` |
| `rank_label_type` | `golden` | Ranking label granularity |
| `relevance_threshold` | `4` | Minimum score to consider a document relevant |


## Models Configuration

Models are configured via [`models.yaml`](../configs/models.yaml). It defines NLI, retrieval, LLM, and executor platform settings.

### NLI Model

| Parameter | Description |
|---|---|
| `nli_model` | HuggingFace model used for natural language inference |
| `hf_token_env` | Env variable name for the HuggingFace token |
| `hf_api_env` | Env variable name for the HuggingFace API key |

### Retrieval Models

Sentence Transformer models used for candidate selection. Each model references a key in `generation.yaml` under `selector_models`.

| Key | Model | Index |
|---|---|---|
| `bm25` | *(none — keyword-based)* | `naive` |
| `bge` | `BAAI/bge-large-en-v1.5` | `naive` |
| `toolret` | `mangopy/ToolRet-trained-bge-large-en-v1.5` | `naive` |

The `index` field controls the retrieval index type. Options: `description`, `agentbase`, `naive`.

### LLM Models

Models are routed via [LiteLLM](https://github.com/BerriAI/litellm). Each key (e.g. `gpt-5.2`) is referenced by `gen_model` and `judge_models` in `generation.yaml`.

| Key | Model path | API key env | Temperature |
|---|---|---|---|
| `gpt-5.2` | `openai/gpt-5.2` | `OPENAI_API_KEY` | `1.0` |
| `gemini-2.5-pro` | `gemini/gemini-2.5-pro` | `GEMINI_API_KEY` | `1.0` |
| `qwen-3.5-397B` | `openrouter/qwen/qwen3.5-397b-a17b` | `OPENROUTER_API_KEY` | `1.0` |

To add a model, append an entry following the same structure. Any LiteLLM-compatible model path is supported.

### Executor Platforms

Platforms used for agent response generation, referenced by `active_platforms` in `generation.yaml`.

| Key | Credential |
|---|---|
| `agentainetwork` | `AGENT_AI_NETWORK_KEY` (env variable) |
| `openaiagents` | `session.json` (session file path) |


## CodeBase Task Formatting

The CodeBase follows a JSONL format for task generation and probing:

```python
Tasks = dict[str, tuple]
# {task_id: (task_text, ...)}
Probes = dict[str, tuple[str, list[str]]]
# {task_id: (task_text, [agent_id1, agent_id2, ...])}
TaskLabels = dict[str, dict[str, int]]
# {task_id: {agent_id: relevance}}
TaskCandidates = dict[str, list[str]]
# {task_id: [agent_id1, agent_id2, ...]}
TaskResponses = dict[str, dict[str, tuple[str, float]]]
# {task_id: {agent_id: (response_text, execution_time)}}
TaskJudgedLabels = dict[str, dict[str, tuple[int, str]]]
# {task_id: {agent_id: (score, reasoning)}}
```

For further information [see utilities](utils.py).