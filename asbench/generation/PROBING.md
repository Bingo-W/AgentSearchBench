# AgentSearchBench Probing

The [AgentBase](https://huggingface.co/datasets/AgentSearch/AgentSearchBench-Agents) dataset contains 9,760 agents collected from multiple platforms, among which 7,867 provide executable interfaces. We outline details on how to run the probing experiments below.

- [`probe.py`](../../scripts/probe.py) is used to launch probing.
- [`generation.yaml`](../configs/generation.yaml) is used to select active executor platforms.
- [`models.yaml`](../configs/models.yaml) contains executor platform configurations.

## AgentBase Dataset

You can download the AgentBase dataset from HuggingFace by running the [`load_dataset.py`](../../scripts/load_dataset.py) script. Alternatively:

```python
dataset = load_dataset("AgentSearch/AgentSearchBench-Agents")
df = pd.DataFrame(dataset["agents"])
df.to_csv("agentbase.csv", index=False)
```

## Executor Platforms

Executor platforms are public platforms to be used for the probing experiments. You can modify the availability by changing `active_platforms` in `generation.yaml`. The platform configuration can be set using `executor_platforms` field in `models.yaml`.

| Key | Credential |
|---|---|
| `agentainetwork` | `AGENT_AI_NETWORK_KEY` (env variable) |
| `openaiagents` | `session.json` (session file path) |


We outline details below on how to setup each platform.

### AgentAI Network

References a list of agents scraped from the [AgentAI platform](https://agent.ai/). To include it in your experimentation, you should create a free account and include your api key in `.env` file. Note that your environment variable name should match the setup in `executor_platforms` defined in `models.yaml`.

### OpenAI Agents

References a list of agents scraped from the [GPT store](https://chatgpt.com/gpts). This platform can be probed via Playwright's Chromium browser. You will be requested to login to your OpenAI account to proceed with probing. You can define a `session.json`file to save your credentials and avoid relogging (see `executor_platforms` in `models.yaml`).

> ⚠️ We recommend upgrading your [OpenAI subscription](https://chatgpt.com/pricing/) to `Go` or `Plus` to avoid being rate limited.

### Adding New Platforms

You are welcome to extend [AgentBase](https://huggingface.co/datasets/AgentSearch/AgentSearchBench-Agents) and include your own platforms. To implement the executor you can create a new executor class at [generation/responses/executors](responses/executors/) folder. The new class should inherit the abstract [`base.py`](responses/base.py) and be reflected in [`executor_factor.py`](responses/executor_factory.py).

These changes should be reflected in `active_platforms` in `generation.yaml` and configured via `executor_platforms` in `models.yaml`. Finally, make sure to update your `agentbase.csv` with the newly added agents.


## Generating Probes

Probes should follow a predefined JSONL format:

```python
Probes = dict[str, tuple[str, list[str]]]
# {task_id: (task_text, [agent_id1, agent_id2, ...])}
```

Here is an example:

```bash
{"q:0": ["Draft a legally compliant performance improvement plan ...", ["agt:openaiagents:402f63@v1.1", "agt:agentainetwork:f39967@v1.1"]]}
```

The users are recommended to implement their own probe generation strategies and test their performance on [AgentBase](https://huggingface.co/datasets/AgentSearch/AgentSearchBench-Agents). A set of preliminary examples can be found in [probes.jsonl](../data/examples/probes.jsonl).


## Running Probes

> ⚠️ Note that the specified candidate agents in `probes.jsonl` **MUST** be executable (i.e. belong to an available platform). Make sure you have setup the [executor platforms](#executor-platforms) before proceeding with probing.

To probe the AgentBase:

```bash
uv run python -m scripts.probe \
    --agentbase asbench/data/agentbase.csv \
    --probes asbench/data/examples/probes.jsonl
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--agentbase` | Path to the dataset. | Required |
| `--probes` | Path to the probing tasks. | Required |
| `--name` | Probing run name for output | `probe` |
| `--config-path` | Path to a custom configuration directory | `asbench/configs/` |
