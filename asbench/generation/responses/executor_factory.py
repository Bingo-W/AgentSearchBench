from asbench.generation.responses.base import PlatformExecutor
from asbench.generation.responses.executors.agentai import AgentAIExecutor
from asbench.generation.responses.executors.openai import OpenAIExecutor


class ExecutorFactory:
    _executors = {
        "agentainetwork": AgentAIExecutor,
        "openaiagents": OpenAIExecutor,
    }

    @classmethod
    def register_executor(cls, platform_name: str, executor_class: type):
        cls._executors[platform_name] = executor_class

    @classmethod
    def create_executor(
        cls, platform_name: str, **kwargs
    ) -> PlatformExecutor:
        executor_class = cls._executors.get(platform_name.lower())
        if not executor_class:
            raise ValueError(f"Unknown platform: {platform_name}")
        return executor_class(**kwargs)
