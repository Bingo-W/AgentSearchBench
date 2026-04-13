# Abstract base class for platform executors

from abc import ABC, abstractmethod


class PlatformExecutor(ABC):
    def __init__(self, credential: str, debug: bool = True):
        self.credential_info = credential
        self.debug = debug

    @abstractmethod
    def execute(self, agent_metadata: dict, query: str) -> tuple[str, float, int]:
        """
        :param agent_metadata: agent information
        :param query: the task to execute
        :return: a tuple of (response, execution_time, status_code)
        """
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def teardown(self):
        pass