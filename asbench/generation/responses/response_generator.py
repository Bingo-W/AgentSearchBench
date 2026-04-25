# AgentBase multi-platform response generator

import time
import logging
import pandas as pd
import socket
from asbench.generation.responses.base import PlatformExecutor
from asbench.generation.responses.executor_factory import ExecutorFactory
from asbench.generation.utils import generic_jsonl_append, generic_jsonl_save


logger = logging.getLogger(__name__)

class ResponseGenerator:
    def __init__(
        self,
        corpus: pd.DataFrame,
        executor_params: dict,
        response_path: str,
    ):
        self.corpus = corpus
        self.tasks = {}
        self.candidates = {}
        self.responses = {}  # {"query_id": {"agent_id": (response, execution_time)}}
        self.response_path = response_path

        self.executor_params = executor_params
        self.executors: dict[str, PlatformExecutor] = {}
        # pre-emptive measure against uncaught system-wide erros, avoids polluting response data
        self.ERROR_LIMIT = 20

    def gen_responses(
        self,
        tasks: dict[str, tuple],
        candidates: dict[str, list[str]],
        no_exit: bool=False, # no need to re-set/re-login (must be exited manually after complete generation)
    ) -> dict[str, dict[str, tuple[str, float]]]:
        
        self.tasks = tasks
        self.candidates = candidates
        self.responses = {}
        error_count = 0

        # due to platform dependencies, group tasks by platform
        platform_tasks = self._assign_platforms()
        platforms = list(set([platform for tasks in platform_tasks.values() for _, platform in tasks])) 
        if self.executors == {}: self._initialize_executors(platforms)

        try:
            for qid, tasks in platform_tasks.items():
                self.responses[qid] = {}
                logger.info(f"Processing {qid} with {len(tasks)} candidate agents...", extra={"importance": "high"})
            
                for agent_id, platform_name in tasks:
                    if error_count >= self.ERROR_LIMIT:
                        raise ErrorLimitExceeded(f"Accumulated error limit of {self.ERROR_LIMIT} reached.")

                    network_down = self._is_network_down()
                    while network_down:
                        logger.warning("Network appears to be down. Retrying in 30 seconds...", extra={"importance": "high"})
                        time.sleep(30)
                        network_down = self._is_network_down()

                    executor = self.executors[platform_name]
                    agent_metadata = self.corpus.loc[
                            self.corpus.agent_id.eq(agent_id)
                        ].to_dict(orient="records")[0]
                    query_text = self.tasks[qid][0]

                    response, execution_time, status_code = executor.execute(
                        agent_metadata, query_text
                    )
                    error_count = error_count + 1 if status_code >= 400 else 0
                    self.responses[qid][agent_id] = (response, execution_time)
                    time.sleep(1)
        
        finally:
            if not no_exit:
                self._cleanup_executors()
        return self.responses

    def _assign_platforms(self) -> dict[str, list[tuple[str, str]]]:
        platform_tasks = {}
        for qid, agent_ids in self.candidates.items():
            platform_tasks[qid] = []

            for agent_id in agent_ids:
                agent = self.corpus.loc[self.corpus.agent_id.eq(agent_id)].iloc[0]
                platform_name = agent["platform_name"]
                platform_tasks[qid].append((agent_id, platform_name)) 
        return platform_tasks

    def _initialize_executors(self, platform_names: list[str]):
        for platform_name in platform_names:
            credential = self.executor_params.get(platform_name)
            params = {"credential": credential, "debug": True}

            logger.info(f"Initializing executor for {platform_name}...")
            executor = ExecutorFactory.create_executor(
                platform_name, **params
            )
            executor.setup()
            self.executors[platform_name] = executor

    def _is_network_down(self) -> bool:
        try:
            socket.setdefaulttimeout(3)
            socket.getaddrinfo("chatgpt.com", 443) # any platform domain
            return False
        except socket.gaierror:
            return True

    def _cleanup_executors(self):
        for platform_name, executor in self.executors.items():
            logger.info(f"Cleaning up executor for {platform_name}...")
            executor.teardown()


class ErrorLimitExceeded(Exception):
    pass