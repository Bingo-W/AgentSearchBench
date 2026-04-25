# Agent AI Network executor

import logging
import time
import requests
import random
from asbench.generation.responses.base import PlatformExecutor

logger = logging.getLogger(__name__)

class AgentAIExecutor(PlatformExecutor):
    def setup(self):
        self.url = "https://api-lr.agent.ai/v1/action/invoke_agent"
        self.api_keys = self.credential_info
        self._switch_api_key()

    def execute(self, agent_metadata: dict, query: str) -> tuple[str, float, int]:
        response = ""
        execution_time = 0.0
        status_code = 200

        network_id = self._get_network_id(agent_metadata)
        payload = {"id": network_id, "user_input": query}
        if not network_id:
            return "Error: No agent network ID found", 0.0, 400

        start_time = time.time()
        try:
            if self.debug: logger.info(f"Executing agent {agent_metadata['agent_id']}")
            api_response = requests.post(self.url, json=payload, headers=self.headers).json()

            if api_response.get("status") != 200:
                logger.error(f"API call failed with status {api_response.get('status')}: {api_response.get('error')}")
                self._switch_api_key() # soft enforcement (e.g., avoid rate limits)
        
            response = api_response.get("response", "")
            execution_time = time.time() - start_time
            status_code = api_response.get("status", 500)

        except Exception as e:
            response = f"Exception during API call: {e}"
            status_code = 500
            logger.error(response)

        response_formatted = f"Text: {response}, Files: [], Images: []"
        return response_formatted, execution_time, status_code

    def _get_network_id(self, agent_metadata: dict) -> str:
        misc_fields = agent_metadata.get("misc", "").split("; ")
        for field in misc_fields:
            if field.startswith("agentnetworkid: "):
                return field.replace("agentnetworkid: ", "").strip()
        return ""
    
    def _switch_api_key(self):
        """ Switch to a different available API key in case of rate/credit limits """
        self.api_key = random.choice(self.api_keys)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def teardown(self):
        pass
