# Generic LLM Inference Class
import logging
import asyncio
import litellm
from litellm import batch_completion
from litellm import batch_completion_models_all_responses


class LLMInference:
    def __init__(
            self,
            model_name,
            api_key: str,
            temperature: float,
    ):
        
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature

        # workaround to surpress litellm logs.
        loggers = ["LiteLLM Proxy", "LiteLLM Router", "LiteLLM", "httpx"]
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1) 

    def invoke(
            self,
            messages: list[dict],
    ) -> dict:
        
        response = litellm.completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=messages,
            temperature=self.temperature,
        )
        return response["choices"][0]["message"]

    def batch_invoke(
            self,
            batch_messages: list[list[dict]],
    ) -> list[dict]:
        responses = litellm.batch_completion(
            model=self.model_name,
            api_key=self.api_key,
            messages=batch_messages,
            temperature=self.temperature,
            cache_control_injection_points=[
                {
                    "location": "message",
                    "role": "system",
                }
            ],
        )
        return [response["choices"][0]["message"] for response in responses]
