# LLM-as-a-Judge for single-agent response quality evaluation

import logging
import time

from asbench.generation.judges.base import BaseLLMJudge
from asbench.generation.utils import extract_json


logger = logging.getLogger(__name__)

class SingleAgentJudge(BaseLLMJudge):
    def judge(
        self, queries: dict[str, tuple], responses: dict[str, dict[str, tuple]]
    ) -> dict[str, dict[str, tuple[int, str]]]:
        judged_labels = {}
        for query_id, (query_text, misc) in queries.items():
            judged_labels[query_id] = {}

            for agent_id, (response_content, *_) in responses.get(query_id, {}).items():
                response_content = self.enc.decode(self.enc.encode(str(response_content))[:self.token_limit])
                golden_answer = misc if self.use_golden_labels else None
                
                messages = self.prompt.get_messages(
                    include_reasoning=True,
                    query=query_text,
                    response=str(response_content),
                    golden_answer=golden_answer,
                )

                response_received = False
                while not response_received:
                    try:
                        llm_response = self.llm.invoke(messages).content
                        extracted_response = extract_json(llm_response)
                        score = extracted_response.get("score")
                        response_received = True
                    except Exception as e:
                        logger.error(f"Error invoking LLM or extracting response: {e}. Retrying in 10 seconds...", extra={"importance": "high"})
                        time.sleep(10)

                reasoning = extracted_response.get("reasoning", "")
                judged_labels[query_id][agent_id] = (score, reasoning)
                logger.info(f"{self.llm_model} Judged query {query_id}, agent {agent_id}{', and gold labels' if golden_answer else ''}: score={score}")

        return judged_labels
    
    def judge_batch(self, queries: dict[str, tuple], responses: dict[str, dict[str, tuple]]):        
        messages = self._get_batch_input(queries, responses)
        logger.info(f"Batch judging {len(queries)} tasks with {len(messages)} calls")
        llm_responses = self.llm.batch_invoke(messages)
        logger.info(f"Completed batch judging for {len(messages)} messages", extra={"importance": "high"})

        # process batch judged labels back to dict format
        judged_labels = self._get_batch_output(llm_responses)
        return judged_labels
    
    async def judge_schedule_batch(self, queries: dict[str, tuple], responses: dict[str, dict[str, tuple]]):
        assert self.llm_async is not None, "Async LLMInference instance is required for scheduled batch judging"
        
        messages = self._get_batch_input(queries, responses)
        logger.info(f"Scheduling batch judging for {len(messages)} messages")
        llm_responses = await self.llm_async.batch_schedule_openai(messages)
        logger.info(f"Completed batch judging for {len(messages)} messages", extra={"importance": "high"})

        # process batch judged labels back to dict format
        judged_labels = self._get_scheduled_batch_output(queries, responses, llm_responses)
        return judged_labels

    def _get_batch_input(
            self,
            queries: dict[str, tuple],
            responses: dict[str, dict[str, tuple]]
    ) -> list[list[dict]]:
        # we create flattened messages (input) of the task/response pairs
        messages = []
        for query_id, (query_text, misc) in queries.items():
            for agent_id, (response_content, *_) in responses.get(query_id, {}).items():
                response_content = self.enc.decode(self.enc.encode(str(response_content))[:self.token_limit])
                golden_answer = misc if self.config["experiment"]["hyperparameters"]["judging"]["use_golden_labels"] else None
                
                message = self.prompt.get_messages(
                    include_reasoning=True,
                    query=query_text,
                    response=str(response_content),
                    golden_answer=golden_answer,
                )
                messages.append(message)
        return messages
    
    def _get_batch_output(
            self,
            response: list[dict]
    ) -> dict[str, dict[str, tuple[int, str]]]:
        judged_labels = {}
        for llm_response in response:
            extracted_response = extract_json(llm_response["choices"][0]["message"]["content"])
            query_id = extracted_response.get("query_id")
            agent_id = extracted_response.get("agent_id")
            score = extracted_response.get("score", 1)
            reasoning = extracted_response.get("reasoning", "")

            if query_id not in judged_labels:
                judged_labels[query_id] = {}
            judged_labels[query_id][agent_id] = (score, reasoning)
            logger.info(f"Batch judged query {query_id}, agent {agent_id}: score={score}")
        return judged_labels
    
    def _get_scheduled_batch_output(
            self,
            queries: dict[str, tuple],
            responses: dict[str, dict[str, tuple]],
            batch_response: list[dict]
    ) -> dict[str, dict[str, tuple[int, str]]]:
        
        # order of batch_response corresponds to order of messages sent in _get_batch_input, which corresponds to order of query/response pairs in queries/responses dicts
        judged_labels = {}
        idx = 0
        for query_id, (query_text, misc) in queries.items():
            judged_labels[query_id] = {}

            for agent_id, (response_content, *_) in responses.get(query_id, {}).items():
                llm_response = batch_response[idx]
                extracted_response = extract_json(llm_response["choices"][0]["message"]["content"])
                score = extracted_response.get("score", 1)
                reasoning = extracted_response.get("reasoning", "")

                judged_labels[query_id][agent_id] = (score, reasoning)
                logger.info(f"Batch judged query {query_id}, agent {agent_id}: score={score}")
                idx += 1
        return judged_labels