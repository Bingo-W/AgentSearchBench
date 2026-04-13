# LLM-as-a-Judge for agent documentation to agent response consistency

import logging

from asbench.generation.judges.base import BaseLLMJudge
from asbench.generation.utils import extract_json
import time


logger = logging.getLogger(__name__)


class DocConsistencyJudge(BaseLLMJudge):
    def judge(
        self, documents: dict[str, dict[str, str]], responses: dict[str, dict[str, list[str]]]
    ) -> dict[str, dict[str, tuple[int, str]]]:
        judged_labels = {}

        for iid, agent_docs in documents.items():
            judged_labels[iid] = {}

            for agent_id, document in agent_docs.items():
                agent_responses = responses[iid].get(agent_id, [])
                agent_responses = [response for response, *_ in agent_responses if response]  # filter out empty responses
                response_content = self.enc.decode(self.enc.encode(str(agent_responses))[:self.token_limit])

                messages = self.prompt.get_messages(
                    include_reasoning=True,
                    document=document,
                    response=str(response_content),
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
                judged_labels[iid][agent_id] = (score, reasoning)
                logger.info(f"{self.llm_model} Judged query {iid}, agent {agent_id}: score={score}")

        return judged_labels
    
    def judge_batch(
        self, documents: dict[str, dict[str, str]], responses: dict[str, dict[str, list[str]]]
    ) -> dict[str, dict[str, tuple[int, str]]]:
        # For simplicity, we call the non-batch judge in this example. In practice, you would implement a batch version of the prompt and LLM invocation.
        return self.judge(documents, responses)