import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from ...schemes.agent_state import AgentState
from ...helpers.AgentTools.logger import AgentLogger
from ...helpers.AgentTools.ollama_client import OllamaClient, ask_ollama


class BaseAgent(ABC):

    def __init__(
        self,
        model_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        ollama_client: Optional[OllamaClient] = None,
    ):

        self.model_name = model_name or self.__class__.__name__
        self.agent_name = agent_name or self.__class__.__name__
        self.ollama_client = ollama_client
        self.logger = AgentLogger()

    async def __call__(self, state: AgentState) -> AgentState:

        start_time = time.time()

        try:
            # Log execution start
            self.logger.log_agent_start(
                self.agent_name,
                state.get("user_id", "unknown"),
                list(state.keys()),
            )

            before_state = dict(state)

            # Execute core agent logic
            updated_state = await self.think_and_act(state)

            # Log successful completion
            duration = time.time() - start_time
            updated_keys = [
                k for k in updated_state.keys()
                if updated_state.get(k) != before_state.get(k)
            ]
            self.logger.log_agent_end(
                self.agent_name,
                state.get("user_id", "unknown"),
                updated_keys,
                duration,
            )

            return updated_state

        except Exception as e:
            # Log failure
            duration = time.time() - start_time
            self.logger.log_step(
                f"{self.agent_name}_ERROR",
                {
                    "error": str(e),
                    "duration_seconds": round(duration, 2)
                },
                level="ERROR",
                error=str(e),
            )

            # Track error in state
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"{self.agent_name}: {str(e)}")

            raise

    @abstractmethod
    async def think_and_act(self, state: AgentState) -> AgentState:

        pass

    def query_llm(self, prompt: str) -> str:

        try:
            response = ask_ollama(self.model_name, prompt, client=self.ollama_client)
            return response
        except Exception as e:
            self.logger.log_tool_call(
                "OllamaLLM",
                {
                    "model": self.model_name,
                    "prompt_length": len(prompt)
                },
                error=str(e),
            )
            raise

    def parse_json_from_llm(
        self,
        response: str,
        output_schema: Optional[BaseModel] = None
    ) -> Dict[str, Any]:
      
        try:
            parser = JsonOutputParser(pydantic_object=output_schema)
            parsed = parser.parse(response)
            return parsed
        except Exception as e:
            self.logger.log_tool_call(
                "JsonParsing",
                {"response_length": len(response)},
                error=str(e),
            )
            # Fallback: return empty dict
            return {}
