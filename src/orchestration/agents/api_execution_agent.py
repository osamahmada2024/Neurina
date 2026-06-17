from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...helpers.AgentTools.serper_images import search_images
from ...config import settings
import uuid


class APIExecutionAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="APIExecutionAgent",
        )

    async def think_and_act(self, state: AgentState) -> AgentState:

        search_query = state.get("generated_search_query", "").strip()

        if not search_query:
            raise ValueError("generated_search_query is required in state")

        try:
            max_candidates = int(settings.MAX_REFERENCE_CANDIDATES or 10)

            # Search for images
            image_urls = await search_images(
                query=search_query,
                num_results=max_candidates,
                retry_count=getattr(settings, "MAX_RETRIES", 3)
            )

            if not image_urls:
                self.logger.log_tool_call(
                    "SerperImageSearch",
                    {"query": search_query, "num_results": max_candidates},
                    output_summary={"found_images": 0},
                )
                state["candidate_images"] = {}
                return state

            # Create candidate dict with unique IDs
            candidate_images = {
                str(uuid.uuid4()): url
                for url in image_urls
            }

            # Log the search
            self.logger.log_tool_call(
                "SerperImageSearch",
                {"query": search_query, "num_results": max_candidates},
                output_summary={"found_images": len(candidate_images)},
            )

            state["candidate_images"] = candidate_images
            return state

        except Exception as e:
            self.logger.log_tool_call(
                "SerperImageSearch",
                {"query": search_query},
                error=str(e),
            )
            state["candidate_images"] = {}
            raise
