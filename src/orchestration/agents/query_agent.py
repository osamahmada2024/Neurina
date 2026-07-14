from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...config import settings
from ...helpers.AgentTools.ollama_client import OllamaClient


class QueryAgent(BaseAgent):

    def __init__(self, ollama_client: OllamaClient | None = None):
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="QueryAgent",
            ollama_client=ollama_client,
        )

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_input = state.get("style_description") or state.get("user_input", "").strip()

        if not user_input:
            raise ValueError("user_input or style_description is required in state")

        primary_query = await self._generate_query(state)
        state["generated_search_query"] = primary_query

        self.logger.log_tool_call(
            "QueryGeneration",
            {"user_input": user_input, "retry_count": state.get("retry_count", 0)},
            output_summary={"query": primary_query},
        )

        return state

    async def _generate_query(self, state: AgentState) -> str:
        user_input = state.get("style_description") or state.get("user_input", "")
        retry_count = state.get("retry_count", 0)
        
        if retry_count > 0:
            prompt = f"""You previously searched for "{state.get('generated_search_query')}" based on "{user_input}" but no good images were found.
Generate a NEW, DIFFERENT search query (max 10 words) that might yield better results.
Focus on: celebrity names, clear styles, simple adjectives.
Return ONLY the query."""
        else:
            prompt = f"""Convert to search query (max 10 words): "{user_input}"
Focus: celebrity names, styles, adjectives.
Return ONLY the query."""

        try:
            query = self.query_llm(prompt).strip()
            if query and len(query) >= 2:
                return query
        except Exception:
            pass

        return user_input
