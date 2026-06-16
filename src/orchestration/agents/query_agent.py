from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...config import settings


class QueryAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="QueryAgent",
        )

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "").strip()

        if not user_input:
            raise ValueError("user_input is required in state")

        primary_query = await self._generate_query(user_input)
        state["generated_search_query"] = primary_query

        self.logger.log_tool_call(
            "QueryGeneration",
            {"user_input": user_input},
            output_summary={"query": primary_query},
        )

        return state

    async def _generate_query(self, user_input: str) -> str:
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
