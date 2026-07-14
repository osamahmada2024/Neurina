from .base_agent import BaseAgent
from ..workflows.style_transfer_langgraph import build_style_transfer_graph
from ...config import settings
from ...schemes.agent_state import AgentState
from ...services.agent_image_service import AgentImageService
from ...helpers.AgentTools.ollama_client import OllamaClient
from .rag_agent import LazyRAGAgentProvider


class SupervisorAgent(BaseAgent):
    """
    Thin wrapper around the compiled LangGraph workflow.
    Nodes: selection_router → intent / web_search / process_selection / rag.
    """

    _compiled_graph = None

    def __init__(
        self,
        *,
        image_service: AgentImageService | None = None,
        rag_provider: LazyRAGAgentProvider | None = None,
        ollama_client: OllamaClient | None = None,
    ):
        super().__init__(
            model_name=settings.SUPERVISOR_MODEL,
            agent_name="SupervisorAgent",
            ollama_client=ollama_client,
        )
        if image_service is not None or rag_provider is not None or ollama_client is not None:
            self.graph = build_style_transfer_graph(
                image_service=image_service,
                rag_provider=rag_provider,
                ollama_client=ollama_client,
            )
            return

        if SupervisorAgent._compiled_graph is None:
            SupervisorAgent._compiled_graph = build_style_transfer_graph(
                image_service=None,
                rag_provider=None,
            )
        self.graph = SupervisorAgent._compiled_graph

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_id = state.get("user_id", "unknown")
        self.logger.log_workflow_start(user_id, state.get("user_input", ""))
        try:
            result = await self.graph.ainvoke(state)
            return result
        except Exception as e:
            state["status"] = "FAILED"
            state.setdefault("errors", []).append(str(e))
            self.logger.log_workflow_end(user_id, success=False, error=str(e))
            return state

    async def astream_events(self, state: AgentState):
        """Yield LangGraph v2 stream events plus custom writer payloads."""
        async for event in self.graph.astream_events(state, version="v2"):
            yield event

    async def astream_updates(self, state: AgentState):
        async for chunk in self.graph.astream(state, stream_mode="updates"):
            yield chunk
