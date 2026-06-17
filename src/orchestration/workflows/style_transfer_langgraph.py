"""LangGraph workflow for style-transfer chat with selection routing and streaming hooks."""

from __future__ import annotations

from typing import Literal

from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph

from ...helpers.AgentTools.image_routes_tool import ImageRoutesTool
from ...schemes.agent_state import AgentState
from ..agents.api_execution_agent import APIExecutionAgent
from ..agents.intent_agent import IntentAgent
from ..agents.quality_control_agent import QualityControlAgent
from ..agents.query_agent import QueryAgent
from ..agents.rag_agent import RAGAgent
from ..agents.reference_selector_agent import ReferenceSelectorAgent
from ..agents.selection_router import SelectionRouterAgent, _apply_candidate_pick
from ...config import settings


def _emit(payload: dict) -> None:
    try:
        writer = get_stream_writer()
        writer(payload)
    except Exception:
        pass


def _route_after_selection(state: AgentState) -> Literal["process_selection", "intent_flow"]:
    if state.get("route") == "process_selection":
        return "process_selection"
    return "intent_flow"


def _route_after_intent(state: AgentState) -> Literal[
    "answer_question",
    "await_source",
    "web_search",
    "await_selection",
    "process_selection",
    "done",
]:
    route = state.get("route")
    if route == "process_selection":
        return "process_selection"
    intent = state.get("intent") or "new_request"
    if intent == "general_question":
        return "answer_question"
    if not (state.get("source_image_id") or "").strip():
        return "await_source"
    if intent == "select_candidate":
        return "await_selection"
    if intent in ("new_request", "refine_prompt", "upload_source"):
        return "web_search"
    return "done"


def build_style_transfer_graph():
    selection_router = SelectionRouterAgent()
    intent_agent = IntentAgent()
    rag_agent = RAGAgent()
    query_agent = QueryAgent()
    api_agent = APIExecutionAgent()
    quality_agent = QualityControlAgent()
    reference_agent = ReferenceSelectorAgent()

    async def selection_router_node(state: AgentState) -> AgentState:
        _emit({"step": "routing", "action": "evaluate_selection"})
        updated = await selection_router(state)
        if updated.get("route") == "process_selection":
            _emit(
                {
                    "step": "selection_resolved",
                    "candidate_id": updated.get("selected_candidate_id"),
                    "status": "PROCESSING",
                }
            )
        return updated

    async def intent_node(state: AgentState) -> AgentState:
        _emit({"step": "intent_analysis", "action": "classify_message"})
        updated = await intent_agent(state)
        intent = updated.get("intent")
        if intent in ("new_request", "upload_source"):
            updated["retry_count"] = 0
            updated["errors"] = []
        elif intent == "refine_prompt":
            updated["retry_count"] = int(updated.get("retry_count") or 0) + 1
        elif intent == "select_candidate":
            idx = int(updated.get("selected_index") or -1)
            items = list((updated.get("candidate_images") or {}).items())
            if 0 <= idx < len(items):
                cid, url = items[idx]
                updated = _apply_candidate_pick(updated, cid, url)
            else:
                updated["status"] = "AWAITING_SELECTION"
                if "Invalid selection index" not in updated.get("errors", []):
                    updated.setdefault("errors", []).append("Invalid selection index")
        if not (updated.get("source_image_id") or "").strip() and updated.get("route") != "process_selection":
            updated["status"] = "AWAITING_SOURCE"
            if "Source image missing" not in updated.get("errors", []):
                updated.setdefault("errors", []).append("Source image missing")
        _emit({"step": "intent_resolved", "intent": intent})
        return updated

    async def rag_node(state: AgentState) -> AgentState:
        _emit({"step": "tool_call", "tool": "rag_answer"})
        return await rag_agent(state)

    async def web_search_node(state: AgentState) -> AgentState:
        max_retries = int(getattr(settings, "MAX_RETRIES", 3))
        retry = int(state.get("retry_count") or 0)

        while retry < max_retries:
            _emit({"step": "tool_call", "tool": "query_generation", "retry": retry})
            state = await query_agent(state)
            query = state.get("generated_search_query", "")
            _emit({"step": "tool_call", "tool": "web_search", "query": query, "retry": retry})
            try:
                state = await api_agent(state)
                _emit({"step": "tool_call", "tool": "quality_scoring"})
                state = await quality_agent(state)
                candidates = state.get("candidate_images") or {}
                if candidates:
                    state["status"] = "AWAITING_SELECTION"
                    _emit(
                        {
                            "step": "yielding_candidates",
                            "data": {
                                "count": len(candidates),
                                "candidate_images": candidates,
                                "quality_score": state.get("quality_score"),
                            },
                        }
                    )
                    return state
            except Exception as exc:
                state.setdefault("errors", []).append(str(exc))
            retry += 1
            state["retry_count"] = retry

        state["status"] = "FAILED"
        state.setdefault("errors", []).append(
            f"Failed to find suitable images after {max_retries} attempts."
        )
        _emit({"step": "error", "message": state["errors"][-1]})
        return state

    async def process_selection_node(state: AgentState) -> AgentState:
        state["status"] = "PROCESSING"
        _emit(
            {
                "step": "uploading_reference",
                "candidate_id": state.get("selected_candidate_id"),
            }
        )
        state = await reference_agent(state)
        mongo_ref = state.get("reference_image_id")
        source_id = (state.get("source_image_id") or "").strip()
        auth = state.get("auth_token") or ""

        if source_id and mongo_ref and auth:
            _emit(
                {
                    "step": "translating_image",
                    "source_image_id": source_id,
                    "reference_image_id": mongo_ref,
                }
            )
            tool = ImageRoutesTool(token=auth)
            translated_id = tool.translate_images(
                source_image_id=source_id,
                reference_image_id=mongo_ref,
            )
            state["translated_image_id"] = translated_id
            state["translation_task_id"] = translated_id
            state["status"] = "COMPLETED"
            _emit({"step": "translation_complete", "translated_image_id": translated_id})
        else:
            state["status"] = "COMPLETED"
            _emit({"step": "reference_ready", "reference_image_id": mongo_ref})
        return state

    async def await_source_node(state: AgentState) -> AgentState:
        state["status"] = "AWAITING_SOURCE"
        return state

    async def await_selection_node(state: AgentState) -> AgentState:
        if state.get("status") != "AWAITING_SELECTION":
            state["status"] = "AWAITING_SELECTION"
        return state

    graph = StateGraph(AgentState)

    graph.add_node("selection_router", selection_router_node)
    graph.add_node("intent_analysis", intent_node)
    graph.add_node("answer_question", rag_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("process_selection", process_selection_node)
    graph.add_node("await_source", await_source_node)
    graph.add_node("await_selection", await_selection_node)

    graph.set_entry_point("selection_router")

    graph.add_conditional_edges(
        "selection_router",
        _route_after_selection,
        {
            "process_selection": "process_selection",
            "intent_flow": "intent_analysis",
        },
    )

    graph.add_conditional_edges(
        "intent_analysis",
        _route_after_intent,
        {
            "answer_question": "answer_question",
            "await_source": "await_source",
            "web_search": "web_search",
            "await_selection": "await_selection",
            "process_selection": "process_selection",
            "done": END,
        },
    )

    graph.add_edge("answer_question", END)
    graph.add_edge("web_search", END)
    graph.add_edge("process_selection", END)
    graph.add_edge("await_source", END)
    graph.add_edge("await_selection", END)

    return graph.compile()
