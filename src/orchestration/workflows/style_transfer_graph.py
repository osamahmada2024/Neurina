from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional

from bson import ObjectId

from ...helpers.AgentTools.image_routes_tool import ImageRoutesTool
from ...helpers.AgentTools.logger import AgentLogger
from ...models.Enums import ImageType
from ...models.database.database import database
from ...schemes.agent_state import AgentState
from ..agents import SupervisorAgent
from ..agents.reference_selector_agent import ReferenceSelectorAgent


class StyleTransferGraph:
    def __init__(self) -> None:
        self.supervisor_agent = SupervisorAgent()
        self.reference_selector_agent = ReferenceSelectorAgent()
        self.logger = AgentLogger()
        self.collection = database["agent_sessions"]

    @staticmethod
    def _new_state(
        user_id: ObjectId,
        session_id: str,
        user_input: str,
        auth_token: str,
        source_image_id: Optional[str] = None,
    ) -> AgentState:
        return {
            "user_input": user_input,
            "source_image_id": (source_image_id or "").strip(),
            "auth_token": auth_token,
            "user_id": str(user_id),
            "session_id": session_id,
            "status": "NEW",
            "chat_history": [],
            "retry_count": 0,
            "intent": None,
            "style_description": None,
            "selected_index": None,
            "rag_response": None,
            "generated_search_query": None,
            "candidate_images": None,
            "selected_reference_url": None,
            "uploaded_reference_url": None,
            "reference_image_id": None,
            "translation_task_id": None,
            "translated_image_id": None,
            "final_output_url": None,
            "quality_score": None,
            "errors": [],
            "route": None,
            "selected_candidate_id": None,
        }

    @staticmethod
    def _ensure_state_shape(state: dict) -> AgentState:
        defaults = StyleTransferGraph._new_state(
            user_id=ObjectId("000000000000000000000000"),
            session_id="",
            user_input="",
            auth_token="",
        )
        merged = {**defaults, **state}
        merged.setdefault("chat_history", [])
        merged.setdefault("errors", [])
        return merged  # type: ignore[return-value]

    async def _load_session(
        self,
        session_id: str,
        user_id: ObjectId,
    ) -> tuple[Optional[dict], Optional[AgentState]]:
        doc = await self.collection.find_one({"session_id": session_id})
        if not doc:
            return None, None
        owner = str(doc.get("user_id", ""))
        if owner and owner != str(user_id):
            raise PermissionError("Session does not belong to this user")
        state = self._ensure_state_shape(doc.get("state") or {})
        return doc, state

    async def _save_session(
        self,
        session_id: str,
        user_id: ObjectId,
        state: AgentState,
        status: str,
    ) -> None:
        await self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "session_id": session_id,
                    "user_id": str(user_id),
                    "state": state,
                    "status": status,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

    async def _validate_source_image(
        self,
        user_id: ObjectId,
        source_image_id: str,
    ) -> bool:
        sid = (source_image_id or "").strip()
        if not sid or not ObjectId.is_valid(sid):
            return False
        doc = await database["images"].find_one(
            {
                "_id": ObjectId(sid),
                "user_id": user_id,
                "image_type": ImageType.SOURCE.value,
            }
        )
        return doc is not None

    async def _validate_reference_image(
        self,
        user_id: ObjectId,
        reference_image_id: str,
    ) -> bool:
        rid = (reference_image_id or "").strip()
        if not rid or not ObjectId.is_valid(rid):
            return False
        doc = await database["images"].find_one(
            {
                "_id": ObjectId(rid),
                "user_id": user_id,
                "image_type": ImageType.REFERENCE.value,
            }
        )
        return doc is not None

    def _build_response(self, final_state: AgentState, status: str, message: str) -> Dict[str, Any]:
        return {
            "success": status not in ("FAILED", "AWAITING_SOURCE"),
            "status": status,
            "message": message,
            "candidate_images": final_state.get("candidate_images") or {},
            "reference_image_id": final_state.get("reference_image_id"),
            "translated_image_id": final_state.get("translated_image_id"),
            "selected_reference_url": final_state.get("selected_reference_url"),
            "quality_score": final_state.get("quality_score"),
            "errors": final_state.get("errors", []),
        }

    async def execute(
        self,
        user_id: ObjectId,
        session_id: str,
        user_input: str,
        auth_token: str,
        source_image_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        _, existing = await self._load_session(session_id, user_id)

        if existing is not None:
            state = existing
            state["user_input"] = user_input
            if source_image_id and source_image_id.strip():
                state["source_image_id"] = source_image_id.strip()
            state["auth_token"] = auth_token
        else:
            state = self._new_state(
                user_id=user_id,
                session_id=session_id,
                user_input=user_input,
                auth_token=auth_token,
                source_image_id=source_image_id,
            )

        state["chat_history"].append({"role": "user", "content": user_input})

        try:
            final_state = await self.supervisor_agent(state)
            status = final_state.get("status", "FAILED")
            message = self._status_message(status, final_state)

            final_state["chat_history"].append({"role": "agent", "content": message})
            await self._save_session(session_id, user_id, final_state, status)

            return self._build_response(final_state, status, message)

        except Exception as e:
            self.logger.log_step(
                "StyleTransferGraph_ERROR",
                {"error": str(e), "user_id": str(user_id)},
                level="ERROR",
            )
            return {
                "success": False,
                "status": "FAILED",
                "message": f"Workflow failed: {str(e)}",
                "errors": [str(e)],
            }

    async def execute_stream(
        self,
        user_id: ObjectId,
        session_id: str,
        user_input: str,
        auth_token: str,
        source_image_id: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Server-Sent Events stream of agent steps and final session result."""
        _, existing = await self._load_session(session_id, user_id)

        if existing is not None:
            state = existing
            state["user_input"] = user_input
            if source_image_id and source_image_id.strip():
                state["source_image_id"] = source_image_id.strip()
            state["auth_token"] = auth_token
        else:
            state = self._new_state(
                user_id=user_id,
                session_id=session_id,
                user_input=user_input,
                auth_token=auth_token,
                source_image_id=source_image_id,
            )

        state["chat_history"].append({"role": "user", "content": user_input})
        final_state: AgentState = state

        def _sse(payload: dict) -> str:
            return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

        yield _sse({"step": "session_start", "session_id": session_id})

        try:
            async for event in self.supervisor_agent.astream_events(state):
                etype = event.get("event")
                name = event.get("name", "")

                if etype == "on_custom_event":
                    data = event.get("data")
                    if isinstance(data, dict):
                        yield _sse(data)
                    continue

                if etype == "on_chain_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if isinstance(chunk, dict):
                        yield _sse({"step": "node_update", "node": name, "data": chunk})
                    continue

                if etype == "on_chain_end" and event.get("data", {}).get("output"):
                    output = event["data"]["output"]
                    if isinstance(output, dict) and name in (
                        "selection_router",
                        "intent_analysis",
                        "web_search",
                        "process_selection",
                        "answer_question",
                        "await_source",
                        "await_selection",
                    ):
                        final_state = output
                        yield _sse(
                            {
                                "step": "node_complete",
                                "node": name,
                                "status": output.get("status"),
                            }
                        )

            status = final_state.get("status", "FAILED")
            message = self._status_message(status, final_state)
            final_state["chat_history"].append({"role": "agent", "content": message})
            await self._save_session(session_id, user_id, final_state, status)

            result = self._build_response(final_state, status, message)
            yield _sse({"step": "complete", "data": result})

        except Exception as e:
            self.logger.log_step(
                "StyleTransferGraph_STREAM_ERROR",
                {"error": str(e), "user_id": str(user_id)},
                level="ERROR",
            )
            yield _sse(
                {
                    "step": "error",
                    "message": str(e),
                    "data": {
                        "success": False,
                        "status": "FAILED",
                        "message": f"Workflow failed: {str(e)}",
                        "errors": [str(e)],
                    },
                }
            )

    async def execute_request(
        self,
        user_id: ObjectId,
        session_id: str,
        source_image_id: str,
        reference_image_id: str,
        auth_token: str,
        style_description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Resume an existing chat session: validate source, resolve reference (candidate UUID
        or uploaded reference id), upload if needed, then call /translate.
        """
        _, state = await self._load_session(session_id, user_id)
        if state is None:
            state = self._new_state(
                user_id=user_id,
                session_id=session_id,
                user_input=style_description or "",
                auth_token=auth_token,
                source_image_id=source_image_id,
            )
        else:
            state["auth_token"] = auth_token

        state["errors"] = []
        state["source_image_id"] = (source_image_id or state.get("source_image_id") or "").strip()
        if style_description and style_description.strip():
            state["style_description"] = style_description.strip()
            state["user_input"] = style_description.strip()

        if not state["source_image_id"]:
            state["status"] = "AWAITING_SOURCE"
            state["errors"].append("Source image missing")
            message = self._status_message("AWAITING_SOURCE", state)
            await self._save_session(session_id, user_id, state, "AWAITING_SOURCE")
            return self._build_response(state, "AWAITING_SOURCE", message)

        if not await self._validate_source_image(user_id, state["source_image_id"]):
            state["status"] = "AWAITING_SOURCE"
            state["errors"].append(
                f"Source image '{state['source_image_id']}' was not found for this user"
            )
            message = self._status_message("AWAITING_SOURCE", state)
            await self._save_session(session_id, user_id, state, "AWAITING_SOURCE")
            return self._build_response(state, "AWAITING_SOURCE", message)

        ref_key = (reference_image_id or "").strip()
        if not ref_key:
            state["status"] = "AWAITING_SELECTION"
            state["errors"].append("reference_image_id is required")
            message = self._status_message("AWAITING_SELECTION", state)
            await self._save_session(session_id, user_id, state, "AWAITING_SELECTION")
            return self._build_response(state, "AWAITING_SELECTION", message)

        candidates = state.get("candidate_images") or {}
        mongo_ref_id: Optional[str] = None

        try:
            state["status"] = "PROCESSING"

            if ref_key in candidates:
                state["selected_reference_url"] = candidates[ref_key]
                state = await self.reference_selector_agent(state)
                mongo_ref_id = state.get("reference_image_id")
            elif await self._validate_reference_image(user_id, ref_key):
                mongo_ref_id = ref_key
                state["reference_image_id"] = ref_key
            elif state.get("reference_image_id") and str(state["reference_image_id"]) == ref_key:
                mongo_ref_id = ref_key
            else:
                state["status"] = "AWAITING_SELECTION"
                state["errors"].append(
                    "reference_image_id is not a known candidate id and was not found in your library"
                )
                message = self._status_message("AWAITING_SELECTION", state)
                await self._save_session(session_id, user_id, state, "AWAITING_SELECTION")
                return self._build_response(state, "AWAITING_SELECTION", message)

            if not mongo_ref_id:
                raise ValueError("Failed to resolve reference image id")

            image_tool = ImageRoutesTool(token=auth_token)
            translated_id = image_tool.translate_images(
                source_image_id=state["source_image_id"],
                reference_image_id=mongo_ref_id,
            )
            state["translated_image_id"] = translated_id
            state["translation_task_id"] = translated_id
            state["status"] = "COMPLETED"
            message = "Style transfer completed successfully."
            await self._save_session(session_id, user_id, state, "COMPLETED")
            return self._build_response(state, "COMPLETED", message)

        except Exception as e:
            state["status"] = "FAILED"
            state["errors"].append(str(e))
            message = self._status_message("FAILED", state)
            await self._save_session(session_id, user_id, state, "FAILED")
            self.logger.log_step(
                "StyleTransferGraph_REQUEST_ERROR",
                {"error": str(e), "user_id": str(user_id), "session_id": session_id},
                level="ERROR",
            )
            return self._build_response(state, "FAILED", message)

    @staticmethod
    def _status_message(status: str, final_state: AgentState) -> str:
        if status == "AWAITING_SOURCE":
            return "Please upload your source photo so I can apply the style."
        if status == "AWAITING_SELECTION":
            return (
                "I found these candidate images based on your prompt. Please select the best one "
                "by replying with the image number (e.g., 'option 1', 'second one'), or tell me "
                "they are bad to search again."
            )
        if status == "PROCESSING":
            return "Great choice! I am processing your image now..."
        if status == "COMPLETED":
            if final_state.get("translated_image_id"):
                return "Style transfer completed successfully."
            return "Reference image selected successfully!"
        if status == "ANSWERED_QUESTION":
            return final_state.get("rag_response", "I don't know")
        if status == "FAILED":
            return "I'm sorry, I couldn't complete the request. " + ", ".join(
                final_state.get("errors", [])
            )
        return ""
