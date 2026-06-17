"""Deterministic + LLM candidate selection routing before web search."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from .base_agent import BaseAgent
from ...config import settings
from ...schemes.agent_state import AgentState


_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_ORDINAL_WORDS = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "fourth": 3,
    "4th": 3,
    "four": 3,
    "fifth": 4,
    "5th": 4,
    "five": 4,
}

_OPTION_NUM_RE = re.compile(
    r"(?:option|choice|pick|select|number|#|no\.?|image)\s*(\d+)",
    re.IGNORECASE,
)


class SemanticSelectionAnalysis(BaseModel):
    selected_index: int = Field(
        description="0-based index of chosen candidate, or -1 if not a selection"
    )
    is_selection: bool = Field(
        description="True when the user is picking a shown candidate image"
    )


def _normalize_user_text(text: str) -> str:
    return (text or "").strip().strip('"\'`')


def _looks_like_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))


def _parse_ordinal_index(text: str, max_items: int) -> Optional[int]:
    lowered = text.lower().strip()
    for word, idx in _ORDINAL_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return idx if idx < max_items else None
    m = _OPTION_NUM_RE.search(lowered)
    if m:
        idx = int(m.group(1)) - 1  # users say "option 1" -> index 0
        return idx if 0 <= idx < max_items else None
    if lowered.isdigit():
        idx = int(lowered) - 1
        return idx if 0 <= idx < max_items else None
    return None


def _apply_candidate_pick(state: AgentState, candidate_id: str, url: str) -> AgentState:
    state["selected_candidate_id"] = candidate_id
    state["selected_reference_url"] = url
    state["intent"] = "select_candidate"
    state["route"] = "process_selection"
    state["status"] = "PROCESSING"
    if "Invalid selection index" in state.get("errors", []):
        state["errors"] = [e for e in state["errors"] if e != "Invalid selection index"]
    return state


class SelectionRouterAgent(BaseAgent):
    """Runs before web search: UUID match, ordinal/semantic selection, or defer to search."""

    def __init__(self) -> None:
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="SelectionRouterAgent",
        )

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_input = _normalize_user_text(state.get("user_input", ""))
        candidates: dict[str, str] = dict(state.get("candidate_images") or {})
        items = list(candidates.items())

        if not user_input:
            state["route"] = "intent_flow"
            return state

        # 1) Exact UUID key match (bypass search entirely)
        if candidates and user_input in candidates:
            self.logger.log_step(
                "SelectionRouter",
                {"decision": "uuid_exact_match", "candidate_id": user_input},
            )
            return _apply_candidate_pick(state, user_input, candidates[user_input])

        # Bare UUID pasted but key might differ in casing
        if _looks_like_uuid(user_input):
            for cid, url in items:
                if cid.lower() == user_input.lower():
                    self.logger.log_step(
                        "SelectionRouter",
                        {"decision": "uuid_case_insensitive", "candidate_id": cid},
                    )
                    return _apply_candidate_pick(state, cid, url)

        # 2) Ordinal / option N heuristics when we have candidates
        if items and state.get("status") in ("AWAITING_SELECTION", "PROCESSING", "COMPLETED", "NEW"):
            idx = _parse_ordinal_index(user_input, len(items))
            if idx is not None:
                cid, url = items[idx]
                self.logger.log_step(
                    "SelectionRouter",
                    {"decision": "ordinal_match", "index": idx, "candidate_id": cid},
                )
                return _apply_candidate_pick(state, cid, url)

        # 3) LLM semantic selection when candidates exist
        if items and state.get("status") == "AWAITING_SELECTION":
            picked = await self._llm_resolve_selection(user_input, items)
            if picked is not None:
                cid, url = picked
                self.logger.log_step(
                    "SelectionRouter",
                    {"decision": "llm_semantic", "candidate_id": cid},
                )
                return _apply_candidate_pick(state, cid, url)

        state["route"] = "intent_flow"
        return state

    async def _llm_resolve_selection(
        self,
        user_input: str,
        items: list[tuple[str, str]],
    ) -> Optional[tuple[str, str]]:
        numbered = "\n".join(
            f"{i + 1}. id={cid} url={url[:80]}..." if len(url) > 80 else f"{i + 1}. id={cid} url={url}"
            for i, (cid, url) in enumerate(items)
        )
        prompt = f"""The user is viewing numbered reference image candidates and replied:
"{user_input}"

Candidates (1-based display order):
{numbered}

If the user is selecting one of these images (e.g. "the second one", "I like option 3", "pick the first"),
return JSON: {{"is_selection": true, "selected_index": <0-based index>}}
If they are NOT selecting (new style request, rejection, question), return:
{{"is_selection": false, "selected_index": -1}}

Respond ONLY with JSON."""

        try:
            response = self.query_llm(prompt)
            parsed = self.parse_json_from_llm(response, SemanticSelectionAnalysis)
            if not parsed.get("is_selection"):
                return None
            idx = int(parsed.get("selected_index", -1))
            if 0 <= idx < len(items):
                return items[idx]
        except Exception as exc:
            self.logger.log_step(
                "SelectionRouter_LLM",
                {"error": str(exc)},
                level="WARNING",
            )
        return None
