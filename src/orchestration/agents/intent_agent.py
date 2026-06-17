from pydantic import BaseModel, Field
from typing import Dict, Any
from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...config import settings

class IntentAnalysis(BaseModel):
    intent: str = Field(description="One of: 'new_request', 'refine_prompt', 'select_candidate', 'upload_source', 'general_question'")
    extracted_prompt: str = Field(description="The user's style prompt if present, or empty string")
    selected_index: int = Field(description="The index (0-based) of the selected candidate image if applicable, or -1")

class IntentAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="IntentAgent",
        )

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_input = state.get("user_input", "")
        chat_history = state.get("chat_history", [])
        status = state.get("status", "NEW")
        
        prompt = f"""You are analyzing a user's message in an AI assistant application.
Current Status: {status}
User Message: "{user_input}"

Determine the user's intent from the following:
- new_request: The user is requesting an image style transfer or transformation (e.g. "I want my photo in Tamer Hosny style").
- refine_prompt: The user is providing a better prompt because previous image searches failed (e.g. "Try finding Tamer Hosny 2010 concert").
- select_candidate: The user is selecting an image from the shown candidates (e.g. "I choose the second one", "option 1").
- upload_source: The user is notifying that they uploaded their photo.
- general_question: The user is asking a general question about the website, app, or documentation, and NOT requesting an image transformation (e.g. "How does this work?", "What features do you have?").

Extract the style prompt if they are describing a style for transformation.
If they are selecting a candidate, extract the 0-based index.

Respond ONLY in JSON matching this schema:
{{"intent": "...", "extracted_prompt": "...", "selected_index": 0}}
"""
        response = self.query_llm(prompt)
        parsed = self.parse_json_from_llm(response, IntentAnalysis)
        
        state["intent"] = parsed.get("intent", "new_request")
        extracted_prompt = parsed.get("extracted_prompt", "")
        
        if state["intent"] == "new_request":
            state["style_description"] = extracted_prompt
        elif extracted_prompt:
            state["style_description"] = extracted_prompt
            
        state["selected_index"] = parsed.get("selected_index", -1)

        self.logger.log_step(
            "IntentAnalysis",
            {
                "intent": state["intent"],
                "extracted_prompt": extracted_prompt,
                "selected_index": state["selected_index"],
            }
        )

        return state
