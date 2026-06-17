from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...helpers.AgentTools.image_routes_tool import ImageRoutesTool
from ...config import settings

class ReferenceSelectorAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model_name=settings.REASONING_MODEL,
            agent_name="ReferenceSelectorAgent",
        )

    async def think_and_act(self, state: AgentState) -> AgentState:
        auth_token = state.get("auth_token", "")
        selected_url = state.get("selected_reference_url")

        if not selected_url:
            raise ValueError("No selected_reference_url in state")

        if not auth_token:
            raise ValueError("auth_token is required in state")

        # Upload the selected reference
        try:
            image_tool = ImageRoutesTool(token=auth_token)
            reference_image_id = image_tool.upload_reference_from_url(selected_url)
            
            state["reference_image_id"] = reference_image_id
            
            self.logger.log_tool_call(
                "ImageRoutesTool.upload",
                {"url": selected_url},
                output_summary={"image_id": reference_image_id}
            )
        except Exception as e:
            self.logger.log_tool_call(
                "ImageRoutesTool.upload",
                {"url": selected_url},
                error=str(e)
            )
            raise ValueError(f"Failed to upload reference image: {e}")

        return state