from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...config import settings
from ...services.agent_image_service import AgentImageService, agent_image_service
from bson import ObjectId

class ReferenceSelectorAgent(BaseAgent):
    def __init__(self, image_service: AgentImageService | None = None):
        super().__init__(
            model_name=settings.REASONING_MODEL,
            agent_name="ReferenceSelectorAgent",
        )
        self.image_service = image_service or agent_image_service

    async def think_and_act(self, state: AgentState) -> AgentState:
        selected_url = state.get("selected_reference_url")
        user_id = state.get("user_id", "")

        if not selected_url:
            raise ValueError("No selected_reference_url in state")

        if not ObjectId.is_valid(str(user_id)):
            raise ValueError("valid user_id is required in state")

        # Upload the selected reference
        try:
            reference_image_id = await self.image_service.upload_reference_from_url(
                user_id=ObjectId(str(user_id)),
                image_url=selected_url,
            )
            
            state["reference_image_id"] = reference_image_id
            
            self.logger.log_tool_call(
                "AgentImageService.upload_reference_from_url",
                {"url": selected_url},
                output_summary={"image_id": reference_image_id}
            )
        except Exception as e:
            self.logger.log_tool_call(
                "AgentImageService.upload_reference_from_url",
                {"url": selected_url},
                error=str(e)
            )
            raise ValueError(f"Failed to upload reference image: {e}")

        return state
