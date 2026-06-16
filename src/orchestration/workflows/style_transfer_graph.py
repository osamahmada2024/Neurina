from typing import Dict, Any
from bson import ObjectId

from ...schemes.agent_state import AgentState
from ..agents import SupervisorAgent
from ...helpers.AgentTools.logger import AgentLogger


class StyleTransferGraph:

    def __init__(self):
        self.supervisor_agent = SupervisorAgent()
        self.logger = AgentLogger()

    async def execute(
        self,
        user_id: ObjectId,
        user_input: str,
        auth_token: str,
    ) -> Dict[str, Any]:
        # Initialize workflow state
        state: AgentState = {
            "user_input": user_input,
            "source_image_id": "",
            "auth_token": auth_token,
            "user_id": str(user_id),

            "generated_search_query": None,
            "candidate_images": None,
            "selected_reference_url": None,
            "uploaded_reference_url": None,

            "translation_task_id": None,
            "final_output_url": None,

            "quality_score": None,
            "errors": [],
        }

        try:
            # Execute supervisor agent
            final_state = await self.supervisor_agent(state)

            # Check success
            if final_state.get("selected_reference_url"):
                return {
                    "success": True,
                    "reference_image_id": final_state.get("reference_image_id"),
                    "selected_reference_url": final_state.get("selected_reference_url"),
                    "quality_score": final_state.get("quality_score"),
                    "message": "Reference image selected! Please upload your source image.",
                    "errors": final_state.get("errors", []),
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to find suitable reference image",
                    "errors": final_state.get("errors", ["No reference image selected"]),
                }

        except Exception as e:
            self.logger.log_step(
                "StyleTransferGraph_ERROR",
                {"error": str(e), "user_id": str(user_id)},
                level="ERROR",
            )
            return {
                "success": False,
                "message": f"Workflow failed: {str(e)}",
                "errors": [str(e)],
            }
