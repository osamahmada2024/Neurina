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
        candidate_images = state.get("candidate_images", {})
        auth_token = state.get("auth_token", "")
        user_input = state.get("user_input", "")

        if not candidate_images:
            raise ValueError("No candidate images to select from")

        if not auth_token:
            raise ValueError("auth_token is required in state")

        if len(candidate_images) == 1:
            selected_url = list(candidate_images.values())[0]
        else:
            selected_url = await self._select_best_image(
                list(candidate_images.values()),
                user_input
            )

        self.logger.log_step(
            "ReferenceSelectionComplete",
            {
                "total_candidates": len(candidate_images),
                "selected_url": selected_url[:50] + "..." if len(selected_url) > 50 else selected_url,
            },
        )

        image_routes_tool = ImageRoutesTool(auth_token)
        uploaded_reference_id = image_routes_tool.upload_reference_from_url(selected_url)

        self.logger.log_tool_call(
            "ImageUpload",
            {"image_type": "reference", "source": "web_search"},
            output_summary={"reference_image_id": uploaded_reference_id},
        )

        state["selected_reference_url"] = selected_url
        state["reference_image_id"] = uploaded_reference_id

        return state

    async def _select_best_image(self, image_urls: list, user_input: str) -> str:
        if len(image_urls) <= 1:
            return image_urls[0]

        prompt = f"""You are an expert at analyzing style images and selecting the best reference.

Style description from user: "{user_input}"

Here are {len(image_urls)} candidate reference images:
{chr(10).join([f"{i+1}. {url}" for i, url in enumerate(image_urls)])}

Based on the style description, which image number (1-{len(image_urls)}) would be the BEST reference for this style?
Consider: overall style, quality, how well it matches the description.

Return ONLY the image number as a single digit, nothing else."""

        try:
            response = self.query_llm(prompt).strip()

            try:
                image_idx = int(response) - 1
                if 0 <= image_idx < len(image_urls):
                    return image_urls[image_idx]
            except (ValueError, IndexError):
                pass

            return image_urls[0]

        except Exception as e:
            self.logger.log_tool_call(
                "LLMImageSelection",
                {"num_images": len(image_urls)},
                error=str(e),
            )
            return image_urls[0]
