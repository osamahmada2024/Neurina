from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from ...helpers.AgentTools.face_score import batch_score_images
from ...config import settings


class QualityControlAgent(BaseAgent):


    def __init__(self):
        super().__init__(
            model_name=settings.QUERY_MODEL,
            agent_name="QualityControlAgent",
        )
        self.quality_threshold = float(getattr(settings, "QUALITY_GATE_THRESHOLD", 0.7))

    async def think_and_act(self, state: AgentState) -> AgentState:
      
        candidate_images = state.get("candidate_images", {})

        if not candidate_images:
            self.logger.log_step(
                "QualityControlAgent",
                {"status": "no_candidates"},
                level="WARNING",
            )
            state["candidate_images"] = {}
            state["quality_score"] = "0.0"
            return state

        try:
            # Extract URLs
            image_urls = list(candidate_images.values())

            # Score all images
            scored_images = batch_score_images(image_urls)

            # Filter by quality criteria
            passing_images = []
            best_score = 0.0

            for url, score_dict in scored_images:
                if score_dict.get("quality_score", 0.0) >= self.quality_threshold:
                    passing_images.append((url, score_dict))
                    best_score = max(best_score, score_dict.get("quality_score", 0.0))

            # Log results
            self.logger.log_tool_call(
                "ImageQualityScoring",
                {"total_images": len(candidate_images)},
                output_summary={
                    "passed_quality": len(passing_images),
                    "best_score": round(best_score, 3),
                },
            )

            # Update state
            if passing_images:
                # Keep only passing images in candidate_images preserving UUID keys
                passing_urls = {url for url, _ in passing_images}
                filtered_candidates = {
                    uid: url for uid, url in candidate_images.items() if url in passing_urls
                }
                state["candidate_images"] = filtered_candidates
                state["quality_score"] = str(round(best_score, 3))
            else:
                # No images passed: keep the best one anyway
                if scored_images:
                    best_url, best_dict = scored_images[0]
                    best_uid = next((uid for uid, url in candidate_images.items() if url == best_url), "unknown")
                    state["candidate_images"] = {best_uid: best_url}
                    state["quality_score"] = str(round(best_dict.get("quality_score", 0.0), 3))
                    self.logger.log_step(
                        "QualityControlAgent",
                        {"status": "no_passing_images_using_best"},
                        level="WARNING",
                    )
                else:
                    state["candidate_images"] = {}
                    state["quality_score"] = "0.0"

            return state

        except Exception as e:
            self.logger.log_tool_call(
                "ImageQualityScoring",
                {"total_images": len(candidate_images)},
                error=str(e),
            )
            # On error, keep all candidates
            state["quality_score"] = "0.0"
            raise
