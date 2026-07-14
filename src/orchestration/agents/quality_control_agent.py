import asyncio

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
            scored_images = await asyncio.to_thread(batch_score_images, image_urls)

            # Filter by quality criteria. The translation model expects a single clear
            # face reference; a sharp image with no face or multiple faces can still
            # produce unstable style codes, so we require both the aggregate gate and
            # the explicit one-face check from the scorer.
            passing_images = []
            best_score = 0.0

            for url, score_dict in scored_images:
                quality_score = float(score_dict.get("quality_score", 0.0))
                face_count = int(score_dict.get("face_count", 0))
                passes_gate = bool(score_dict.get("passes_gate", False))
                best_score = max(best_score, quality_score)

                if (
                    passes_gate
                    and face_count == 1
                    and quality_score >= self.quality_threshold
                ):
                    passing_images.append((url, score_dict))

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
                state["candidate_images"] = {}
                state["quality_score"] = str(round(best_score, 3))
                self.logger.log_step(
                    "QualityControlAgent",
                    {"status": "no_images_passed_face_quality_gate"},
                    level="WARNING",
                )

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
