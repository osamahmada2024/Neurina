from .base_agent import BaseAgent
from ...schemes.agent_state import AgentState
from .query_agent import QueryAgent
from .api_execution_agent import APIExecutionAgent
from .quality_control_agent import QualityControlAgent
from .reference_selector_agent import ReferenceSelectorAgent
from ...config import settings


class SupervisorAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            model_name=settings.SUPERVISOR_MODEL,
            agent_name="SupervisorAgent",
        )
        self.query_agent = QueryAgent()
        self.api_execution_agent = APIExecutionAgent()
        self.quality_control_agent = QualityControlAgent()
        self.reference_selector_agent = ReferenceSelectorAgent()

    async def think_and_act(self, state: AgentState) -> AgentState:
        user_id = state.get("user_id", "unknown")

        self.logger.log_workflow_start(user_id, state.get("user_input", ""))

        agents = [
            ("QueryAgent", self.query_agent),
            ("APIExecutionAgent", self.api_execution_agent),
            ("QualityControlAgent", self.quality_control_agent),
            ("ReferenceSelectorAgent", self.reference_selector_agent),
        ]

        try:
            # Execute agents in sequence
            for agent_name, agent in agents:
                try:
                    state = await agent(state)
                except Exception as e:
                    # Log error but continue if possible
                    self.logger.log_step(
                        f"{agent_name}_FAILED",
                        {"error": str(e)},
                        level="ERROR",
                    )

                    # If this is a critical agent, fail the workflow
                    if agent_name in ["QueryAgent", "APIExecutionAgent"]:
                        state["errors"].append(f"{agent_name}: {str(e)}")
                        raise

                    # For later agents, try to continue
                    state["errors"].append(f"{agent_name}: {str(e)}")

            # Workflow completed successfully
            success = bool(state.get("selected_reference_url"))

            self.logger.log_workflow_end(
                user_id,
                success=success,
                final_state={
                    k: v for k, v in state.items()
                    if k not in ["auth_token", "candidate_images"]
                },
            )

            return state

        except Exception as e:
            # Workflow failed
            self.logger.log_workflow_end(
                user_id,
                success=False,
                error=str(e),
            )
            raise
