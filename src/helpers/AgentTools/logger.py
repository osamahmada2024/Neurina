import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger("neurina.agents")


class AgentLogger:
    """Structured logging for agent workflow execution."""

    @staticmethod
    def log_step(
        step_name: str,
        data: Dict[str, Any],
        level: str = "INFO",
        error: Optional[str] = None,
    ) -> None:
        

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": step_name,
            "data": data,
        }

        if error:
            log_entry["error"] = error

        log_message = json.dumps(log_entry)

        if level.upper() == "DEBUG":
            logger.debug(log_message)
        elif level.upper() == "WARNING":
            logger.warning(log_message)
        elif level.upper() == "ERROR":
            logger.error(log_message)
        else:
            logger.info(log_message)

    @staticmethod
    def log_agent_start(agent_name: str, user_id: str, state_keys: list) -> None:
        AgentLogger.log_step(
            f"{agent_name}_START",
            {
                "user_id": user_id,
                "state_keys": state_keys,
            }
        )

    @staticmethod
    def log_agent_end(
        agent_name: str,
        user_id: str,
        updated_keys: list,
        duration_seconds: float,
    ) -> None:
        

        AgentLogger.log_step(
            f"{agent_name}_END",
            {
                "user_id": user_id,
                "updated_keys": updated_keys,
                "duration_seconds": round(duration_seconds, 2),
            }
        )

    @staticmethod
    def log_tool_call(
        tool_name: str,
        input_params: Dict[str, Any],
        output_summary: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        

        data = {
            "tool": tool_name,
            "input_params": input_params,
        }

        if output_summary:
            data["output"] = output_summary

        AgentLogger.log_step(
            f"TOOL_CALL_{tool_name}",
            data,
            level="ERROR" if error else "INFO",
            error=error,
        )

    @staticmethod
    def log_workflow_start(user_id: str, user_input: str) -> None:

        AgentLogger.log_step(
            "WORKFLOW_START",
            {
                "user_id": user_id,
                "user_input": user_input[:100],
            },
        )

    @staticmethod
    def log_workflow_end(
        user_id: str,
        success: bool,
        final_state: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        
        
        data = {
            "user_id": user_id,
            "success": success,
        }

        if final_state:
            data["final_state"] = {
                k: v
                for k, v in final_state.items()
                if k != "candidate_images"
            }

        AgentLogger.log_step(
            "WORKFLOW_END",
            data,
            level="ERROR" if error and not success else "INFO",
            error=error,
        )