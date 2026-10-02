"""
ASTRA Action Validator

Provides a procedure-aware validation layer to protect downstream components 
(Temporal GRU and State Machine) from noisy or impossible raw predictions 
from the ST-GCN action recognition model.
"""
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ActionValidator:
    """
    Validates ST-GCN actions based on procedure awareness and confidence gating.
    """
    def __init__(self, experiment_config=None, pose_confidence_threshold: float = 0.3):
        self.config = experiment_config
        self.pose_confidence_threshold = pose_confidence_threshold
        
    def validate(
        self, 
        raw_action: str, 
        raw_confidence: float, 
        pose_confidence: float, 
        current_step_id: Optional[str]
    ) -> Tuple[str, bool, str]:
        """
        Validates the raw action.
        
        Args:
            raw_action: The raw action from ST-GCN.
            raw_confidence: The confidence of the raw action.
            pose_confidence: The mean pose confidence.
            current_step_id: The ID of the currently expected step in the procedure.
            
        Returns:
            validated_action: The action allowed downstream (e.g. raw_action, 'uncertain', or 'invalid').
            is_valid: Boolean indicating if it's a valid prediction.
            reason: String explanation of the validation result.
        """
        # 1. Pose Confidence Gating
        if pose_confidence < self.pose_confidence_threshold:
            return "uncertain", False, f"Low pose confidence: {pose_confidence:.3f} < {self.pose_confidence_threshold}"

        # 2. Procedure-Aware State Gating
        if self.config and current_step_id:
            current_step = self.config.get_step(current_step_id)
            if current_step:
                # Get the required actions for the current step
                allowed_actions = set(current_step.required_actions)
                
                # Also allow 'idle' to be a valid transition state
                allowed_actions.add("idle")

                # If the current step can transition to next steps, we should also allow their actions
                # so that the state machine can detect the transition.
                for next_step_id in current_step.allowed_next_steps:
                    if next_step_id != "COMPLETE":
                        ns = self.config.get_step(next_step_id)
                        if ns:
                            allowed_actions.update(ns.required_actions)
                            
                if raw_action not in allowed_actions:
                    return "invalid", False, f"Action '{raw_action}' is OUT_OF_SEQUENCE for step '{current_step_id}'"

        return raw_action, True, "Valid action sequence"
