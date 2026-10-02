"""
ASTRA — Voice Alert Module

Offline TTS using pyttsx3 (no internet required).
Fixed templates with event cooldowns.

Voice alerts:
- Step completion
- Next step instruction
- Out-of-order warning
- Uncertain action warning
- Experiment completion
"""
from __future__ import annotations

import logging
import time
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# Fixed voice templates
VOICE_TEMPLATES = {
    "step_complete": "Step {step_name} completed.",
    "next_step": "Next step: {step_name}.",
    "out_of_order": "Warning. Action out of sequence. Expected {expected}, detected {detected}.",
    "skipped_step": "Warning. Step {step_name} appears to be skipped.",
    "uncertain": "Step uncertain. Please repeat the action.",
    "experiment_complete": "Experiment complete. All steps finished successfully.",
    "experiment_failed": "Experiment incomplete. Please review the procedure.",
    "timeout_warning": "Warning. Step {step_name} is taking too long.",
}


class VoiceAlertModule:
    """
    Offline voice alert system with cooldown management.

    Uses pyttsx3 for offline TTS. Falls back to console logging
    if pyttsx3 is not available.
    """

    def __init__(
        self,
        enabled: bool = True,
        cooldown_seconds: float = 3.0,
        rate: int = 150,
        volume: float = 0.9,
    ):
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self._last_alert_time: Dict[str, float] = {}
        self._engine = None
        self._tts_available = False
        self._lock = threading.Lock()

        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", rate)
                self._engine.setProperty("volume", volume)
                self._tts_available = True
                logger.info("Voice alerts: pyttsx3 initialized")
            except Exception as e:
                logger.warning(f"Voice alerts: pyttsx3 not available ({e}). Using console fallback.")
                self._tts_available = False

    def speak(self, alert_type: str, **kwargs) -> bool:
        """
        Speak an alert if cooldown has expired.

        Args:
            alert_type: Key from VOICE_TEMPLATES.
            **kwargs: Template format arguments.

        Returns:
            True if alert was spoken, False if on cooldown or disabled.
        """
        if not self.enabled:
            return False

        # Check cooldown
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < self.cooldown_seconds:
            return False

        # Format message
        template = VOICE_TEMPLATES.get(alert_type, alert_type)
        try:
            message = template.format(**kwargs)
        except KeyError:
            message = template

        # Update cooldown
        self._last_alert_time[alert_type] = now

        # Log the alert
        logger.info(f"🔊 VOICE: {message}")

        # Speak asynchronously
        if self._tts_available:
            threading.Thread(target=self._speak_async, args=(message,), daemon=True).start()

        return True

    def _speak_async(self, message: str):
        """Speak message in a background thread."""
        with self._lock:
            try:
                self._engine.say(message)
                self._engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTS error: {e}")

    def on_step_complete(self, step_name: str):
        """Alert when a step is completed."""
        self.speak("step_complete", step_name=step_name)

    def on_next_step(self, step_name: str):
        """Alert the next required step."""
        self.speak("next_step", step_name=step_name)

    def on_out_of_order(self, expected: str, detected: str):
        """Alert out-of-order action."""
        self.speak("out_of_order", expected=expected, detected=detected)

    def on_skipped_step(self, step_name: str):
        """Alert a skipped step."""
        self.speak("skipped_step", step_name=step_name)

    def on_uncertain(self):
        """Alert uncertain action."""
        self.speak("uncertain")

    def on_experiment_complete(self):
        """Alert experiment completion."""
        self.speak("experiment_complete")

    def on_timeout_warning(self, step_name: str):
        """Alert step timeout warning."""
        self.speak("timeout_warning", step_name=step_name)

    def close(self):
        """Clean up TTS engine."""
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
