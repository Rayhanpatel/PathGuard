import time


class AudioAlerter:
    def __init__(self, cooldown_sec: float = 1.5) -> None:
        self.cooldown_sec = max(0.1, float(cooldown_sec))
        self._last_alert_t = 0.0

    def _can_alert(self) -> bool:
        now = time.time()
        if now - self._last_alert_t < self.cooldown_sec:
            return False
        self._last_alert_t = now
        return True

    def _beep(self, freq: int, dur_ms: int) -> None:
        try:
            import winsound  # Windows only

            winsound.Beep(int(freq), int(dur_ms))
            return
        except Exception:
            pass

        # Fallback terminal bell for non-Windows environments.
        print("\a", end="", flush=True)

    def alert(self, state: str) -> None:
        if not self._can_alert():
            return

        if state == "TRIP RISK":
            self._beep(1300, 180)
            time.sleep(0.08)
            self._beep(1300, 220)
