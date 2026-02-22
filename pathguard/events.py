from dataclasses import dataclass
from typing import Dict, List, Optional


STATE_CLEAR = "CLEAR"
STATE_OBSTACLE = "OBSTACLE AHEAD"
STATE_TRIP = "TRIP RISK"


@dataclass
class EventRecord:
    timestamp_s: float
    state: str
    occupancy_score: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp_s": round(self.timestamp_s, 2),
            "state": self.state,
            "occupancy_score": round(self.occupancy_score, 4),
        }


class EventStateMachine:
    def __init__(self, persistence_frames: int = 4, debounce_sec: float = 2.0, max_events: int = 200) -> None:
        self.persistence_frames = persistence_frames
        self.debounce_sec = debounce_sec
        self.max_events = max_events

        self.current_state = STATE_CLEAR
        self._hazard_counter = 0
        self._last_event_ts: Dict[str, float] = {}
        self.events: List[EventRecord] = []

    def _append_event(
        self,
        timestamp_s: float,
        state: str,
        occupancy_score: float,
    ) -> None:
        last_ts = self._last_event_ts.get(state, -1e9)
        if timestamp_s - last_ts < self.debounce_sec:
            return
        self._last_event_ts[state] = timestamp_s

        self.events.append(
            EventRecord(
                timestamp_s=timestamp_s,
                state=state,
                occupancy_score=occupancy_score,
            )
        )
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events :]

    def update(
        self,
        timestamp_s: float,
        occupancy_score: float,
        occ_thresh: float,
        trip_risk_flag: bool,
        low_quality: bool = False,
    ) -> str:
        hazard_now = (not low_quality) and (occupancy_score > occ_thresh)
        if hazard_now:
            self._hazard_counter += 1
        else:
            self._hazard_counter = 0

        target_state = STATE_CLEAR
        if self._hazard_counter >= self.persistence_frames:
            target_state = STATE_TRIP if trip_risk_flag else STATE_OBSTACLE

        if target_state != self.current_state:
            self.current_state = target_state
            self._append_event(timestamp_s, target_state, occupancy_score)

        return self.current_state

    def events_as_rows(self) -> List[Dict[str, object]]:
        return [event.as_dict() for event in self.events]
