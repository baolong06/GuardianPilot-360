from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

@dataclass
class PipelineContext:
    """
    Strongly-typed PipelineContext object for passing data across pipeline stages.
    Provides dict-like access for backward compatibility with dictionary-based stages.
    """
    frame: Optional[np.ndarray] = None
    frame_idx: int = 0
    detections: Optional[np.ndarray] = None
    track_ids: List[int] = field(default_factory=list)
    class_names: List[str] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    distances: Dict[int, float] = field(default_factory=dict)
    ego_dx: float = 0.0
    ego_dy: float = 0.0
    predicted_trajectories: Dict[int, List[List[float]]] = field(default_factory=dict)
    pedestrian_trajectories: Dict[int, List[List[float]]] = field(default_factory=dict)
    turn_signals: Dict[str, Any] = field(default_factory=dict)
    behaviors: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    pedestrian_behaviors: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    traffic_lights: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    warning_level: str = "NONE"
    nearest_object: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, item: str) -> Any:
        if hasattr(self, item):
            val = getattr(self, item)
            return val
        return self.extra.get(item)

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra[key] = value

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item) or item in self.extra

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            val = getattr(self, key)
            return val if val is not None else default
        return self.extra.get(key, default)
