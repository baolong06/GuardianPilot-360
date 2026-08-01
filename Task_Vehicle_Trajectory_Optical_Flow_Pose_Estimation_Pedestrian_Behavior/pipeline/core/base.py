from abc import ABC, abstractmethod
from typing import Any, Dict

class Stage(ABC):
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        pass

class Pipeline(ABC):
    def __init__(self, stages: list):
        self.stages = stages

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Chạy pipeline với data là dict. Giữ nguyên data."""
        for stage in self.stages:
            data = stage.process(data)
        return data