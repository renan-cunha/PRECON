from abc import ABC, abstractmethod
import numpy as np


class ExpertAgent(ABC):

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        pass
