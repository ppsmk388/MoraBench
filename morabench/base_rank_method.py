from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):


    @abstractmethod
    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def rank(self, model_set, split=None,*args: Any, **kwargs: Any):
        """Abstract method for ranking model set.

        Parameters
        ----------
        """
        pass

    @abstractmethod
    def save(self, destination: str) -> None:
        """Abstract method for save the model output.
        ----------
        """
        pass

    @abstractmethod
    def load(self, source: str) -> None:
        """Abstract method for result save.
        ----------
        """
        pass
