from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseTemplate(ABC):
    """
    Abstract Base Class for all StatsEngine templates.
    Enforces the dual-track execution requirement (math + simulation).
    """
    
    @property
    @abstractmethod
    def template_id(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def required_params(self) -> List[str]:
        pass

    @abstractmethod
    def solve_math(self, **kwargs) -> float:
        """Exact analytical math solution using SciPy/SymPy."""
        pass

    @abstractmethod
    def simulate(self, **kwargs) -> float:
        """Vectorized Monte Carlo fallback using NumPy."""
        pass

    def validate_params(self, params: Dict[str, Any]) -> None:
        """Validates that all required parameters are present in the payload."""
        missing = [p for p in self.required_params if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters for {self.template_id}: {missing}")