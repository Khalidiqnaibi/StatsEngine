from core.base_template import BaseTemplate
import numpy as np
from ...registry import TemplateRegistry


class SummaryStatsTemplate(BaseTemplate):
    """
    Not a probability template - returns a dict of descriptive stats instead
    of a single float. Router accepts any JSON-serializable return value.
    """
    template_id = "summary_statistics"
    description = "Mean, variance, sample std dev, median, min, max, and range for a dataset."
    required_params = ["data"]

    def solve_math(self, data) -> dict:
        arr = np.array(data, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "variance": float(np.var(arr, ddof=1)) if arr.size > 1 else 0.0,
            "std_dev": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.max(arr) - np.min(arr)),
            "n": int(arr.size),
        }

    def simulate(self, data, trials: int = 1) -> dict:
        # Deterministic stat - no Monte Carlo track needed; alias to solve_math
        # so the dual-track fallback still works if solve_math ever raises.
        return self.solve_math(data)


TemplateRegistry().register(SummaryStatsTemplate)
