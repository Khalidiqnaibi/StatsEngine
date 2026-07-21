from core.base_template import BaseTemplate
import numpy as np
from ..registry import TemplateRegistry

class ExpectedValueTemplate(BaseTemplate):
    template_id = "expected_value_discrete"
    description = "Expected value of a discrete random variable given a list of outcomes and their associated probabilities."
    required_params = ["outcomes", "probabilities"]

    def solve_math(self, outcomes, probabilities) -> float:
        outcomes = np.array(outcomes, dtype=float)
        probabilities = np.array(probabilities, dtype=float)
        return float(np.sum(outcomes * probabilities))

    def simulate(self, outcomes, probabilities, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        outcomes = np.array(outcomes, dtype=float)
        probabilities = np.array(probabilities, dtype=float)
        results = np.random.choice(outcomes, size=trials, p=probabilities)
        return float(np.mean(results))

TemplateRegistry().register(ExpectedValueTemplate)