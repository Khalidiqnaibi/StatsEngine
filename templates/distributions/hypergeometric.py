from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry


class HypergeometricTemplate(BaseTemplate):
    template_id = "hypergeometric_event"
    description = "Probability of drawing exactly K successes in N draws without replacement from a finite population with K_total successes."
    required_params = ["population_size", "success_states", "draws", "k"]

    def solve_math(self, population_size: int, success_states: int, draws: int, k: int) -> float:
        return float(stats.hypergeom.pmf(k, population_size, success_states, draws))

    def simulate(self, population_size: int, success_states: int, draws: int, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.hypergeometric(success_states, population_size - success_states, draws, trials)
        return float(np.mean(results == k))

TemplateRegistry().register(HypergeometricTemplate)