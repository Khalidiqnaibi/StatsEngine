from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry

class PoissonExactTemplate(BaseTemplate):
    template_id = "poisson_event"
    description = "Probability of exactly K events occurring given an average rate lambda over a fixed interval."
    required_params = ["lam", "k"]

    def solve_math(self, lam: float, k: int) -> float:
        return float(stats.poisson.pmf(k, lam))

    def simulate(self, lam: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.poisson(lam, trials)
        return float(np.mean(results == k))


class PoissonAtLeastTemplate(BaseTemplate):
    template_id = "poisson_at_least"
    description = "Probability of at least K events occurring given an average rate lambda over a fixed interval."
    required_params = ["lam", "k"]

    def solve_math(self, lam: float, k: int) -> float:
        return float(1 - stats.poisson.cdf(k - 1, lam))

    def simulate(self, lam: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.poisson(lam, trials)
        return float(np.mean(results >= k))

TemplateRegistry().register(PoissonExactTemplate)
TemplateRegistry().register(PoissonAtLeastTemplate)