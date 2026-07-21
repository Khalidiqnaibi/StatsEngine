from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry

class GeometricFirstSuccessTemplate(BaseTemplate):
    template_id = "geometric_first_success"
    description = "Probability that the first success occurs on exactly trial K, given success probability P."
    required_params = ["p", "k"]

    def solve_math(self, p: float, k: int) -> float:
        return float(stats.geom.pmf(k, p))

    def simulate(self, p: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.geometric(p, trials)
        return float(np.mean(results == k))


class NegativeBinomialTemplate(BaseTemplate):
    template_id = "negative_binomial_event"
    description = "Probability that the Rth success occurs on exactly the Kth trial, given success probability P."
    required_params = ["r", "p", "k"]

    def solve_math(self, r: int, p: float, k: int) -> float:
        # scipy's nbinom models number of failures before r successes
        failures = k - r
        return float(stats.nbinom.pmf(failures, r, p))

    def simulate(self, r: int, p: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        failures = np.random.negative_binomial(r, p, trials)
        return float(np.mean(failures == (k - r)))
    
TemplateRegistry().register(GeometricFirstSuccessTemplate)
TemplateRegistry().register(NegativeBinomialTemplate)
