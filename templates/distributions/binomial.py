from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np

class BinomialTrialTemplate(BaseTemplate):
    template_id = "binomial_event"
    description = "Probability of exactly K successes in N independent trials with probability P."
    required_params = ["n", "p", "k"]

    def solve_math(self, n: int, p: float, k: int) -> float:
        """Exact probability using SciPy."""
        return float(stats.binom.pmf(k, n, p))

    def simulate(self, n: int, p: float, k: int, trials: int = 100_000) -> float:
        """Vectorized Monte Carlo fallback using NumPy."""
        # Hard cap simulation trials for memory safety as per spec
        trials = min(trials, 1_000_000) 
        results = np.random.binomial(n, p, trials)
        return float(np.mean(results == k))