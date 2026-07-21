from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry


class ExponentialWaitTemplate(BaseTemplate):
    template_id = "exponential_wait_time"
    description = "Probability that the wait time for an event is less than or equal to T, given average rate lambda."
    required_params = ["lam", "t"]

    def solve_math(self, lam: float, t: float) -> float:
        return float(stats.expon.cdf(t, scale=1 / lam))

    def simulate(self, lam: float, t: float, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.exponential(1 / lam, trials)
        return float(np.mean(results <= t))


class UniformRangeTemplate(BaseTemplate):
    template_id = "uniform_range_probability"
    description = "Probability that a uniformly distributed variable over [low, high] falls between two bounds."
    required_params = ["low", "high", "lower", "upper"]

    def solve_math(self, low: float, high: float, lower: float, upper: float) -> float:
        return float(stats.uniform.cdf(upper, loc=low, scale=high - low) - stats.uniform.cdf(lower, loc=low, scale=high - low))

    def simulate(self, low: float, high: float, lower: float, upper: float, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.uniform(low, high, trials)
        return float(np.mean((results >= lower) & (results <= upper)))
    

TemplateRegistry().register(ExponentialWaitTemplate)
TemplateRegistry().register(UniformRangeTemplate)