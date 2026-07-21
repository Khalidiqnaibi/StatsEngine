from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np

from ..registry import TemplateRegistry


class NormalCDFTemplate(BaseTemplate):
    template_id = "normal_less_than"
    description = "Probability that a normally distributed variable is less than or equal to X, given mean and std dev."
    required_params = ["mean", "std", "x"]

    def solve_math(self, mean: float, std: float, x: float) -> float:
        return float(stats.norm.cdf(x, loc=mean, scale=std))

    def simulate(self, mean: float, std: float, x: float, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.normal(mean, std, trials)
        return float(np.mean(results <= x))


class NormalBetweenTemplate(BaseTemplate):
    template_id = "normal_between"
    description = "Probability that a normally distributed variable falls between two bounds, given mean and std dev."
    required_params = ["mean", "std", "lower", "upper"]

    def solve_math(self, mean: float, std: float, lower: float, upper: float) -> float:
        return float(stats.norm.cdf(upper, loc=mean, scale=std) - stats.norm.cdf(lower, loc=mean, scale=std))

    def simulate(self, mean: float, std: float, lower: float, upper: float, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.normal(mean, std, trials)
        return float(np.mean((results >= lower) & (results <= upper)))

TemplateRegistry().register(NormalCDFTemplate)
TemplateRegistry().register(NormalBetweenTemplate)