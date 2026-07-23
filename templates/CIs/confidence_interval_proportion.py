from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry


class ProportionConfidenceIntervalTemplate(BaseTemplate):
    template_id = "confidence_interval_proportion"
    description = (
        "Confidence interval for a population proportion given number of successes, "
        "sample size, and confidence level (Wald normal approximation)."
    )
    required_params = ["successes", "n", "confidence"]

    def solve_math(self, successes: int, n: int, confidence: float) -> dict:
        p_hat = successes / n
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n)
        return {"lower": float(max(0.0, p_hat - margin)), "upper": float(min(1.0, p_hat + margin)), "margin_of_error": float(margin)}

    def simulate(self, successes: int, n: int, confidence: float, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        p_hat = successes / n
        draws = np.random.binomial(n, p_hat, trials) / n
        alpha = 1 - confidence
        lower = float(np.percentile(draws, 100 * alpha / 2))
        upper = float(np.percentile(draws, 100 * (1 - alpha / 2)))
        return {"lower": lower, "upper": upper, "margin_of_error": float((upper - lower) / 2)}


TemplateRegistry().register(ProportionConfidenceIntervalTemplate)
