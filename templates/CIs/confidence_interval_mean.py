from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry


class MeanConfidenceIntervalTemplate(BaseTemplate):
    template_id = "confidence_interval_mean"
    description = (
        "Confidence interval for a population mean given sample mean, sample std dev, "
        "sample size, and confidence level. Uses the t-distribution."
    )
    required_params = ["sample_mean", "sample_std", "n", "confidence"]

    def solve_math(self, sample_mean: float, sample_std: float, n: int, confidence: float) -> dict:
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * (sample_std / np.sqrt(n))
        return {"lower": float(sample_mean - margin), "upper": float(sample_mean + margin), "margin_of_error": float(margin)}

    def simulate(self, sample_mean: float, sample_std: float, n: int, confidence: float, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        # Bootstrap-style simulation: draw synthetic samples from N(sample_mean, sample_std)
        # and empirically bound the middle `confidence` fraction of resulting sample means.
        draws = np.random.normal(sample_mean, sample_std, size=(trials, n))
        sample_means = draws.mean(axis=1)
        alpha = 1 - confidence
        lower = float(np.percentile(sample_means, 100 * alpha / 2))
        upper = float(np.percentile(sample_means, 100 * (1 - alpha / 2)))
        return {"lower": lower, "upper": upper, "margin_of_error": float((upper - lower) / 2)}


TemplateRegistry().register(MeanConfidenceIntervalTemplate)
