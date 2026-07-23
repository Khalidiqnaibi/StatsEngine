from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry


class OneSampleZTestTemplate(BaseTemplate):
    template_id = "one_sample_z_test"
    description = (
        "One-sample z-test: p-value for observed sample mean vs a hypothesized population mean, "
        "given known population std dev and sample size."
    )
    required_params = ["sample_mean", "pop_mean", "pop_std", "n"]

    def solve_math(self, sample_mean: float, pop_mean: float, pop_std: float, n: int) -> dict:
        se = pop_std / np.sqrt(n)
        z = (sample_mean - pop_mean) / se
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        return {"z_statistic": float(z), "p_value": p_value}

    def simulate(self, sample_mean: float, pop_mean: float, pop_std: float, n: int, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        se = pop_std / np.sqrt(n)
        z_obs = (sample_mean - pop_mean) / se
        null_draws = np.random.normal(0, 1, trials)
        p_value = float(np.mean(np.abs(null_draws) >= abs(z_obs)))
        return {"z_statistic": float(z_obs), "p_value": p_value}


TemplateRegistry().register(OneSampleZTestTemplate)
