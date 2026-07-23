from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry


class OneSampleTTestTemplate(BaseTemplate):
    template_id = "one_sample_t_test"
    description = (
        "One-sample t-test: p-value that a dataset's mean differs from a hypothesized population mean."
    )
    required_params = ["data", "pop_mean"]

    def solve_math(self, data, pop_mean: float) -> dict:
        arr = np.array(data, dtype=float)
        t_stat, p_value = stats.ttest_1samp(arr, pop_mean)
        return {"t_statistic": float(t_stat), "p_value": float(p_value), "df": int(arr.size - 1)}

    def simulate(self, data, pop_mean: float, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        arr = np.array(data, dtype=float)
        n = arr.size
        sample_std = np.std(arr, ddof=1)
        t_obs = (np.mean(arr) - pop_mean) / (sample_std / np.sqrt(n))
        null_draws = np.random.standard_t(df=n - 1, size=trials)
        p_value = float(np.mean(np.abs(null_draws) >= abs(t_obs)))
        return {"t_statistic": float(t_obs), "p_value": p_value, "df": int(n - 1)}


TemplateRegistry().register(OneSampleTTestTemplate)
