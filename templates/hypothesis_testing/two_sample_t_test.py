from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ...registry import TemplateRegistry


class TwoSampleTTestTemplate(BaseTemplate):
    template_id = "two_sample_t_test"
    description = (
        "Independent two-sample t-test (Welch's, unequal variance by default): p-value that "
        "two datasets have different means."
    )
    required_params = ["data_a", "data_b"]

    def solve_math(self, data_a, data_b, equal_var: bool = False) -> dict:
        a, b = np.array(data_a, dtype=float), np.array(data_b, dtype=float)
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)
        return {"t_statistic": float(t_stat), "p_value": float(p_value)}

    def simulate(self, data_a, data_b, equal_var: bool = False, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        a, b = np.array(data_a, dtype=float), np.array(data_b, dtype=float)
        observed_diff = np.mean(a) - np.mean(b)
        pooled = np.concatenate([a, b])
        n_a = a.size
        diffs = np.empty(trials)
        for i in range(trials):
            np.random.shuffle(pooled)
            diffs[i] = pooled[:n_a].mean() - pooled[n_a:].mean()
        p_value = float(np.mean(np.abs(diffs) >= abs(observed_diff)))
        # Report a t-like statistic for interface consistency even though the
        # simulation track uses a permutation test rather than the t-distribution.
        return {"t_statistic": float(observed_diff), "p_value": p_value}


TemplateRegistry().register(TwoSampleTTestTemplate)
