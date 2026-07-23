from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ...registry import TemplateRegistry


class PearsonCorrelationTemplate(BaseTemplate):
    template_id = "pearson_correlation"
    description = "Pearson correlation coefficient and p-value between two equal-length numeric datasets."
    required_params = ["data_x", "data_y"]

    def solve_math(self, data_x, data_y) -> dict:
        x, y = np.array(data_x, dtype=float), np.array(data_y, dtype=float)
        r, p_value = stats.pearsonr(x, y)
        return {"r": float(r), "p_value": float(p_value)}

    def simulate(self, data_x, data_y, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        x, y = np.array(data_x, dtype=float), np.array(data_y, dtype=float)
        r_obs = np.corrcoef(x, y)[0, 1]
        n = x.size
        perm_r = np.empty(trials)
        y_shuffled = y.copy()
        for i in range(trials):
            np.random.shuffle(y_shuffled)
            perm_r[i] = np.corrcoef(x, y_shuffled)[0, 1]
        p_value = float(np.mean(np.abs(perm_r) >= abs(r_obs)))
        return {"r": float(r_obs), "p_value": p_value}


TemplateRegistry().register(PearsonCorrelationTemplate)
