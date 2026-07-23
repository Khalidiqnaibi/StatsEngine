from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ...registry import TemplateRegistry


class ChiSquareGoodnessOfFitTemplate(BaseTemplate):
    template_id = "chi_square_goodness_of_fit"
    description = (
        "Chi-square goodness-of-fit test: p-value that observed category counts match "
        "expected counts."
    )
    required_params = ["observed", "expected"]

    def solve_math(self, observed, expected) -> dict:
        obs, exp = np.array(observed, dtype=float), np.array(expected, dtype=float)
        chi2_stat, p_value = stats.chisquare(obs, f_exp=exp)
        return {"chi2_statistic": float(chi2_stat), "p_value": float(p_value), "df": int(obs.size - 1)}

    def simulate(self, observed, expected, trials: int = 100_000) -> dict:
        trials = min(trials, 1_000_000)
        obs, exp = np.array(observed, dtype=float), np.array(expected, dtype=float)
        chi2_obs = float(np.sum((obs - exp) ** 2 / exp))
        null_draws = np.random.chisquare(df=obs.size - 1, size=trials)
        p_value = float(np.mean(null_draws >= chi2_obs))
        return {"chi2_statistic": chi2_obs, "p_value": p_value, "df": int(obs.size - 1)}


TemplateRegistry().register(ChiSquareGoodnessOfFitTemplate)
