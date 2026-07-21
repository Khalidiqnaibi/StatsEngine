from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry

class ValueAtRiskTemplate(BaseTemplate):
    template_id = "value_at_risk"
    description = "Value at Risk (VaR): the loss threshold not exceeded with confidence level C, given portfolio mean return and std dev of returns."
    required_params = ["mean_return", "std_return", "confidence", "portfolio_value"]

    def solve_math(self, mean_return: float, std_return: float, confidence: float, portfolio_value: float) -> float:
        z = stats.norm.ppf(1 - confidence)
        var_return = mean_return + z * std_return
        return float(-var_return * portfolio_value)

    def simulate(self, mean_return: float, std_return: float, confidence: float, portfolio_value: float, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        returns = np.random.normal(mean_return, std_return, trials)
        pnl = returns * portfolio_value
        var = -np.percentile(pnl, (1 - confidence) * 100)
        return float(var)

TemplateRegistry().register(ValueAtRiskTemplate)