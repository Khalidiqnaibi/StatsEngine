from core.base_template import BaseTemplate
import scipy.stats as stats
from ...registry import TemplateRegistry


class SimpleLinearRegressionTemplate(BaseTemplate):
    template_id = "simple_linear_regression"
    description = (
        "Ordinary least squares regression of Y on X: slope, intercept, R-squared, "
        "and predicted value at a given X (optional)."
    )
    required_params = ["data_x", "data_y"]

    def solve_math(self, data_x, data_y, predict_x: float = None) -> dict:
        import numpy as np
        x, y = np.array(data_x, dtype=float), np.array(data_y, dtype=float)
        result = stats.linregress(x, y)
        output = {
            "slope": float(result.slope),
            "intercept": float(result.intercept),
            "r_squared": float(result.rvalue ** 2),
            "p_value": float(result.pvalue),
            "std_err": float(result.stderr),
        }
        if predict_x is not None:
            output["prediction"] = float(result.slope * predict_x + result.intercept)
        return output

    def simulate(self, data_x, data_y, predict_x: float = None, trials: int = 1) -> dict:
        # OLS is a closed-form solution; simulation track exists only to satisfy the
        # dual-track interface and acts as a deterministic fallback.
        return self.solve_math(data_x, data_y, predict_x)


TemplateRegistry().register(SimpleLinearRegressionTemplate)
