from core.base_template import BaseTemplate
from ..registry import TemplateRegistry


class ZScoreTemplate(BaseTemplate):
    template_id = "z_score"
    description = "Z-score of a raw value X given a population mean and standard deviation."
    required_params = ["x", "mean", "std"]

    def solve_math(self, x: float, mean: float, std: float) -> float:
        return float((x - mean) / std)

    def simulate(self, x: float, mean: float, std: float, trials: int = 1) -> float:
        return self.solve_math(x, mean, std)


TemplateRegistry().register(ZScoreTemplate)
