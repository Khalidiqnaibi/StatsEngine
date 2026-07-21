from core.base_template import BaseTemplate
import numpy as np
from ..registry import TemplateRegistry

class RouletteSingleNumberTemplate(BaseTemplate):
    template_id = "roulette_single_number"
    description = "Probability of a single chosen number hitting on a roulette wheel with a given number of pockets (e.g. 37 for European, 38 for American)."
    required_params = ["pockets"]

    def solve_math(self, pockets: int) -> float:
        return 1.0 / pockets

    def simulate(self, pockets: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.randint(0, pockets, trials)
        return float(np.mean(results == 0))


class RouletteConsecutiveColorTemplate(BaseTemplate):
    template_id = "roulette_consecutive_color"
    description = "Probability of the same color (red/black) hitting K times in a row on roulette, given color probability P (typically 18/37 or 18/38)."
    required_params = ["p", "k"]

    def solve_math(self, p: float, k: int) -> float:
        return float(p ** k)

    def simulate(self, p: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        spins = np.random.random((trials, k)) < p
        return float(np.mean(np.all(spins, axis=1)))

TemplateRegistry().register(RouletteSingleNumberTemplate)
TemplateRegistry().register(RouletteConsecutiveColorTemplate)