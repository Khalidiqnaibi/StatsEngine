from core.base_template import BaseTemplate
import numpy as np
from ..registry import TemplateRegistry

class GamblersRuinTemplate(BaseTemplate):
    template_id = "gamblers_ruin_probability"
    description = "Probability of reaching a target wealth before going broke, starting with 'start' capital, target 'goal', and per-bet win probability P."
    required_params = ["start", "goal", "p"]

    def solve_math(self, start: int, goal: int, p: float) -> float:
        q = 1 - p
        if p == 0.5:
            return float(start / goal)
        ratio = q / p
        numerator = 1 - ratio ** start
        denominator = 1 - ratio ** goal
        return float(numerator / denominator)

    def simulate(self, start: int, goal: int, p: float, trials: int = 10_000) -> float:
        trials = min(trials, 100_000)
        wins = 0
        max_steps = 100_000
        for _ in range(trials):
            wealth = start
            steps = 0
            while 0 < wealth < goal and steps < max_steps:
                if np.random.random() < p:
                    wealth += 1
                else:
                    wealth -= 1
                steps += 1
            if wealth >= goal:
                wins += 1
        return float(wins / trials)

TemplateRegistry().register(GamblersRuinTemplate)