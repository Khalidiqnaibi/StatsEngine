from core.base_template import BaseTemplate
import numpy as np
from itertools import product
from ..registry import TemplateRegistry

class DiceSumTemplate(BaseTemplate):
    template_id = "dice_sum_probability"
    description = "Probability that the sum of rolling N dice with S sides each equals target sum."
    required_params = ["n_dice", "sides", "target_sum"]

    def solve_math(self, n_dice: int, sides: int, target_sum: int) -> float:
        outcomes = list(product(range(1, sides + 1), repeat=n_dice))
        total = len(outcomes)
        favorable = sum(1 for o in outcomes if sum(o) == target_sum)
        return favorable / total

    def simulate(self, n_dice: int, sides: int, target_sum: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        rolls = np.random.randint(1, sides + 1, size=(trials, n_dice))
        sums = rolls.sum(axis=1)
        return float(np.mean(sums == target_sum))


class DiceAtLeastOneTemplate(BaseTemplate):
    template_id = "dice_at_least_one_value"
    description = "Probability of rolling at least one die showing the target value, out of N dice with S sides."
    required_params = ["n_dice", "sides", "target_value"]

    def solve_math(self, n_dice: int, sides: int, target_value: int) -> float:
        p_miss = (sides - 1) / sides
        return 1 - p_miss ** n_dice

    def simulate(self, n_dice: int, sides: int, target_value: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        rolls = np.random.randint(1, sides + 1, size=(trials, n_dice))
        has_target = np.any(rolls == target_value, axis=1)
        return float(np.mean(has_target))

TemplateRegistry().register(DiceSumTemplate)
TemplateRegistry().register(DiceAtLeastOneTemplate)