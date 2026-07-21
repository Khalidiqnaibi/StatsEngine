from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry

class CardDrawTemplate(BaseTemplate):
    template_id = "card_draw_probability"
    description = "Probability of drawing exactly K target cards when drawing N cards without replacement from a deck of given size with a given number of target cards."
    required_params = ["deck_size", "target_cards", "draws", "k"]

    def solve_math(self, deck_size: int, target_cards: int, draws: int, k: int) -> float:
        return float(stats.hypergeom.pmf(k, deck_size, target_cards, draws))

    def simulate(self, deck_size: int, target_cards: int, draws: int, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.hypergeometric(target_cards, deck_size - target_cards, draws, trials)
        return float(np.mean(results == k))


class CardAtLeastOneTemplate(BaseTemplate):
    template_id = "card_at_least_one"
    description = "Probability of drawing at least one target card when drawing N cards without replacement from a deck."
    required_params = ["deck_size", "target_cards", "draws"]

    def solve_math(self, deck_size: int, target_cards: int, draws: int) -> float:
        return float(1 - stats.hypergeom.pmf(0, deck_size, target_cards, draws))

    def simulate(self, deck_size: int, target_cards: int, draws: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.hypergeometric(target_cards, deck_size - target_cards, draws, trials)
        return float(np.mean(results >= 1))

TemplateRegistry().register(CardDrawTemplate)
TemplateRegistry().register(CardAtLeastOneTemplate)