from core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np
from ..registry import TemplateRegistry

class CoinFlipStreakTemplate(BaseTemplate):
    template_id = "coin_flip_exact_heads"
    description = "Probability of getting exactly K heads in N flips of a coin with heads probability P."
    required_params = ["n", "p", "k"]

    def solve_math(self, n: int, p: float, k: int) -> float:
        return float(stats.binom.pmf(k, n, p))

    def simulate(self, n: int, p: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        results = np.random.binomial(n, p, trials)
        return float(np.mean(results == k))


class CoinFlipConsecutiveTemplate(BaseTemplate):
    template_id = "coin_flip_consecutive_run"
    description = "Probability of observing a run of at least K consecutive heads within N flips of a fair or biased coin."
    required_params = ["n", "p", "k"]

    def solve_math(self, n: int, p: float, k: int) -> float:
        # Exact probability via Markov-chain DP over run-length states.
        # States 0..k-1 track the current trailing run of heads; state k is absorbing (success).
        if k > n:
            return 0.0
        q = 1 - p
        state_probs = np.zeros(k + 1)  # extra absorbing "success" state at index k
        state_probs[0] = 1.0
        for _ in range(n):
            new_state = np.zeros(k + 1)
            new_state[k] = state_probs[k]  # absorbing state stays
            for s in range(k):
                new_state[0] += state_probs[s] * q
                if s + 1 == k:
                    new_state[k] += state_probs[s] * p
                else:
                    new_state[s + 1] += state_probs[s] * p
            state_probs = new_state
        return float(state_probs[k])

    def simulate(self, n: int, p: float, k: int, trials: int = 100_000) -> float:
        trials = min(trials, 1_000_000)
        flips = np.random.random((trials, n)) < p
        found = np.zeros(trials, dtype=bool)
        run_len = np.zeros(trials, dtype=int)
        for i in range(n):
            run_len = np.where(flips[:, i], run_len + 1, 0)
            found = found | (run_len >= k)
        return float(np.mean(found))


TemplateRegistry().register(CoinFlipStreakTemplate)
TemplateRegistry().register(CoinFlipConsecutiveTemplate)