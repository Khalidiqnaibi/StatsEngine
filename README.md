#  StatsEngine : Deterministic Probability & Statistics Engine

> **Deterministic, Template-Driven Statistical Processing.**  
> *No LLM math guesswork. No dynamic code execution. 100% reproducible results.*

---

##  Philosophy

Language models are world-class translators, but unreliable calculators. 

The **StatsEngine** decouples linguistic understanding from mathematical execution:
1. **LLM / Parser Layer:** Normalizes natural language and extracts parameters into a rigid JSON schema.
2. **Template Registry:** Maps parameters to pre-tested, high-performance Python templates.
3. **Dual-Track Executor:** Solves using exact symbolic math (`SciPy`/`SymPy`) with an automatic fallback to vectorized Monte Carlo simulations (`NumPy`).
4. **Formatter Layer:** Formats the raw numeric output into human-readable prose.

---

##  System Flow


```
[ Natural Language Query ]
│
▼
[ LLM Normalizer ] ──► (Outputs: template_id + params JSON)
│
▼
[ Template Lookup ]
│
├──► Track A: Analytical Math (SymPy / SciPy)
│        │
│        └──► Success? ──► [ Raw Output ]
│                │
│             Timeout / Fail
│                │
└──► Track B: Monte Carlo Simulation (NumPy)
│
└──► [ Raw Output ]
│
▼
[ System Orchestrator ] ──► [ Response Formatter ]
```

---

##  Key Features

* **Zero Hallucination:** Math is handled strictly by Python binaries and proven libraries.
* **Dual-Track Execution:** Every template supports exact symbolic math and Monte Carlo simulation.
* **Sub-Millisecond Speed:** Pure math execution runs in microseconds. Vectorized Monte Carlo runs 100,000 trials in milliseconds.
* **Slot-Filling Ready:** Intercepts missing variables and prompts the user or orchestrator for context before execution.
* **Extensible Architecture:** Adding support for a new probability scenario takes under 5 minutes.

---

##  Project Structure

```text
stats_engine/
├── core/
│   ├── router.py          # Executes templates, handles timeouts & fallbacks
│   └── base_template.py   # Abstract Base Class for all templates
├── templates/
│   ├── registry.py        # Master dictionary of registered template IDs
│   ├── gaming/            # Dice, cards, coin flips, roulette
│   ├── distributions/     # Binomial, Normal, Poisson, Hypergeometric
│   └── finance/           # Expected value, gambler's ruin, risk calculations
├── schemas/               # Parameter validation schemas
└── tests/                 # Unit tests verifying math accuracy against Monte Carlo

```
##  Template Architecture
Every template inherits from BaseTemplate and implements two methods:
```python
from stats_engine.core.base_template import BaseTemplate
import scipy.stats as stats
import numpy as np

class BinomialTrialTemplate(BaseTemplate):
    template_id = "binomial_event"
    description = "Probability of exactly K successes in N independent trials with probability P."
    required_params = ["n", "p", "k"]

    def solve_math(self, n: int, p: float, k: int) -> float:
        """Exact probability using SciPy."""
        return float(stats.binom.pmf(k, n, p))

    def simulate(self, n: int, p: float, k: int, trials: int = 100_000) -> float:
        """Vectorized Monte Carlo fallback using NumPy."""
        results = np.random.binomial(n, p, trials)
        return float(np.mean(results == k))

```
##  Quickstart / Usage
### Direct Python Invocation
```python
from stats_engine.core.router import StatsEngine

engine = StatsEngine()

# Example payload extracted by the LLM/Parser layer
payload = {
    "template_id": "binomial_event",
    "params": {
        "n": 10,
        "p": 0.16666666666666666, # Rolling a 6 on a 6-sided die
        "k": 3
    }
}

# Run execution (Defaults to Math -> Fallback to Simulation)
result = engine.execute(payload)

print(result)
# Output:
# {
#   "status": "success",
#   "method": "analytical_math",
#   "value": 0.15504535938881268,
#   "execution_time_ms": 0.12
# }

```
##  Adding New Templates (The 3-Step Process)
 1. Create a new file in templates/<category>/my_scenario.py.
 2. Inherit from BaseTemplate, define required_params, and implement solve_math() and simulate().
 3. Register it in templates/registry.py:
```python
REGISTRY = {
    "binomial_event": BinomialTrialTemplate,
    "my_scenario": MyScenarioTemplate,  # <--- Added
}

```
##  Modular System Integration
 * **Intent Router:** Routes math, statistical, or predictive queries directly to the engine's tool wrapper.
 * **Context Layer:** Injects missing variables (such as deck sizes or win probabilities from local configuration/state) before execution.
 * **State Manager:** Logs execution_time_ms, method_used, and raw numeric outputs into current runtime state.
 * **Formatting Layer:** Receives raw numeric output and context strings to generate precise user-facing summaries.
##  Safeguards & Limits
 * **Execution Timeout:** Math solver thread automatically terminates after **1.5s** and triggers Monte Carlo fallback to prevent infinite loops.
 * **Simulation Cap:** Default Monte Carlo cap set to **1,000,000 trials** to keep memory consumption low.
 * **No eval():** Complete prohibition on execution of dynamic LLM-generated code strings.
