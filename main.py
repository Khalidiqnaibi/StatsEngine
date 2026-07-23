"""
StatsEngine Showcase
=====================
A tour of what the engine can do now that it spans gaming, distributions,
finance, and full statistical inference (descriptive stats, confidence
intervals, hypothesis testing, and variable relationships) - all through
the same dual-track (analytical math + Monte Carlo fallback) interface.

Also renders a chart for every scenario where a picture actually adds
insight beyond the raw number (distribution shape, CI bounds, regression
fit, before/after comparison, dual-track timing) and saves them to
./charts/.

Run: python main.py
Requires: numpy, scipy, matplotlib (see requirements.txt)
"""

from core.router import StatsEngine
from templates.registry import template_registry
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend, no display server required
import matplotlib.pyplot as plt
import scipy.stats as stats

CHART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHART_DIR, exist_ok=True)


def banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def show(label: str, payload: dict, engine: StatsEngine) -> dict:
    result = engine.execute(payload)
    print(f"\n--- {label} ---")
    print(f"template_id: {payload['template_id']}")
    print(json.dumps(result, indent=2))
    return result


def save(fig, name: str) -> None:
    path = os.path.join(CHART_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[chart saved -> charts/{name}]")


def run_test():
    engine = StatsEngine()

    # ------------------------------------------------------------------
    # 0. Catalog
    # ------------------------------------------------------------------
    banner("0. TEMPLATE CATALOG")
    catalog = template_registry.list_templates()
    print(f"{len(catalog)} templates registered across gaming, distributions, "
          f"finance, and inference:\n")
    for entry in catalog:
        print(f"  - {entry['template_id']:<32} {entry['required_params']}")

    # ------------------------------------------------------------------
    # 1. Gaming: poker flush draw across turn and river
    # ------------------------------------------------------------------
    banner("1. GAMING - Poker flush draw across turn and river")
    show(
        "Odds of catching a flush card at least once (turn+river, 9 outs, 47 unseen)",
        {"template_id": "card_at_least_one",
         "params": {"deck_size": 47, "target_cards": 9, "draws": 2}},
        engine,
    )

    # Chart: full hit-count distribution for context (0, 1, or 2 flush cards)
    hit_probs = [
        stats.hypergeom.pmf(k, 47, 9, 2) for k in range(0, 3)
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["0 cards", "1 card", "2 cards"], hit_probs,
                   color=["#c0392b", "#2980b9", "#27ae60"])
    ax.set_title("Flush cards caught across turn + river")
    ax.set_ylabel("Probability")
    for b, p in zip(bars, hit_probs):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.01, f"{p:.1%}", ha="center")
    save(fig, "01_poker_flush_distribution.png")

    # ------------------------------------------------------------------
    # 2. Distributions: call-center overload risk (Poisson)
    # ------------------------------------------------------------------
    banner("2. DISTRIBUTIONS - Call center overload risk")
    show(
        "P(at least 12 calls in an hour | average rate = 7/hr)",
        {"template_id": "poisson_at_least", "params": {"lam": 7, "k": 12}},
        engine,
    )

    lam = 7
    ks = np.arange(0, 21)
    pmf = stats.poisson.pmf(ks, lam)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if k >= 12 else "#3498db" for k in ks]
    ax.bar(ks, pmf, color=colors)
    ax.axvline(12, color="black", linestyle="--", linewidth=1)
    ax.set_title("Poisson(7): call volume per hour\nred = overload zone (>=12 calls)")
    ax.set_xlabel("Calls in an hour")
    ax.set_ylabel("Probability")
    save(fig, "02_poisson_overload.png")

    # ------------------------------------------------------------------
    # 3. Finance: VaR + gambler's ruin
    # ------------------------------------------------------------------
    banner("3. FINANCE - Portfolio risk stack")
    mean_r, std_r, conf, pv = 0.0004, 0.012, 0.95, 250_000
    var_result = show(
        "95% 1-day VaR on a $250k portfolio (mean 0.04% / day, std 1.2%)",
        {"template_id": "value_at_risk",
         "params": {"mean_return": mean_r, "std_return": std_r,
                     "confidence": conf, "portfolio_value": pv}},
        engine,
    )
    show(
        "P(strategy reaches +50 units before busting) starting at 20, edge p=0.52",
        {"template_id": "gamblers_ruin_probability",
         "params": {"start": 20, "goal": 50, "p": 0.52}},
        engine,
    )

    # Chart: portfolio P&L distribution with the VaR cutoff shaded
    x = np.linspace(mean_r - 4 * std_r, mean_r + 4 * std_r, 500)
    pdf = stats.norm.pdf(x, mean_r, std_r) * pv
    pnl_x = x * pv
    var_loss = -var_result["value"]  # negative = loss threshold in dollars
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pnl_x, pdf, color="#2c3e50")
    ax.fill_between(pnl_x, pdf, where=(pnl_x <= var_loss), color="#e74c3c", alpha=0.6,
                     label=f"5% tail (VaR = ${-var_loss:,.0f})")
    ax.axvline(var_loss, color="#c0392b", linestyle="--")
    ax.set_title("Portfolio 1-day P&L distribution")
    ax.set_xlabel("P&L ($)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, "03_portfolio_var.png")

    # ------------------------------------------------------------------
    # 4/5. Descriptive stats + confidence interval on the same dataset
    # ------------------------------------------------------------------
    banner("4. INFERENCE / DESCRIPTIVE - Summarize raw response-time data")
    response_times_ms = [182, 190, 175, 210, 205, 198, 188, 250, 300, 195]
    show(
        "Summary statistics over 10 API response-time samples (ms)",
        {"template_id": "summary_statistics", "params": {"data": response_times_ms}},
        engine,
    )

    arr = np.array(response_times_ms, dtype=float)
    banner("5. INFERENCE / CONFIDENCE INTERVAL - True mean latency, 95% CI")
    ci_result = show(
        "95% CI for true mean latency given the sample above",
        {"template_id": "confidence_interval_mean",
         "params": {"sample_mean": float(arr.mean()), "sample_std": float(arr.std(ddof=1)),
                     "n": int(arr.size), "confidence": 0.95}},
        engine,
    )

    # Chart: raw samples as a strip plot + the CI band around the mean
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(arr, np.zeros_like(arr) + 0.5, color="#7f8c8d", zorder=3, label="samples")
    ax.axvline(arr.mean(), color="#2980b9", label=f"sample mean = {arr.mean():.1f} ms")
    ax.axvspan(ci_result["value"]["lower"], ci_result["value"]["upper"],
               color="#2980b9", alpha=0.2, label="95% CI for true mean")
    ax.set_yticks([])
    ax.set_xlabel("Response time (ms)")
    ax.set_title("Latency samples & 95% confidence interval for the true mean")
    ax.legend(loc="upper right")
    save(fig, "05_latency_confidence_interval.png")

    # ------------------------------------------------------------------
    # 6. Hypothesis testing: migration before/after + ticket distribution
    # ------------------------------------------------------------------
    banner("6. INFERENCE / HYPOTHESIS TESTING - Did the migration help?")
    before = [182, 190, 175, 210, 205, 198, 188, 250, 300, 195]
    after = [160, 172, 168, 175, 180, 165, 190, 171, 178, 183]
    show(
        "Two-sample t-test: response times before vs. after migration",
        {"template_id": "two_sample_t_test", "params": {"data_a": before, "data_b": after}},
        engine,
    )
    observed = [42, 38, 55, 30, 35]
    expected = [40, 40, 40, 40, 40]
    show(
        "Chi-square goodness of fit: are support tickets evenly split across 5 categories?",
        {"template_id": "chi_square_goodness_of_fit",
         "params": {"observed": observed, "expected": expected}},
        engine,
    )

    # Chart A: before/after boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([before, after], label=["Before", "After"], patch_artist=True,
               boxprops=dict(facecolor="#aed6f1"))
    ax.set_ylabel("Response time (ms)")
    ax.set_title("Migration impact on response times")
    save(fig, "06a_migration_before_after.png")

    # Chart B: observed vs expected ticket counts
    categories = [f"Cat {i+1}" for i in range(len(observed))]
    x_pos = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    ax.bar(x_pos - width / 2, observed, width, label="Observed", color="#e67e22")
    ax.bar(x_pos + width / 2, expected, width, label="Expected", color="#95a5a6")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Ticket count")
    ax.set_title("Support ticket categories: observed vs. expected")
    ax.legend()
    save(fig, "06b_ticket_chi_square.png")

    # ------------------------------------------------------------------
    # 7. Variable relationships: correlation + regression
    # ------------------------------------------------------------------
    banner("7. INFERENCE / VARIABLE RELATIONSHIPS - Ad spend vs. signups")
    ad_spend = [200, 400, 600, 800, 1000, 1200]
    signups = [18, 35, 41, 63, 70, 91]
    show(
        "Pearson correlation between weekly ad spend and signups",
        {"template_id": "pearson_correlation", "params": {"data_x": ad_spend, "data_y": signups}},
        engine,
    )
    reg_result = show(
        "OLS regression: predict signups at $1,500 weekly ad spend",
        {"template_id": "simple_linear_regression",
         "params": {"data_x": ad_spend, "data_y": signups, "predict_x": 1500}},
        engine,
    )

    slope = reg_result["value"]["slope"]
    intercept = reg_result["value"]["intercept"]
    r2 = reg_result["value"]["r_squared"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(ad_spend, signups, color="#2c3e50", zorder=3, label="observed weeks")
    x_line = np.linspace(0, 1600, 50)
    ax.plot(x_line, slope * x_line + intercept, color="#e74c3c",
            label=f"OLS fit (R\u00b2={r2:.2f})")
    ax.scatter([1500], [reg_result["value"]["prediction"]], color="#27ae60", s=90,
               zorder=4, label=f"prediction @ $1500 = {reg_result['value']['prediction']:.0f}")
    ax.set_xlabel("Weekly ad spend ($)")
    ax.set_ylabel("Signups")
    ax.set_title("Ad spend vs. signups: fit + out-of-sample prediction")
    ax.legend()
    save(fig, "07_adspend_regression.png")

    # ------------------------------------------------------------------
    # 8. Dual-track proof: force math to fail, fall back to Monte Carlo
    # ------------------------------------------------------------------
    banner("8. DUAL-TRACK FALLBACK - Forcing a math failure mid-flight")
    payload = {"template_id": "binomial_event", "params": {"n": 10, "p": 1 / 6, "k": 3}}
    math_result = show("Standard analytical execution", payload, engine)

    template_instance = engine.registry["binomial_event"]
    original_math = template_instance.solve_math
    template_instance.solve_math = lambda self, **kwargs: 1 / 0  # force ZeroDivisionError
    payload["params"]["trials"] = 1_000_000
    sim_result = show("Same query, math track sabotaged -> Monte Carlo fallback kicks in",
                       payload, engine)
    template_instance.solve_math = original_math

    # Chart: value agreement + execution time cost of each track
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    tracks = ["Analytical\nmath", "Monte Carlo\n(1M trials)"]
    values = [math_result["value"], sim_result["value"]]
    times = [math_result["execution_time_ms"], sim_result["execution_time_ms"]]

    ax1.bar(tracks, values, color=["#2980b9", "#8e44ad"])
    ax1.set_title("Result agreement across tracks")
    ax1.set_ylabel("P(exactly 3 sixes in 10 rolls)")
    for i, v in enumerate(values):
        ax1.text(i, v + 0.002, f"{v:.4f}", ha="center")

    ax2.bar(tracks, times, color=["#2980b9", "#8e44ad"])
    ax2.set_title("Execution time by track")
    ax2.set_ylabel("ms")
    for i, t in enumerate(times):
        ax2.text(i, t + max(times) * 0.02, f"{t:.2f} ms", ha="center")

    save(fig, "08_dual_track_comparison.png")

    banner("DONE")
    print(f"Every result above came from the same StatsEngine().execute(payload) "
          f"call - gaming, distributions, finance, and full inference all share "
          f"one dual-track interface.\n"
          f"Charts written to: {CHART_DIR}")


if __name__ == "__main__":
    run_test()