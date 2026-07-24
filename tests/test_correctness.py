"""
Correctness tests against hand-computed / textbook answers, and math-vs-
simulation agreement checks. These are a deliberately small set covering
one template per category - the smoke test in test_all_templates.py is
what gives blanket coverage across all 31; this file is for "is the number
actually right", not "does it crash".
"""

import templates  # noqa: F401
from core.router import StatsEngine

engine = StatsEngine()


def test_coin_flip_fair_probability_exactly_half():
    # P(exactly 5 heads in 10 fair flips) is a known textbook value: 252/1024
    result = engine.execute({
        "template_id": "coin_flip_exact_heads",
        "params": {"n": 10, "p": 0.5, "k": 5},
    })
    assert result["status"] == "success"
    assert abs(result["value"] - 252 / 1024) < 1e-9


def test_roulette_single_number_european():
    result = engine.execute({"template_id": "roulette_single_number", "params": {"pockets": 37}})
    assert result["status"] == "success"
    assert abs(result["value"] - 1 / 37) < 1e-9


def test_normal_standard_symmetry():
    # P(X < 0) for a standard normal must be exactly 0.5
    result = engine.execute({
        "template_id": "normal_less_than",
        "params": {"mean": 0, "std": 1, "x": 0},
    })
    assert result["status"] == "success"
    assert abs(result["value"] - 0.5) < 1e-9


def test_confidence_interval_mean_is_centered_on_sample_mean():
    result = engine.execute({
        "template_id": "confidence_interval_mean",
        "params": {"sample_mean": 50, "sample_std": 5, "n": 30, "confidence": 0.95},
    })
    assert result["status"] == "success"
    ci = result["value"]
    assert ci["lower"] < 50 < ci["upper"]
    assert abs((ci["lower"] + ci["upper"]) / 2 - 50) < 1e-6


def test_two_sample_t_test_detects_an_obvious_difference():
    result = engine.execute({
        "template_id": "two_sample_t_test",
        "params": {"data_a": [1, 2, 1, 2, 1], "data_b": [100, 101, 99, 102, 98]},
    })
    assert result["status"] == "success"
    assert result["value"]["p_value"] < 0.01  # obviously different groups


def test_pearson_correlation_perfect_positive_line():
    result = engine.execute({
        "template_id": "pearson_correlation",
        "params": {"data_x": [1, 2, 3, 4, 5], "data_y": [2, 4, 6, 8, 10]},
    })
    assert result["status"] == "success"
    assert abs(result["value"]["r"] - 1.0) < 1e-9


def test_dual_track_fallback_agrees_with_analytical_math():
    """
    Sabotage the analytical track for one template and confirm the Monte
    Carlo fallback still lands within a reasonable tolerance of the true
    analytical answer - this is the core promise of the dual-track design.
    """
    template_instance = engine.registry["binomial_event"]
    true_value = template_instance().solve_math(n=10, p=1 / 6, k=3)

    original_math = template_instance.solve_math
    template_instance.solve_math = lambda self, **kwargs: 1 / 0
    try:
        result = engine.execute({
            "template_id": "binomial_event",
            "params": {"n": 10, "p": 1 / 6, "k": 3, "trials": 500_000},
        })
    finally:
        template_instance.solve_math = original_math

    assert result["status"] == "success"
    assert result["method"] == "monte_carlo_simulation"
    assert abs(result["value"] - true_value) < 0.01  # within 1pp at 500k trials
