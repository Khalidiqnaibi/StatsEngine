"""
One valid, hand-picked params dict per template_id. This is the ONLY place
sample inputs live — the smoke test in test_all_templates.py iterates the
live registry and looks up fixtures here, so adding a new template later
means adding one dict entry here, not a new test file.
"""

VALID_PARAMS = {
    # --- gaming ---
    "card_draw_probability": {"deck_size": 52, "target_cards": 13, "draws": 5, "k": 2},
    "card_at_least_one": {"deck_size": 47, "target_cards": 9, "draws": 2},
    "coin_flip_exact_heads": {"n": 10, "p": 0.5, "k": 5},
    "coin_flip_consecutive_run": {"n": 10, "p": 0.5, "k": 3},
    "dice_sum_probability": {"n_dice": 2, "sides": 6, "target_sum": 7},
    "dice_at_least_one_value": {"n_dice": 3, "sides": 6, "target_value": 6},
    "roulette_single_number": {"pockets": 37},
    "roulette_consecutive_color": {"p": 18 / 37, "k": 3},

    # --- distributions ---
    "binomial_event": {"n": 10, "p": 1 / 6, "k": 3},
    "exponential_wait_time": {"lam": 0.5, "t": 2},
    "uniform_range_probability": {"low": 0, "high": 10, "lower": 2, "upper": 6},
    "geometric_first_success": {"p": 0.3, "k": 4},
    "negative_binomial_event": {"r": 3, "p": 0.4, "k": 6},
    "hypergeometric_event": {"population_size": 50, "success_states": 10, "draws": 5, "k": 2},
    "normal_less_than": {"mean": 0, "std": 1, "x": 1.96},
    "normal_between": {"mean": 0, "std": 1, "lower": -1, "upper": 1},
    "poisson_event": {"lam": 7, "k": 5},
    "poisson_at_least": {"lam": 7, "k": 12},

    # --- finance ---
    "expected_value_discrete": {"outcomes": [10, -5, 20], "probabilities": [0.3, 0.5, 0.2]},
    "gamblers_ruin_probability": {"start": 20, "goal": 50, "p": 0.52},
    "value_at_risk": {"mean_return": 0.0004, "std_return": 0.012, "confidence": 0.95, "portfolio_value": 250_000},

    # --- inference / descriptive ---
    "summary_statistics": {"data": [182, 190, 175, 210, 205, 198, 188, 250, 300, 195]},
    "z_score": {"x": 82, "mean": 70, "std": 8},

    # --- inference / confidence intervals ---
    "confidence_interval_mean": {"sample_mean": 50, "sample_std": 5, "n": 30, "confidence": 0.95},
    "confidence_interval_proportion": {"successes": 42, "n": 100, "confidence": 0.95},

    # --- inference / hypothesis testing ---
    "one_sample_z_test": {"sample_mean": 105, "pop_mean": 100, "pop_std": 15, "n": 40},
    "one_sample_t_test": {"data": [5, 6, 7, 5, 6, 8, 7, 9], "pop_mean": 5},
    "two_sample_t_test": {
        "data_a": [182, 190, 175, 210, 205, 198, 188, 250, 300, 195],
        "data_b": [160, 172, 168, 175, 180, 165, 190, 171, 178, 183],
    },
    "chi_square_goodness_of_fit": {"observed": [42, 38, 55, 30, 35], "expected": [40, 40, 40, 40, 40]},

    # --- inference / variable relationships ---
    "pearson_correlation": {"data_x": [1, 2, 3, 4, 5], "data_y": [2, 4, 5, 4, 5]},
    "simple_linear_regression": {"data_x": [200, 400, 600, 800, 1000, 1200], "data_y": [18, 35, 41, 63, 70, 91], "predict_x": 1500},
}
