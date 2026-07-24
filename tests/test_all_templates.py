"""
Registry-driven smoke test. This single parametrized test covers every
template currently registered - adding a new template means adding a
fixtures.py entry, NOT a new test file.

What it checks, per template:
  1. The fixture set actually satisfies required_params (catches drift
     between a template's signature and its fixture).
  2. solve_math() runs and returns a JSON-serializable result (float or
     dict of floats/ints) without raising.
  3. simulate() runs independently and returns the same shape of result.
  4. Going through the full StatsEngine().execute() (the dual-track path
     a real caller uses) succeeds and reports status == "success".
  5. Every template in fixtures.py corresponds to a real registered
     template_id, and every registered template_id has a fixture (so a
     newly added template can't silently ship without a test).
"""

import math
import pytest

import templates  # noqa: F401  (registers all templates as a side effect)
from templates.registry import template_registry
from core.router import StatsEngine
from tests.fixtures import VALID_PARAMS

ALL_TEMPLATE_IDS = sorted(template_registry._registry.keys())


def _is_json_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _assert_valid_result(value) -> None:
    """A template result must be a finite number, or a dict of finite numbers."""
    if isinstance(value, dict):
        assert value, "dict result must not be empty"
        for k, v in value.items():
            assert _is_json_number(v), f"non-numeric/non-finite value for key '{k}': {v!r}"
    else:
        assert _is_json_number(value), f"result is not a finite number: {value!r}"


def test_every_registered_template_has_a_fixture():
    missing = set(ALL_TEMPLATE_IDS) - set(VALID_PARAMS.keys())
    assert not missing, f"No test fixture for: {sorted(missing)} - add one to tests/fixtures.py"


def test_every_fixture_maps_to_a_real_template():
    stale = set(VALID_PARAMS.keys()) - set(ALL_TEMPLATE_IDS)
    assert not stale, f"Fixtures reference unknown template_ids: {sorted(stale)}"


@pytest.mark.parametrize("template_id", ALL_TEMPLATE_IDS)
def test_solve_math_runs_and_returns_valid_result(template_id):
    params = VALID_PARAMS[template_id]
    template_cls = template_registry.get(template_id)
    template = template_cls()

    template.validate_params(params)  # should not raise - fixture must be complete
    result = template.solve_math(**params)
    _assert_valid_result(result)


@pytest.mark.parametrize("template_id", ALL_TEMPLATE_IDS)
def test_simulate_runs_and_returns_valid_result(template_id):
    params = VALID_PARAMS[template_id]
    template_cls = template_registry.get(template_id)
    template = template_cls()

    result = template.simulate(**params)
    _assert_valid_result(result)


@pytest.mark.parametrize("template_id", ALL_TEMPLATE_IDS)
def test_full_engine_execute_succeeds(template_id):
    engine = StatsEngine()
    payload = {"template_id": template_id, "params": VALID_PARAMS[template_id]}
    result = engine.execute(payload)

    assert result["status"] == "success", f"{template_id} failed: {result}"
    assert result["method"] in ("analytical_math", "monte_carlo_simulation")
    _assert_valid_result(result["value"])


def test_missing_required_param_returns_error_not_crash():
    result = StatsEngine().execute({"template_id": "binomial_event", "params": {"n": 10}})
    assert result["status"] == "error"


def test_unknown_template_id_returns_error_not_crash():
    result = StatsEngine().execute({"template_id": "not_a_real_template", "params": {}})
    assert result["status"] == "error"
