"""
StatsEngine MCP Server
=======================
Thin MCP adapter over the existing StatsEngine. This is the ONLY file in the
repo that imports the `mcp` package — core/ and templates/ remain pure Python
with zero MCP dependency, so StatsEngine keeps working as a plain library
(Flask app, RAG pipeline, notebook, etc.) whether or not this file, or the
`mcp` package, is even installed.

Every tool here is a direct pass-through to code that already exists in
core/router.py and templates/registry.py — no new business logic lives here.

Run directly (stdio transport, e.g. for Claude Desktop / Claude Code):
    python -m mcp_server.server

Or via the installed console-script (see pyproject.toml / README):
    statsengine-mcp
"""

from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from core.router import StatsEngine
from templates.registry import template_registry
import templates  # noqa: F401  (import side-effect: registers all templates)

mcp = FastMCP(
    "StatsEngine",
    instructions=(
        "Probability, statistics, and risk calculator. Call list_stats_templates "
        "first to see what's available, then describe_stats_template for a given "
        "template's exact parameter schema before calling run_stats_template. "
        "Every template runs a dual-track computation: an exact analytical solution, "
        "with an automatic Monte Carlo fallback if the analytical track fails or "
        "times out — the response tells you which one actually ran."
    ),
)

_engine = StatsEngine()


@mcp.tool()
def list_stats_templates() -> List[Dict[str, Any]]:
    """
    List every statistics/probability template StatsEngine currently supports,
    across gaming, distributions, finance, and inference (descriptive stats,
    confidence intervals, hypothesis testing, variable relationships).

    Returns each template's id, a one-line description, and its required
    parameter names — enough to pick the right template_id for a task and
    call run_stats_template directly for simple cases. For a full parameter
    schema (types, optional params, defaults), call describe_stats_template.
    """
    return template_registry.list_templates()


@mcp.tool()
def describe_stats_template(template_id: str) -> Dict[str, Any]:
    """
    Get the full parameter schema for one template: which params are
    required, their expected types (inferred from the template's type hints),
    and any optional params with defaults (e.g. `trials`, `predict_x`).

    Call this before run_stats_template whenever a template's required_params
    alone (from list_stats_templates) isn't enough to know what shape of data
    to pass — e.g. whether a param wants a single number or a list.
    """
    template_cls = template_registry.get(template_id)
    if template_cls is None:
        return {
            "status": "error",
            "message": f"Unknown template_id: '{template_id}'. "
                       f"Call list_stats_templates to see valid ids.",
        }

    import inspect

    sig = inspect.signature(template_cls.solve_math)
    params_schema = {}
    for name, param in sig.parameters.items():
        if name in ("self", "kwargs"):
            continue
        annotation = param.annotation
        type_name = getattr(annotation, "__name__", str(annotation)) \
            if annotation is not inspect.Parameter.empty else "unspecified"
        entry: Dict[str, Any] = {
            "type": type_name,
            "required": param.default is inspect.Parameter.empty,
        }
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        params_schema[name] = entry

    return {
        "status": "success",
        "template_id": template_id,
        "description": template_cls.description,
        "required_params": template_cls.required_params,
        "params": params_schema,
    }


@mcp.tool()
def run_stats_template(template_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a StatsEngine template and get back the result.

    Args:
        template_id: One of the ids returned by list_stats_templates
                      (e.g. "binomial_event", "confidence_interval_mean",
                      "two_sample_t_test", "simple_linear_regression").
        params: Keyword arguments the template needs. Check required_params
                 from list_stats_templates, or call describe_stats_template
                 for the full schema first if unsure.

    Returns a dict with:
        status: "success" | "error" | "needs_input"
        method: "analytical_math" or "monte_carlo_simulation" (which track
                 actually produced the result — simulation is an automatic
                 fallback, not something the caller chooses)
        value: the result (a number, or a dict for multi-value templates
                 like confidence intervals or regression)
        execution_time_ms: how long the successful track took

    If required params are missing, returns status="needs_input" with the
    list of missing param names instead of a bare error, so the caller can
    ask the user for them and retry.
    """
    template_cls = template_registry.get(template_id)
    if template_cls is None:
        return {
            "status": "error",
            "message": f"Unknown template_id: '{template_id}'. "
                       f"Call list_stats_templates to see valid ids.",
        }

    missing = [p for p in template_cls.required_params if p not in (params or {})]
    if missing:
        return {
            "status": "needs_input",
            "template_id": template_id,
            "missing": missing,
            "message": f"Missing required parameters for {template_id}: {missing}",
        }

    return _engine.execute({"template_id": template_id, "params": params})


def main() -> None:
    """Entry point for the `statsengine-mcp` console script (stdio transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
