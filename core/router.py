import time
import concurrent.futures
from typing import Dict, Any
from templates.registry import REGISTRY

class StatsEngine:
    def __init__(self, timeout_sec: float = 1.5):
        self.timeout_sec = timeout_sec
        self.registry = REGISTRY

    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a statistical query using the dual-track methodology.
        """
        template_id = payload.get("template_id")
        params = payload.get("params", {})

        if not template_id or template_id not in self.registry:
            return {
                "status": "error",
                "message": f"Invalid or missing template_id: '{template_id}'"
            }

        # Initialize the target template
        template_class = self.registry[template_id]
        template = template_class()

        try:
            template.validate_params(params)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        start_time = time.perf_counter()
        
        # Track A: Analytical Math (with timeout)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(template.solve_math, **params)
                # Enforce the 1.5s timeout constraint
                result_val = future.result(timeout=self.timeout_sec)
            
            method_used = "analytical_math"

        except (concurrent.futures.TimeoutError, Exception) as math_err:
            # Track B: Monte Carlo Simulation Fallback
            try:
                result_val = template.simulate(**params)
                method_used = "monte_carlo_simulation"
            except Exception as sim_err:
                return {
                    "status": "error",
                    "message": f"Both tracks failed. Math Error: {math_err} | Sim Error: {sim_err}"
                }

        end_time = time.perf_counter()
        exec_time_ms = (end_time - start_time) * 1000

        return {
            "status": "success",
            "method": method_used,
            "value": result_val,
            "execution_time_ms": round(exec_time_ms, 4)
        }