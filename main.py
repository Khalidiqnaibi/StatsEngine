from core.router import StatsEngine
import json

def run_test():
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

    # Execute primary track (Math)
    print("Executing Standard Payload...")
    result = engine.execute(payload)
    print(json.dumps(result, indent=2))
    
    # Test Fallback Track (Simulation) by forcing a math failure
    print("\nExecuting Forced Fallback Payload (triggering Exception in Math track)...")
    
    # Mocking a math failure by temporarily overriding the method
    template_instance = engine.registry["binomial_event"]
    original_math = template_instance.solve_math
    template_instance.solve_math = lambda self, **kwargs: 1 / 0 # Force ZeroDivisionError
    
    fallback_result = engine.execute(payload)
    print(json.dumps(fallback_result, indent=2))
    
    # Restore original method
    template_instance.solve_math = original_math

if __name__ == "__main__":
    run_test()