import argparse
import os

from openevolve import run_evolution

EVALUATOR_PATH = os.path.join(os.path.dirname(__file__), "evaluator.py")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
EVOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "evolutions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("initial_program", help="Path to the initial policy program.")
    parser.add_argument("name", help="Name for the best result, written to evolutions/<name>.py.")
    args = parser.parse_args()

    result = run_evolution(
        initial_program=args.initial_program,
        evaluator=EVALUATOR_PATH,
        config=CONFIG_PATH,
    )

    os.makedirs(EVOLUTIONS_DIR, exist_ok=True)
    os.chmod(EVOLUTIONS_DIR, 0o777)
    out_path = os.path.join(EVOLUTIONS_DIR, f"{args.name}.py")
    with open(out_path, "w") as f:
        f.write(result.best_code)
    os.chmod(out_path, 0o666)
