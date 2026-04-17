import os

from openevolve import run_evolution

INITIAL_PROGRAM_PATH = os.path.join(os.path.dirname(__file__), "initial_program.py")
EVALUATOR_PATH = os.path.join(os.path.dirname(__file__), "evaluator.py")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

if __name__ == "__main__":
    result = run_evolution(
        initial_program=INITIAL_PROGRAM_PATH,
        evaluator=EVALUATOR_PATH,
        config=CONFIG_PATH
    )

    print("---------- Evolved Code ----------")
    print(result.best_code)
    print("---------- Evolved Code ----------")
