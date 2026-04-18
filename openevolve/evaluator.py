import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark import run_simulation


def evaluate(program_path):
    try:
        metrics = run_simulation(program_path, cleanup=True)
    except Exception as e:
        return {"combined_score": -1e9, "error": repr(e)}

    throughput = float(metrics["steady_state_total_token_throughput_tok_per_s"])
    hbf_rate = float(metrics["steady_state_hbf_write_rate_MBps"])

    return {
        "combined_score": -hbf_rate,
        "throughput_tok_per_s": throughput,
        "hbf_write_rate_MBps": hbf_rate,
    }
