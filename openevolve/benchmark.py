"""Run the simulator with a candidate `_device_allocate_policy` swapped in.

Writes the simulator's metrics JSON to ``openevolve/results/<program_stem>.json``
and returns the parsed dict. Usable as a library or CLI:
``python openevolve/benchmark.py path/to/policy.py``.
"""
import json
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
POLICY_PATH = os.path.join(REPO_ROOT, "inference_serving", "device_allocate_policy.py")
BACKUP_PATH = POLICY_PATH + ".evolve_bak"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_simulation(program_path, show_output=False, cleanup=False):
    program_name = os.path.splitext(os.path.basename(program_path))[0]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.chmod(RESULTS_DIR, 0o777)
    metrics_path = os.path.join(RESULTS_DIR, f"{program_name}.json")

    shutil.copy2(POLICY_PATH, BACKUP_PATH)
    try:
        shutil.copy2(program_path, POLICY_PATH)
        subprocess.run(
            [
                "python", "main.py",
                "--cluster-config", "our_cluster_config/6_hbm_2_hbf.json",
                "--fp", "16",
                "--block-size", "16",
                "--dataset", "dataset/ShareGPT_Vicuna_unfiltered_req5000_rate200_sys10x256.jsonl",
                "--output", "output/example_single_run.csv",
                "--num-req", "3000",
                "--max-batch", "512",
                "--enable-prefix-caching",
                "--enable-hbf-offload",
                "--enable-hbf-kv",
                "--log-interval", "1",
                "--log-level", "WARNING",
                "--metrics-output", metrics_path,
            ],
            cwd=REPO_ROOT,
            check=True,
            stdout=None if show_output else subprocess.DEVNULL,
            stderr=None if show_output else subprocess.DEVNULL,
        )
    finally:
        shutil.move(BACKUP_PATH, POLICY_PATH)

    os.chmod(metrics_path, 0o666)
    with open(metrics_path) as f:
        metrics = json.load(f)
    if cleanup:
        os.remove(metrics_path)
    return metrics


if __name__ == "__main__":
    run_simulation(sys.argv[1], show_output=True)
