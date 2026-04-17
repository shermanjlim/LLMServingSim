import json
import os
import re
import shutil
import subprocess
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
MEMORY_MODEL_PATH = os.path.join(REPO_ROOT, "inference_serving", "memory_model.py")
BACKUP_PATH = MEMORY_MODEL_PATH + ".evolve_bak"
METRICS_FILENAME = "baseline.json"
METRICS_PATH = os.path.join(REPO_ROOT, METRICS_FILENAME)

CMD = [
    "python", "main.py",
    "--cluster-config", "our_cluster_config/6_hbm_2_hbf.json",
    "--fp", "16",
    "--block-size", "16",
    "--dataset", "dataset/ShareGPT_Vicuna_unfiltered_req5000_rate200.jsonl",
    "--output", "output/example_single_run.csv",
    "--num-req", "2000",
    "--max-batch", "512",
    "--enable-prefix-caching",
    "--enable-hbf-offload",
    "--enable-hbf-kv",
    "--log-interval", "1",
    "--log-level", "WARNING",
    "--metrics-output", METRICS_FILENAME,
]

EVOLVE_BLOCK_RE = re.compile(
    r"#\s*EVOLVE-BLOCK-START\s*\n(?P<body>.*?)#\s*EVOLVE-BLOCK-END",
    re.DOTALL,
)
METHOD_RE = re.compile(
    r"^    def apply_kv_cache_events\(self\):.*?(?=^    def )",
    re.DOTALL | re.MULTILINE,
)


def _extract_evolved_function(program_path):
    with open(program_path) as f:
        src = f.read()
    m = EVOLVE_BLOCK_RE.search(src)
    if m is None:
        raise RuntimeError("EVOLVE-BLOCK markers not found in program file")
    body = m.group("body").strip("\n")
    if "def apply_kv_cache_events" not in body:
        raise RuntimeError("evolved block must define apply_kv_cache_events")
    return body


def _patch_memory_model(evolved_text):
    with open(MEMORY_MODEL_PATH) as f:
        original = f.read()
    indented_lines = []
    for line in evolved_text.splitlines():
        indented_lines.append(("    " + line) if line.strip() else line)
    indented = "\n".join(indented_lines) + "\n\n"
    patched, n = METHOD_RE.subn(indented, original, count=1)
    if n == 0:
        raise RuntimeError("could not locate apply_kv_cache_events in memory_model.py")
    with open(MEMORY_MODEL_PATH, "w") as f:
        f.write(patched)


def _restore_memory_model():
    if os.path.exists(BACKUP_PATH):
        shutil.move(BACKUP_PATH, MEMORY_MODEL_PATH)


def evaluate(program_path):
    shutil.copy2(MEMORY_MODEL_PATH, BACKUP_PATH)
    try:
        evolved = _extract_evolved_function(program_path)
        _patch_memory_model(evolved)

        if os.path.exists(METRICS_PATH):
            os.remove(METRICS_PATH)

        start = time.perf_counter()
        proc = subprocess.run(
            CMD,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        elapsed = time.perf_counter() - start

        if proc.returncode != 0:
            return {
                "combined_score": 0.0,
                "error": f"simulation rc={proc.returncode}",
                "stderr_tail": proc.stderr[-800:],
                "elapsed_s": elapsed,
            }
        if not os.path.exists(METRICS_PATH):
            return {
                "combined_score": 0.0,
                "error": "metrics file not produced",
                "stdout_tail": proc.stdout[-800:],
                "elapsed_s": elapsed,
            }

        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        throughput = metrics.get("steady_state_total_token_throughput_tok_per_s")
        hbf_rate = metrics.get("steady_state_hbf_write_rate_MBps")
        if throughput is None or hbf_rate is None:
            return {
                "combined_score": 0.0,
                "error": "required steady-state metrics missing",
                "metrics": metrics,
                "elapsed_s": elapsed,
            }

        # Reward throughput, penalize HBF write rate. Adding 1.0 to the
        # denominator keeps the score finite when writes go to zero and gives
        # a smooth gradient near the low-write regime we care about.
        combined = float(throughput) / (1.0 + float(hbf_rate))

        return {
            "combined_score": combined,
            "throughput_tok_per_s": float(throughput),
            "hbf_write_rate_MBps": float(hbf_rate),
            "elapsed_s": elapsed,
        }
    except subprocess.TimeoutExpired:
        return {"combined_score": 0.0, "error": "simulation timeout"}
    except Exception as e:
        return {"combined_score": 0.0, "error": repr(e)}
    finally:
        _restore_memory_model()
