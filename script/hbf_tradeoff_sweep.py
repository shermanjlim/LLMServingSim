#!/usr/bin/env python3
"""
Sweep HBF scenarios across offered load scales and plot throughput/latency/write tradeoffs.

Default scenarios:
  - baseline
  - hbf_weights_only
  - hbf_weights_kv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import pickle
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path


AVG_GEN_TPUT_RE = re.compile(r"Average generation throughput \(tok/s\):\s+([\d.]+)")
TOTAL_TPUT_RE = re.compile(r"Total token throughput \(tok/s\):\s+([\d.]+)")
INSTANCE_RE = re.compile(r"Instance \[(\d+)\]")
TTFT_RE = re.compile(r"Mean TTFT \(ms\):\s+([\d.]+)")
TPOT_RE = re.compile(r"Mean TPOT \(ms\):\s+([\d.]+)")
ITL_RE = re.compile(r"Mean ITL \(ms\):\s+([\d.]+)")
HBF_WRITES_RE = re.compile(r"Total HBF writes \(all instances\):\s+(\d+)")
HBF_WRITE_MB_RE = re.compile(r"Total HBF bytes written \(all instances\):\s+([\d.]+)\s+MB")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

SCENARIO_SPECS = {
    "baseline": {
        "label": "No Spill",
        "flags": [],
        "color": "#4E5B70",
        "marker": "o",
    },
    "hbf_weights_only": {
        "label": "Weight Spill",
        "flags": ["--enable-hbf-offloading"],
        "color": "#D87928",
        "marker": "s",
    },
    "hbf_weights_kv": {
        "label": "Weight+KV Spill",
        "flags": ["--enable-hbf-offloading", "--enable-hbf-kv"],
        "color": "#1C8C73",
        "marker": "^",
    },
    "hbf_kv_only": {
        "label": "KV Spill Only",
        "flags": ["--enable-hbf-kv"],
        "color": "#8A5BD1",
        "marker": "D",
    },
}


class _PickledArrivalTimes:
    def __init__(self, name=None, arrival_times=None):
        self.name = name
        self.arrival_times = arrival_times or []


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def format_metric(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def parse_log_metrics(log_text: str) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "avg_generation_tput_tok_s": None,
        "total_tput_tok_s": None,
        "mean_ttft_ms": None,
        "mean_tpot_ms": None,
        "mean_itl_ms": None,
        "hbf_writes": 0,
        "hbf_write_mb": 0.0,
    }

    instance_metrics: dict[int, dict[str, float | None]] = {}
    current_instance: int | None = None

    for raw_line in log_text.splitlines():
        line = strip_ansi(raw_line.rstrip("\n"))

        avg_gen_match = AVG_GEN_TPUT_RE.search(line)
        if avg_gen_match:
            metrics["avg_generation_tput_tok_s"] = float(avg_gen_match.group(1))
            continue

        total_tput_match = TOTAL_TPUT_RE.search(line)
        if total_tput_match:
            metrics["total_tput_tok_s"] = float(total_tput_match.group(1))
            continue

        hbf_writes_match = HBF_WRITES_RE.search(line)
        if hbf_writes_match:
            metrics["hbf_writes"] = int(hbf_writes_match.group(1))
            continue

        hbf_write_mb_match = HBF_WRITE_MB_RE.search(line)
        if hbf_write_mb_match:
            metrics["hbf_write_mb"] = float(hbf_write_mb_match.group(1))
            continue

        instance_match = INSTANCE_RE.search(line)
        if instance_match:
            current_instance = int(instance_match.group(1))
            instance_metrics.setdefault(
                current_instance,
                {"ttft": None, "tpot": None, "itl": None},
            )
            continue

        if current_instance is None:
            continue

        ttft_match = TTFT_RE.search(line)
        if ttft_match:
            instance_metrics[current_instance]["ttft"] = float(ttft_match.group(1))
            continue

        tpot_match = TPOT_RE.search(line)
        if tpot_match:
            instance_metrics[current_instance]["tpot"] = float(tpot_match.group(1))
            continue

        itl_match = ITL_RE.search(line)
        if itl_match:
            instance_metrics[current_instance]["itl"] = float(itl_match.group(1))
            continue

    metrics["mean_ttft_ms"] = average(
        [values["ttft"] for values in instance_metrics.values()]
    )
    metrics["mean_tpot_ms"] = average(
        [values["tpot"] for values in instance_metrics.values()]
    )
    metrics["mean_itl_ms"] = average(
        [values["itl"] for values in instance_metrics.values()]
    )

    if metrics["avg_generation_tput_tok_s"] is None:
        raise ValueError("Could not find average generation throughput in run log")

    return metrics


def classify_failure(log_text: str) -> str:
    text = strip_ansi(log_text)
    lowered = text.lower()
    if (
        "runtimeerror" in lowered
        and (
            "tried to load" in lowered
            or "exceeds total" in lowered
            or "no requests remain" in lowered
            or "but only" in lowered
        )
    ):
        return "capacity_failure"
    if "oom" in lowered or "out of memory" in lowered:
        return "oom"
    return "error"


def slugify_float(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_window_spec(window: str | None) -> dict[str, float | int] | None:
    if window is None:
        return None
    parts = window.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            "--window must use START:END for request indices or tSTART:END for arrival-time range"
        )
    start_raw, end_raw = parts
    if start_raw.startswith("t"):
        return {
            "mode": "time",
            "start": float(start_raw[1:]),
            "end": float(end_raw),
        }
    return {
        "mode": "index",
        "start": int(start_raw),
        "end": int(end_raw),
    }


def _load_pickled_arrivals(arrival_path: Path) -> list[float]:
    class ArrivalUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "dataset.dataset" and name == "ArrivalTimes":
                return _PickledArrivalTimes
            return super().find_class(module, name)

    with arrival_path.open("rb") as f:
        obj = ArrivalUnpickler(f).load()
    arrival_times = getattr(obj, "arrival_times", None)
    if arrival_times is None:
        raise ValueError(f"Could not extract arrival_times from {arrival_path}")
    return [float(value) for value in arrival_times]


def _load_jsonl_arrivals(dataset_path: Path) -> list[float]:
    arrival_times = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            arrival_times.append(float(row["arrival_time_ns"]) / 1e9)
    return arrival_times


def _parse_dataset_arrivals(repo_root: Path, dataset_spec: str) -> list[float]:
    dataset_path = (repo_root / dataset_spec).resolve()
    if dataset_path.is_file():
        return _load_jsonl_arrivals(dataset_path)

    if ":" not in dataset_spec:
        raise ValueError(
            f"Cannot estimate request rate for dataset '{dataset_spec}'. Expected a jsonl path or ARRIVAL:LENGTH synthetic spec."
        )

    arrival_name, _ = dataset_spec.split(":", 1)
    arrival_path = repo_root / "dataset" / "raw" / f"{arrival_name}.arrival.pkl"
    if not arrival_path.is_file():
        raise FileNotFoundError(f"Arrival trace not found: {arrival_path}")
    return _load_pickled_arrivals(arrival_path)


def estimate_base_request_rate(repo_root: Path, dataset_spec: str, window: str | None) -> float | None:
    arrival_times = _parse_dataset_arrivals(repo_root, dataset_spec)
    if not arrival_times:
        return None

    window_spec = parse_window_spec(window)
    if window_spec is not None:
        if window_spec["mode"] == "index":
            start = int(window_spec["start"])
            end = int(window_spec["end"])
            arrival_times = arrival_times[start:end]
        else:
            start = float(window_spec["start"])
            end = float(window_spec["end"])
            arrival_times = [t for t in arrival_times if start <= t <= end]
            if len(arrival_times) >= 2:
                duration = max(end - start, 0.0)
                if duration > 0:
                    return len(arrival_times) / duration

    if len(arrival_times) < 2:
        return None
    span = arrival_times[-1] - arrival_times[0]
    if span <= 0:
        return None
    return (len(arrival_times) - 1) / span


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep HBF scenarios across load scales and plot throughput/latency/write tradeoffs."
    )
    parser.add_argument(
        "--cluster-config",
        default="cluster_config/single_node_single_instance_hbf.json",
        help="Cluster config passed to main.py",
    )
    parser.add_argument("--dataset", required=True, help="Dataset spec or jsonl path")
    parser.add_argument(
        "--window",
        default=None,
        help="Dataset window. Use 0:60 for request indices or t0:60 for seconds.",
    )
    parser.add_argument(
        "--load-scales",
        default=None,
        help="Comma-separated load scales to sweep.",
    )
    parser.add_argument(
        "--request-rates",
        default=None,
        help="Comma-separated offered request rates (req/s). Converted to load scales using the selected dataset/window.",
    )
    parser.add_argument(
        "--scenarios",
        default="baseline,hbf_weights_only,hbf_weights_kv",
        help="Comma-separated scenario names. Available: baseline,hbf_weights_only,hbf_weights_kv,hbf_kv_only",
    )
    parser.add_argument("--fp", type=int, default=16, help="Floating-point precision")
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="KV cache block size",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Optional max batch override",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=None,
        help="Optional max-num-batched-tokens override",
    )
    parser.add_argument(
        "--num-req",
        type=int,
        default=None,
        help="Optional request cap",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Throughput logging interval",
    )
    parser.add_argument(
        "--steady-min-s",
        type=float,
        default=None,
        help="Warmup duration before resetting counters and starting steady-state profiling.",
    )
    parser.add_argument(
        "--steady-window",
        type=float,
        default=None,
        help="Profiling window duration after --steady-min-s. main.py exits after this window.",
    )
    parser.add_argument(
        "--log-level",
        choices=["WARNING", "INFO", "DEBUG"],
        default="WARNING",
        help="Log verbosity passed to main.py",
    )
    parser.add_argument(
        "--network-backend",
        choices=["analytical", "ns3"],
        default="analytical",
        help="Network backend passed to main.py",
    )
    parser.add_argument(
        "--latency-metric",
        choices=["mean_ttft_ms", "mean_tpot_ms", "mean_itl_ms"],
        default="mean_tpot_ms",
        help="Latency metric used for the TPT-latency tradeoff panel.",
    )
    parser.add_argument(
        "--write-metric",
        choices=["hbf_writes", "hbf_write_mb"],
        default="hbf_writes",
        help="Write metric used for the TPT-write tradeoff panel.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/hbf_tradeoff_sweep",
        help="Directory for raw runs, summary.csv, and plots.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of experiments to run in parallel.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting and only write summary.csv.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run main.py",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra argument appended to every main.py invocation. Repeat as needed.",
    )
    return parser


def build_base_command(args: argparse.Namespace, repo_root: Path, load_scale: float) -> list[str]:
    command = [
        args.python,
        str(repo_root / "main.py"),
        "--cluster-config",
        args.cluster_config,
        "--fp",
        str(args.fp),
        "--block-size",
        str(args.block_size),
        "--dataset",
        args.dataset,
        "--load-scale",
        str(load_scale),
        "--log-interval",
        str(args.log_interval),
        "--log-level",
        args.log_level,
        "--network-backend",
        args.network_backend,
    ]

    if args.window is not None:
        command.extend(["--window", args.window])
    if args.max_batch is not None:
        command.extend(["--max-batch", str(args.max_batch)])
    if args.max_num_batched_tokens is not None:
        command.extend(
            ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
        )
    if args.num_req is not None:
        command.extend(["--num-req", str(args.num_req)])
    if args.steady_min_s is not None:
        command.extend(["--steady-min-s", str(args.steady_min_s)])
    if args.steady_window is not None:
        command.extend(["--steady-window", str(args.steady_window)])
    command.extend(args.extra_arg)
    return command


def plot_results(
    results: list[dict[str, object]],
    scenario_names: list[str],
    latency_metric: str,
    write_metric: str,
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    label_map = {
        "mean_ttft_ms": "Mean TTFT (ms)",
        "mean_tpot_ms": "Mean TPOT (ms)",
        "mean_itl_ms": "Mean ITL (ms)",
        "hbf_writes": "HBF Writes",
        "hbf_write_mb": "HBF Write Volume (MB)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), tight_layout=True)
    ax_rate, ax_latency, ax_write = axes

    use_req_rate_axis = all(row["offered_req_rps"] is not None for row in results)
    x_key = "offered_req_rps" if use_req_rate_axis else "load_scale"
    x_label = "Offered Request Rate (req/s)" if use_req_rate_axis else "Offered Load Scale"

    ax_rate.set_title("Supported Load Range")
    ax_rate.set_xlabel(x_label)
    ax_rate.set_ylabel("Average Generation Throughput (tok/s)")

    ax_latency.set_title("TPT vs Latency")
    ax_latency.set_xlabel(label_map[latency_metric])
    ax_latency.set_ylabel("Average Generation Throughput (tok/s)")

    ax_write.set_title("TPT vs HBF Writes")
    ax_write.set_xlabel(label_map[write_metric])
    ax_write.set_ylabel("Average Generation Throughput (tok/s)")

    ax_rate.set_ylim(bottom=0)
    ax_latency.set_ylim(bottom=0)
    ax_write.set_ylim(bottom=0)

    annotate_points = len({row["load_scale"] for row in results}) <= 8

    for scenario_name in scenario_names:
        spec = SCENARIO_SPECS[scenario_name]
        scenario_rows = [row for row in results if row["scenario"] == scenario_name]
        scenario_rows.sort(key=lambda row: float(row[x_key]))

        success_rows = [row for row in scenario_rows if row["status"] == "success"]
        failed_rows = [row for row in scenario_rows if row["status"] != "success"]

        if success_rows:
            xs_rate = [float(row[x_key]) for row in success_rows]
            ys_tput = [float(row["avg_generation_tput_tok_s"]) for row in success_rows]
            xs_latency = [float(row[latency_metric]) for row in success_rows]
            xs_write = [float(row[write_metric]) for row in success_rows]

            ax_rate.plot(
                xs_rate,
                ys_tput,
                color=spec["color"],
                marker=spec["marker"],
                linewidth=2,
                label=spec["label"],
            )
            ax_latency.plot(
                xs_latency,
                ys_tput,
                color=spec["color"],
                marker=spec["marker"],
                linewidth=2,
                label=spec["label"],
            )
            ax_write.plot(
                xs_write,
                ys_tput,
                color=spec["color"],
                marker=spec["marker"],
                linewidth=2,
                label=spec["label"],
            )

            if annotate_points:
                for row in success_rows:
                    if use_req_rate_axis:
                        text = f"{float(row['offered_req_rps']):.2f} rps"
                    else:
                        text = f"{float(row['load_scale']):g}x"
                    ax_latency.annotate(
                        text,
                        (float(row[latency_metric]), float(row["avg_generation_tput_tok_s"])),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                        color=spec["color"],
                    )
                    ax_write.annotate(
                        text,
                        (float(row[write_metric]), float(row["avg_generation_tput_tok_s"])),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                        color=spec["color"],
                    )

        if failed_rows:
            fail_x = [float(row[x_key]) for row in failed_rows]
            fail_y = [0.0 for _ in failed_rows]
            ax_rate.scatter(
                fail_x,
                fail_y,
                color=spec["color"],
                marker="x",
                s=60,
                zorder=5,
            )
            for row in failed_rows:
                ax_rate.annotate(
                    str(row["failure_kind"]),
                    (float(row[x_key]), 0.0),
                    textcoords="offset points",
                    xytext=(4, 8),
                    fontsize=8,
                    color=spec["color"],
                )

    handles, labels = ax_rate.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
        fig.subplots_adjust(top=0.83)

    tradeoff_path = output_dir / "tradeoff.png"
    fig.savefig(tradeoff_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    *,
    repo_root: Path,
    raw_dir: Path,
    args: argparse.Namespace,
    scenario_name: str,
    load_scale: float,
    offered_req_rps: float | None,
) -> dict[str, object]:
    spec = SCENARIO_SPECS[scenario_name]
    base_command = build_base_command(args, repo_root, load_scale)
    scale_slug = slugify_float(load_scale)
    scenario_csv = raw_dir / f"{scenario_name}_ls{scale_slug}.csv"
    scenario_log = raw_dir / f"{scenario_name}_ls{scale_slug}.log"
    command = [
        *base_command,
        "--output",
        str(scenario_csv.relative_to(repo_root)),
        *spec["flags"],
    ]

    header = f"[HBF Sweep] Running scenario={scenario_name} load_scale={load_scale:g}"
    if offered_req_rps is not None:
        header += f" offered_req_rps={offered_req_rps:.4f}"
    print(header)
    print(f"            {shlex.join(command)}")

    completed = subprocess.run(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "LLMSERVINGSIM_RUN_TAG": f"sweep_{scenario_name}_ls{scale_slug}_{uuid.uuid4().hex[:8]}",
        },
    )
    scenario_log.write_text(completed.stdout, encoding="utf-8")

    row: dict[str, object] = {
        "scenario": scenario_name,
        "scenario_label": spec["label"],
        "load_scale": load_scale,
        "offered_req_rps": offered_req_rps,
        "status": "success" if completed.returncode == 0 else "failed",
        "failure_kind": "",
        "supports_load": 1 if completed.returncode == 0 else 0,
        "avg_generation_tput_tok_s": None,
        "total_tput_tok_s": None,
        "mean_ttft_ms": None,
        "mean_tpot_ms": None,
        "mean_itl_ms": None,
        "hbf_writes": 0,
        "hbf_write_mb": 0.0,
        "log_file": str(scenario_log),
        "csv_file": str(scenario_csv),
    }

    if completed.returncode == 0:
        try:
            row.update(parse_log_metrics(completed.stdout))
        except Exception as exc:
            row["status"] = "failed"
            row["failure_kind"] = f"parse_error:{exc}"
            row["supports_load"] = 0
    else:
        row["failure_kind"] = classify_failure(completed.stdout)

    return row


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.request_rates and args.load_scales:
        raise ValueError("Use either --request-rates or --load-scales, not both.")

    base_request_rate = None
    sweep_points: list[tuple[float, float | None]] = []
    if args.request_rates:
        base_request_rate = estimate_base_request_rate(repo_root, args.dataset, args.window)
        if base_request_rate is None or base_request_rate <= 0:
            raise ValueError(
                "Could not estimate the base offered request rate for this dataset/window."
            )
        request_rates = parse_float_list(args.request_rates)
        sweep_points = [(rate / base_request_rate, rate) for rate in request_rates]
    else:
        load_scale_raw = args.load_scales if args.load_scales is not None else "0.5,1.0,1.5,2.0,2.5,3.0"
        load_scales = parse_float_list(load_scale_raw)
        base_request_rate = estimate_base_request_rate(repo_root, args.dataset, args.window)
        sweep_points = [
            (load_scale, base_request_rate * load_scale if base_request_rate is not None else None)
            for load_scale in load_scales
        ]

    scenario_names = [name.strip() for name in args.scenarios.split(",") if name.strip()]
    if not scenario_names:
        raise ValueError("Expected at least one scenario")
    unknown_scenarios = [name for name in scenario_names if name not in SCENARIO_SPECS]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown_scenarios)}")

    print(f"[HBF Sweep] Repo root: {repo_root}")
    print(f"[HBF Sweep] Output dir: {output_dir}")
    if args.request_rates:
        print(f"[HBF Sweep] Request rates: {', '.join(f'{req:g}' for _, req in sweep_points if req is not None)}")
    else:
        print(f"[HBF Sweep] Load scales: {', '.join(f'{value:g}' for value, _ in sweep_points)}")
    if base_request_rate is not None:
        print(f"[HBF Sweep] Base offered request rate at load_scale=1: {base_request_rate:.4f} req/s")
    print(f"[HBF Sweep] Scenarios: {', '.join(scenario_names)}")
    print(f"[HBF Sweep] Parallel jobs: {args.jobs}")
    if args.window and not args.window.startswith("t"):
        print(
            f"[HBF Sweep] Note: --window {args.window} is a request-index slice, not seconds. "
            f"Use tSTART:END for a time window."
        )

    jobs = max(1, args.jobs)
    task_specs = [
        {
            "repo_root": repo_root,
            "raw_dir": raw_dir,
            "args": args,
            "scenario_name": scenario_name,
            "load_scale": load_scale,
            "offered_req_rps": offered_req_rps,
        }
        for load_scale, offered_req_rps in sweep_points
        for scenario_name in scenario_names
    ]

    results: list[dict[str, object]] = []
    if jobs == 1:
        for spec in task_specs:
            results.append(run_experiment(**spec))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            future_to_spec = {
                executor.submit(run_experiment, **spec): spec for spec in task_specs
            }
            for future in concurrent.futures.as_completed(future_to_spec):
                results.append(future.result())

    results.sort(
        key=lambda row: (
            float(row["offered_req_rps"]) if row["offered_req_rps"] is not None else float(row["load_scale"]),
            scenario_names.index(str(row["scenario"])),
        )
    )

    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "scenario",
        "scenario_label",
        "load_scale",
        "offered_req_rps",
        "status",
        "failure_kind",
        "supports_load",
        "avg_generation_tput_tok_s",
        "total_tput_tok_s",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "hbf_writes",
        "hbf_write_mb",
        "log_file",
        "csv_file",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    if not args.no_plot:
        plot_results(results, scenario_names, args.latency_metric, args.write_metric, output_dir)

    print()
    print(
        "scenario".ljust(18),
        "rps".rjust(8),
        "scale".rjust(8),
        "status".rjust(10),
        "gen_tput".rjust(10),
        "lat".rjust(10),
        "writes".rjust(10),
    )
    print("-" * 76)
    for row in results:
        latency_value = row[args.latency_metric]
        writes_value = row[args.write_metric]
        print(
            str(row["scenario"]).ljust(18),
            format_metric(row["offered_req_rps"], digits=2).rjust(8),
            format_metric(float(row["load_scale"]), digits=2).rjust(8),
            str(row["status"]).rjust(10),
            format_metric(row["avg_generation_tput_tok_s"]).rjust(10),
            format_metric(latency_value).rjust(10),
            format_metric(writes_value).rjust(10),
        )

    print()
    print(f"[HBF Sweep] Wrote summary: {summary_path}")
    if not args.no_plot:
        print(f"[HBF Sweep] Wrote plot: {output_dir / 'tradeoff.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
