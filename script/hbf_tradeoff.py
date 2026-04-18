#!/usr/bin/env python3
"""
Run a small HBF experiment matrix and summarize throughput/latency tradeoffs.

Scenarios:
  - baseline
  - hbf_kv_only
  - hbf_weights_only
  - hbf_weights_kv
"""

from __future__ import annotations

import argparse
import csv
import os
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


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def percent_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in (None, 0):
        return None
    return ((value - baseline) / baseline) * 100.0


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark HBF throughput/latency tradeoffs across four flag combinations."
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
        "--load-scale",
        type=float,
        default=1.0,
        help="Load scale passed to main.py",
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
        "--output-dir",
        default="output/hbf_tradeoff",
        help="Directory for scenario logs, per-request CSVs, and summary.csv",
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


def scenario_matrix() -> list[tuple[str, list[str]]]:
    return [
        ("baseline", []),
        ("hbf_kv_only", ["--enable-hbf-kv"]),
        ("hbf_weights_only", ["--enable-hbf-offloading"]),
        ("hbf_weights_kv", ["--enable-hbf-offloading", "--enable-hbf-kv"]),
    ]


def build_base_command(args: argparse.Namespace, repo_root: Path) -> list[str]:
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
        str(args.load_scale),
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


def format_metric(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HBF] Repo root: {repo_root}")
    print(f"[HBF] Output dir: {output_dir}")
    if args.window and not args.window.startswith("t"):
        print(
            f"[HBF] Note: --window {args.window} is a request-index slice, not seconds. "
            f"Use tSTART:END for a time window."
        )

    base_command = build_base_command(args, repo_root)
    results: list[dict[str, object]] = []

    for scenario_name, scenario_flags in scenario_matrix():
        scenario_csv = output_dir / f"{scenario_name}.csv"
        scenario_log = output_dir / f"{scenario_name}.log"
        command = [
            *base_command,
            "--output",
            str(scenario_csv.relative_to(repo_root)),
            *scenario_flags,
        ]

        print(f"[HBF] Running {scenario_name}")
        print(f"       {shlex.join(command)}")

        completed = subprocess.run(
            command,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "LLMSERVINGSIM_RUN_TAG": f"matrix_{scenario_name}_{uuid.uuid4().hex[:8]}",
            },
        )
        scenario_log.write_text(completed.stdout, encoding="utf-8")

        if completed.returncode != 0:
            print(
                f"[HBF] Scenario {scenario_name} failed. See {scenario_log}",
                file=sys.stderr,
            )
            return completed.returncode

        parsed = parse_log_metrics(completed.stdout)
        parsed["scenario"] = scenario_name
        parsed["log_file"] = str(scenario_log)
        parsed["csv_file"] = str(scenario_csv)
        results.append(parsed)

    baseline = next(result for result in results if result["scenario"] == "baseline")
    baseline_tput = baseline["avg_generation_tput_tok_s"]
    baseline_ttft = baseline["mean_ttft_ms"]
    baseline_tpot = baseline["mean_tpot_ms"]
    baseline_itl = baseline["mean_itl_ms"]

    for result in results:
        result["gen_tput_delta_pct"] = percent_delta(
            result["avg_generation_tput_tok_s"], baseline_tput  # type: ignore[arg-type]
        )
        result["ttft_delta_pct"] = percent_delta(
            result["mean_ttft_ms"], baseline_ttft  # type: ignore[arg-type]
        )
        result["tpot_delta_pct"] = percent_delta(
            result["mean_tpot_ms"], baseline_tpot  # type: ignore[arg-type]
        )
        result["itl_delta_pct"] = percent_delta(
            result["mean_itl_ms"], baseline_itl  # type: ignore[arg-type]
        )

    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "scenario",
        "avg_generation_tput_tok_s",
        "total_tput_tok_s",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_itl_ms",
        "hbf_writes",
        "hbf_write_mb",
        "gen_tput_delta_pct",
        "ttft_delta_pct",
        "tpot_delta_pct",
        "itl_delta_pct",
        "log_file",
        "csv_file",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(
        "scenario".ljust(18),
        "gen_tput".rjust(10),
        "ttft".rjust(10),
        "tpot".rjust(10),
        "itl".rjust(10),
        "hbf_writes".rjust(12),
        "hbf_mb".rjust(10),
    )
    print("-" * 82)
    for result in results:
        print(
            str(result["scenario"]).ljust(18),
            format_metric(result["avg_generation_tput_tok_s"]).rjust(10),
            format_metric(result["mean_ttft_ms"]).rjust(10),
            format_metric(result["mean_tpot_ms"]).rjust(10),
            format_metric(result["mean_itl_ms"]).rjust(10),
            format_metric(result["hbf_writes"], digits=0).rjust(12),
            format_metric(result["hbf_write_mb"]).rjust(10),
        )

    inactive_kv_scenarios = [
        str(result["scenario"])
        for result in results
        if "kv" in str(result["scenario"]) and result["hbf_writes"] == 0
    ]
    if inactive_kv_scenarios:
        print()
        print(
            "[HBF] KV offloading did not write to HBF in:",
            ", ".join(inactive_kv_scenarios),
        )
        print(
            "[HBF] If you expected KV spill, increase concurrency/load, use a longer time window, "
            "or reduce NPU memory in the cluster config."
        )

    print()
    print(f"[HBF] Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
