from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True, slots=True)
class BenchResult:
    name: str
    params: dict[str, Any]
    times_s: list[float]
    stats: dict[str, float]


def parse_common_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--repeats",
        type=int,
        required=True,
        help="Number of timed repetitions per case.",
    )
    p.add_argument(
        "--warmup", type=int, required=True, help="Number of warmup runs per case."
    )
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Seed used for synthetic input generation.",
    )
    p.add_argument(
        "--json",
        type=str,
        required=False,
        default="",
        help="Optional path to write JSON results.",
    )
    return p.parse_args()


def time_call(fn: Callable[[], Any]) -> float:
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter()
    return float(t1 - t0)


def summarize(times_s: list[float]) -> dict[str, float]:
    a = np.array(times_s, dtype=np.float64)
    return {
        "min_s": float(a.min()) if a.size else 0.0,
        "median_s": float(np.median(a)) if a.size else 0.0,
        "mean_s": float(a.mean()) if a.size else 0.0,
        "p90_s": float(np.quantile(a, 0.9)) if a.size else 0.0,
        "max_s": float(a.max()) if a.size else 0.0,
    }


def run_case(
    name: str,
    params: dict[str, Any],
    make_fn: Callable[[], Callable[[], Any]],
    warmup: int,
    repeats: int,
) -> BenchResult:
    fn = make_fn()
    for _ in range(warmup):
        fn()
    times: list[float] = []
    for _ in range(repeats):
        times.append(time_call(fn))
    return BenchResult(name=name, params=params, times_s=times, stats=summarize(times))


def format_row(name: str, params: dict[str, Any], stats: dict[str, float]) -> str:
    ps = " ".join([f"{k}={params[k]}" for k in sorted(params.keys())])
    return (
        f"{name:<22} {ps:<40} "
        f"min={stats['min_s']:.6f}s "
        f"med={stats['median_s']:.6f}s "
        f"mean={stats['mean_s']:.6f}s "
        f"p90={stats['p90_s']:.6f}s"
    )


def env_info() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", ""),
    }


def write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
