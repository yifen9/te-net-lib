from __future__ import annotations

from typing import Any

import numpy as np

from bench.common import env_info, format_row, parse_common_args, run_case, write_json
from te_net_lib.preprocess import factor_neutralize_svd


def main() -> None:
    args = parse_common_args("Benchmark: te_net_lib.preprocess.factor_neutralize_svd")

    sizes = [
        (50, 200),
        (100, 200),
        (200, 500),
        (300, 800),
    ]
    ks = [0, 3, 5]

    center = True

    results = []
    for N, T in sizes:
        rng = np.random.default_rng(args.seed + N * 10000 + T)
        R = rng.normal(size=(T, N)).astype(np.float64)

        for k in ks:

            def make_fn():
                def fn():
                    factor_neutralize_svd(R, k, center)

                return fn

            params: dict[str, Any] = {
                "N": N,
                "T": T,
                "k": k,
                "center": center,
            }
            results.append(
                run_case(
                    "factor_neutralize_svd", params, make_fn, args.warmup, args.repeats
                )
            )

    print("Benchmark: factor_neutralize_svd")
    print(f"repeats={args.repeats} warmup={args.warmup} seed={args.seed}")
    for r in results:
        print(format_row(r.name, r.params, r.stats))

    payload = {
        "bench": "factor_neutralize_svd",
        "env": env_info(),
        "repeats": args.repeats,
        "warmup": args.warmup,
        "seed": args.seed,
        "results": [
            {"name": r.name, "params": r.params, "times_s": r.times_s, "stats": r.stats}
            for r in results
        ],
    }
    write_json(args.json, payload)


if __name__ == "__main__":
    main()
