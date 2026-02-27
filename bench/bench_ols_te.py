from __future__ import annotations

from typing import Any

import numpy as np

from bench.common import env_info, format_row, parse_common_args, run_case, write_json
from te_net_lib.te import ols_te_matrix


def main() -> None:
    args = parse_common_args("Benchmark: te_net_lib.te.ols_te_matrix")

    sizes = [
        (50, 200),
        (100, 200),
        (100, 500),
        (200, 500),
    ]

    lag = 1
    add_intercept = True
    exclude_self = True

    results = []
    for N, T in sizes:
        rng = np.random.default_rng(args.seed + N * 10000 + T)
        R = rng.normal(size=(T, N)).astype(np.float64)

        def make_fn():
            def fn():
                ols_te_matrix(R, lag, add_intercept, exclude_self)

            return fn

        params: dict[str, Any] = {
            "N": N,
            "T": T,
            "lag": lag,
            "add_intercept": add_intercept,
            "exclude_self": exclude_self,
        }
        results.append(
            run_case("ols_te_matrix", params, make_fn, args.warmup, args.repeats)
        )

    print("Benchmark: ols_te_matrix")
    print(f"repeats={args.repeats} warmup={args.warmup} seed={args.seed}")
    for r in results:
        print(format_row(r.name, r.params, r.stats))

    payload = {
        "bench": "ols_te_matrix",
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
