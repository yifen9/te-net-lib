from __future__ import annotations

from typing import Any

import numpy as np

from bench.common import env_info, format_row, parse_common_args, run_case, write_json
from te_net_lib.te import lasso_te_path


def main() -> None:
    args = parse_common_args("Benchmark: te_net_lib.te.lasso_te_path")

    sizes = [
        (50, 200),
        (100, 200),
        (100, 500),
    ]
    alpha_grids = [
        [0.2, 0.1, 0.05, 0.02, 0.0],
        [0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
        [0.3, 0.25, 0.2, 0.15, 0.1, 0.08, 0.06, 0.04, 0.02, 0.0],
    ]

    lag = 1
    max_iter = 500
    tol = 1e-8
    add_intercept = True
    standardize = True
    exclude_self = True

    results = []
    for N, T in sizes:
        rng = np.random.default_rng(args.seed + N * 10000 + T)
        R = rng.normal(size=(T, N)).astype(np.float64)

        for grid in alpha_grids:
            grid_len = len(grid)

            def make_fn():
                def fn():
                    lasso_te_path(
                        R,
                        lag,
                        grid,
                        max_iter,
                        tol,
                        add_intercept,
                        standardize,
                        exclude_self,
                    )

                return fn

            params: dict[str, Any] = {
                "N": N,
                "T": T,
                "lag": lag,
                "K": grid_len,
                "max_iter": max_iter,
                "tol": tol,
                "add_intercept": add_intercept,
                "standardize": standardize,
                "exclude_self": exclude_self,
            }
            results.append(
                run_case("lasso_te_path", params, make_fn, args.warmup, args.repeats)
            )

    print("Benchmark: lasso_te_path")
    print(f"repeats={args.repeats} warmup={args.warmup} seed={args.seed}")
    for r in results:
        print(format_row(r.name, r.params, r.stats))

    payload = {
        "bench": "lasso_te_path",
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
