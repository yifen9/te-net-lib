from __future__ import annotations

from typing import Any

import numpy as np

from bench.common import env_info, format_row, parse_common_args, run_case, write_json
from te_net_lib.graph import select_fixed_density


def main() -> None:
    args = parse_common_args("Benchmark: te_net_lib.graph.select_fixed_density")

    Ns = [200, 500, 1000]
    density = 0.05
    exclude_self = True
    mode = "abs"

    results = []
    for N in Ns:
        rng = np.random.default_rng(args.seed + N * 10000)
        S = rng.normal(size=(N, N)).astype(np.float64)

        def make_fn():
            def fn():
                select_fixed_density(S, density, exclude_self, mode)

            return fn

        params: dict[str, Any] = {
            "N": N,
            "density": density,
            "exclude_self": exclude_self,
            "mode": mode,
        }
        results.append(
            run_case("select_fixed_density", params, make_fn, args.warmup, args.repeats)
        )

    print("Benchmark: select_fixed_density")
    print(f"repeats={args.repeats} warmup={args.warmup} seed={args.seed}")
    for r in results:
        print(format_row(r.name, r.params, r.stats))

    payload = {
        "bench": "select_fixed_density",
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
