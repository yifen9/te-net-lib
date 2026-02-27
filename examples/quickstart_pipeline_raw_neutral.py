from __future__ import annotations

import numpy as np

from te_net_lib.dgp import simulate_garch_factor
from te_net_lib.eval import compute_nio
from te_net_lib.graph import select_fixed_density
from te_net_lib.te import ols_raw_and_neutral


def main() -> None:
    rng = np.random.default_rng(2)

    N = 30
    T = 400
    burnin = 50
    lag = 1

    dgp = simulate_garch_factor(
        rng=rng,
        N=N,
        T=T,
        k=3,
        omega=0.05,
        alpha=0.05,
        beta=0.90,
        loading_scale=0.30,
        burnin=burnin,
    )
    returns = dgp.returns

    out = ols_raw_and_neutral(
        returns=returns,
        lag=lag,
        add_intercept=True,
        exclude_self=True,
        n_components=3,
        center=True,
    )

    adj_raw = select_fixed_density(out.beta_raw, 0.10, True, "abs")
    adj_neu = select_fixed_density(out.beta_neutral, 0.10, True, "abs")

    nio_raw = compute_nio(adj_raw, True, True).nio
    nio_neu = compute_nio(adj_neu, True, True).nio

    print("Raw vs neutral pipeline (OLS)")
    print(f"N={N} T={T} lag={lag}")
    print(
        f"nio_raw_mean={float(nio_raw.mean()):.6f} nio_raw_std={float(nio_raw.std()):.6f}"
    )
    print(
        f"nio_neu_mean={float(nio_neu.mean()):.6f} nio_neu_std={float(nio_neu.std()):.6f}"
    )


if __name__ == "__main__":
    main()
