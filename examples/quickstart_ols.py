from __future__ import annotations

import numpy as np

from te_net_lib.dgp import simulate_planted_signal_var
from te_net_lib.graph import graph_density, precision_recall_f1, select_fixed_density
from te_net_lib.te import ols_te_matrix


def main() -> None:
    rng = np.random.default_rng(0)

    N = 20
    T = 300
    burnin = 50
    lag = 1

    dgp = simulate_planted_signal_var(
        rng=rng,
        N=N,
        T=T,
        edge_prob=0.10,
        coef_strength=0.40,
        noise_scale=1.00,
        burnin=burnin,
    )
    returns = dgp.returns
    true_adj = dgp.true_adj

    te = ols_te_matrix(
        returns=returns,
        lag=lag,
        add_intercept=True,
        exclude_self=True,
    )

    pred_adj = select_fixed_density(
        scores=te.beta,
        density=0.10,
        exclude_self=True,
        mode="abs",
    )

    d_true = graph_density(true_adj, True)
    d_pred = graph_density(pred_adj, True)
    prec, rec, f1 = precision_recall_f1(pred_adj, true_adj, True)

    print("OLS TE quickstart")
    print(f"N={N} T={T} lag={lag}")
    print(f"true_density={d_true:.4f} pred_density={d_pred:.4f}")
    print(f"precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")


if __name__ == "__main__":
    main()
