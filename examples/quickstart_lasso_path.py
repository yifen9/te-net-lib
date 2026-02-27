from __future__ import annotations

import numpy as np

from te_net_lib.dgp import simulate_planted_signal_var
from te_net_lib.graph import precision_recall_f1, select_fixed_density
from te_net_lib.te import lasso_te_path


def main() -> None:
    rng = np.random.default_rng(3)

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

    alphas = [0.20, 0.10, 0.05, 0.02, 0.00]
    path = lasso_te_path(
        returns=returns,
        lag=lag,
        alphas=alphas,
        max_iter=800,
        tol=1e-8,
        add_intercept=True,
        standardize=True,
        exclude_self=True,
    )

    density = 0.10
    print("Lasso TE path quickstart")
    print(f"N={N} T={T} lag={lag} density={density}")
    for i, a in enumerate(path.alphas.tolist()):
        adj = select_fixed_density(path.betas[i], density, True, "abs")
        prec, rec, f1 = precision_recall_f1(adj, true_adj, True)
        nnz = int(np.count_nonzero(path.betas[i]))
        print(
            f"alpha={a:.6f} nnz_beta={nnz} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}"
        )


if __name__ == "__main__":
    main()
