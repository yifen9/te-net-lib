import numpy as np

from te_net_lib.graph import select_fixed_density


def test_select_fixed_density_shapes_and_diag_excluded():
    N = 6
    g = np.random.default_rng(1)
    S = g.normal(size=(N, N)).astype(np.float64)
    A = select_fixed_density(S, 0.2, True, "abs")
    assert A.shape == (N, N)
    assert np.all(np.diag(A) == 0)


def test_select_fixed_density_density_close():
    N = 10
    g = np.random.default_rng(2)
    S = g.normal(size=(N, N)).astype(np.float64)
    d = 0.15
    A = select_fixed_density(S, d, True, "abs")
    m = N * (N - 1)
    k = int(np.floor(d * m + 1e-12))
    assert int(A.sum()) == k


def test_select_fixed_density_mode_pos_neg():
    N = 8
    S = np.zeros((N, N), dtype=np.float64)
    S[0, 1] = 10.0
    S[2, 3] = -9.0
    Apos = select_fixed_density(S, 0.02, True, "pos")
    Aneg = select_fixed_density(S, 0.02, True, "neg")
    assert Apos[0, 1] == 1
    assert Apos[2, 3] == 0
    assert Aneg[2, 3] == 1
    assert Aneg[0, 1] == 0


def test_select_fixed_density_all_zero_when_no_finite():
    N = 5
    S = np.full((N, N), -1.0, dtype=np.float64)
    A = select_fixed_density(S, 0.3, True, "pos")
    assert int(A.sum()) == 0
