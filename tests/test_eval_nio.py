import numpy as np

from te_net_lib.eval import compute_nio, hub_recovery_from_signal


def test_compute_nio_matches_definition():
    N = 5
    A = np.zeros((N, N), dtype=np.int8)
    A[0, 1] = 1
    A[0, 2] = 1
    A[3, 0] = 1
    out = compute_nio(A, True, True)
    outdeg = np.array([2, 0, 0, 1, 0], dtype=np.float64)
    indeg = np.array([1, 1, 1, 0, 0], dtype=np.float64)
    nio = (outdeg - indeg) / float(N - 1)
    assert np.allclose(out.out_strength, outdeg)
    assert np.allclose(out.in_strength, indeg)
    assert np.allclose(out.nio, nio)


def test_hub_recovery_basic():
    true_sig = np.array([0.0, 1.0, 5.0, 2.0, 3.0], dtype=np.float64)
    pred_sig = np.array([0.0, 2.0, 4.0, 1.0, 3.0], dtype=np.float64)
    r = hub_recovery_from_signal(true_sig, pred_sig, 2)
    assert abs(r - 1.0) < 1e-12


def test_hub_recovery_k0():
    true_sig = np.array([1.0, 2.0], dtype=np.float64)
    pred_sig = np.array([2.0, 1.0], dtype=np.float64)
    r = hub_recovery_from_signal(true_sig, pred_sig, 0)
    assert abs(r - 0.0) < 1e-12
