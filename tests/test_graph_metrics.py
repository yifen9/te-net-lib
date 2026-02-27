import numpy as np

from te_net_lib.graph import (
    confusion_counts,
    graph_density,
    hub_indices,
    in_out_degree,
    precision_recall_f1,
)


def test_graph_density_exclude_self():
    N = 4
    A = np.zeros((N, N), dtype=np.int8)
    A[0, 1] = 1
    A[1, 2] = 1
    d = graph_density(A, True)
    assert abs(d - (2.0 / (N * (N - 1)))) < 1e-12


def test_in_out_degree():
    N = 5
    A = np.zeros((N, N), dtype=np.int8)
    A[0, 1] = 1
    A[0, 2] = 1
    A[3, 2] = 1
    indeg, outdeg = in_out_degree(A, True)
    assert int(outdeg[0]) == 2
    assert int(indeg[2]) == 2
    assert int(indeg[1]) == 1
    assert int(outdeg[3]) == 1


def test_hub_indices():
    N = 6
    A = np.zeros((N, N), dtype=np.int8)
    A[0, 1] = 1
    A[0, 2] = 1
    A[0, 3] = 1
    A[4, 2] = 1
    top = hub_indices(A, 2, "out", True)
    assert top.shape == (2,)
    assert int(top[0]) == 0


def test_precision_recall_f1():
    N = 4
    T = np.zeros((N, N), dtype=np.int8)
    P = np.zeros((N, N), dtype=np.int8)
    T[0, 1] = 1
    T[1, 2] = 1
    P[0, 1] = 1
    P[2, 3] = 1
    prec, rec, f1 = precision_recall_f1(P, T, True)
    assert abs(prec - 0.5) < 1e-12
    assert abs(rec - 0.5) < 1e-12
    assert abs(f1 - 0.5) < 1e-12
    tp, fp, fn, tn = confusion_counts(P, T, True)
    assert (tp, fp, fn) == (1, 1, 1)
