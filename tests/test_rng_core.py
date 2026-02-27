import numpy as np
import pytest

from te_net_lib.rng import RngScope, derive_rng, rng, seed_from_fields


def test_seed_from_fields_stable_order_invariant():
    person = b"te_net_lib"
    digest_size = 8
    a = {"N": 200, "T": 400, "trial": 3, "nested": {"x": 1, "y": [2, 3]}}
    b = {"trial": 3, "T": 400, "nested": {"y": [2, 3], "x": 1}, "N": 200}
    assert seed_from_fields(a, digest_size, person) == seed_from_fields(
        b, digest_size, person
    )


def test_seed_from_fields_changes_when_value_changes():
    person = b"te_net_lib"
    digest_size = 8
    a = {"N": 200, "T": 400, "trial": 3}
    b = {"N": 201, "T": 400, "trial": 3}
    assert seed_from_fields(a, digest_size, person) != seed_from_fields(
        b, digest_size, person
    )


def test_seed_from_fields_rejects_long_person():
    digest_size = 8
    person = b"this_is_way_too_long_person"
    with pytest.raises(ValueError):
        seed_from_fields({"x": 1}, digest_size, person)


def test_rng_accepts_int_and_is_reproducible():
    g1 = rng(123)
    g2 = rng(123)
    x1 = g1.normal(size=10)
    x2 = g2.normal(size=10)
    assert np.allclose(x1, x2)


def test_rng_rejects_negative_seed():
    with pytest.raises(ValueError):
        rng(-1)


def test_derive_rng_distinct_for_distinct_labels():
    base = rng(123)
    person = b"te_net_lib"
    digest_size = 8
    spawn_key = ()
    a = derive_rng(base, ["a"], digest_size, person, spawn_key).normal(size=10)
    b = derive_rng(base, ["b"], digest_size, person, spawn_key).normal(size=10)
    assert not np.allclose(a, b)


def test_derive_rng_reproducible_for_same_label():
    base1 = rng(123)
    base2 = rng(123)
    person = b"te_net_lib"
    digest_size = 8
    spawn_key = ()
    a1 = derive_rng(base1, ["same"], digest_size, person, spawn_key).normal(size=10)
    a2 = derive_rng(base2, ["same"], digest_size, person, spawn_key).normal(size=10)
    assert np.allclose(a1, a2)


def test_rng_scope_child_consistency():
    person = b"te_net_lib"
    digest_size = 8
    spawn_key = ()
    s1 = RngScope.from_seed(123, digest_size, person, spawn_key)
    s2 = RngScope.from_seed(123, digest_size, person, spawn_key)
    x1 = s1.child(["task", "subtask"]).uniform(size=10)
    x2 = s2.child(["task", "subtask"]).uniform(size=10)
    assert np.allclose(x1, x2)
