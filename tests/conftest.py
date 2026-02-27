import pytest

from te_net_lib.rng import rng


@pytest.fixture
def N():
    return 8


@pytest.fixture
def T():
    return 64


@pytest.fixture
def burnin():
    return 10


@pytest.fixture
def g1():
    return rng(123)


@pytest.fixture
def g2():
    return rng(123)
