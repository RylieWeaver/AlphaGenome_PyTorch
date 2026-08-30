# Pytest configuration for the correctness tests. Not adapted from an
# upstream source.

import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast, no model construction")
    config.addinivalue_line(
        "markers", "integration: builds a small model"
    )


def pytest_addoption(parser):
    parser.addoption("--atol", type=float, default=1e-4)
    parser.addoption("--rtol", type=float, default=1e-5)


@pytest.fixture(scope="session")
def tolerances(pytestconfig):
    return {
        "atol": pytestconfig.getoption("--atol"),
        "rtol": pytestconfig.getoption("--rtol"),
    }


# A model is built per test because metadata varies by head, so there is no
# shared model fixture. This only prevents seed state leaking between tests.
@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    yield
