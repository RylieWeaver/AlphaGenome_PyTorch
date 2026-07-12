# Shim. pytest only discovers hooks and fixtures from a file named conftest.py,
# so the real configuration lives in _config.py and is re-exported here.

from ._config import (  # noqa: F401
    _seed,
    pytest_addoption,
    pytest_configure,
    tolerances,
)
