# External
import os
from pathlib import Path
import pandas as pd
import pytest

# Internal
from alphagenome_pt import get_dtype_policy
from .precision import (
    EQUIVALENCE_TEST_POLICIES,
    use_jax_compute_uptype_policy,
)
from .format_report import write_markdown_report
from .utils import difference_metrics, dtype_name

# Don't preallocate GPU memory for JAX so that PyTorch can use it
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# Don't keep JAX GPU memory after the forward so that PyTorch can use it
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
# Select deterministic cuBLAS workspace to have identical results
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


class PrecisionReport:
    IDENTITY_COLUMNS = ("test_name", "dtype_policy", "representation")

    METRIC_COLUMNS = (
        "relative_L2",
        "relative_Linf",
        "max_abs",
        "mean_abs",
        "reference_max_abs",
        "reference_mean_abs",
        "exact_fraction",
        "num_values",
        "pytorch_dtype",
        "jax_dtype",
        "dtype_match",
    )

    def __init__(self):
        self.rows = []

    def record(self, test_name, dtype_policy, representation, metrics):
        identity = dict(zip(
            self.IDENTITY_COLUMNS,
            (test_name, dtype_policy, representation),
            strict=True,
        ))
        self.rows.append({**identity, **metrics})

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = pd.DataFrame(self.rows)
        report = report.reindex(
            columns=[*self.IDENTITY_COLUMNS, *self.METRIC_COLUMNS]
        )
        report.to_csv(path, index=False)
        write_markdown_report(report, path.with_suffix(".md"))


@pytest.fixture
def jax_cpu_device():
    """Run checkpoint-free JAX work on CPU, matching PyTorch."""
    jax = pytest.importorskip("jax")
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.fixture(
    scope="session",
    params=EQUIVALENCE_TEST_POLICIES,
)
def pt_dtype_policy(request):
    """Provide the requested PyTorch dtype policies to equivalence tests."""
    selected_policy = request.config.getoption("--equivalence-policy")
    if selected_policy is not None and request.param != selected_policy:
        pytest.skip(f"equivalence invocation selected {selected_policy}")
    pt_dtype_policy = get_dtype_policy(request.param)
    if pt_dtype_policy.name == "float64":
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
    return pt_dtype_policy


@pytest.fixture
def jax_compute_uptype_policy(pt_dtype_policy):
    """Apply the selected compute uptype to fixed JAX operations."""
    with use_jax_compute_uptype_policy(pt_dtype_policy):
        yield


@pytest.fixture(scope="session")
def precision_report_session(pytestconfig):
    # Create report when first needed
    report = PrecisionReport()
    # Share it with all requesting tests
    yield report
    # Write it after those tests finish
    report.write(pytestconfig.getoption("--equivalence-report"))


@pytest.fixture
def record_and_assert_close(request, precision_report_session):
    comparison_index = 0

    def compare(
        pytorch_value,
        jax_value,
        *,
        dtype_policy,
        relative_L2_threshold,
        relative_Linf_threshold,
        name=None,
    ):
        nonlocal comparison_index
        comparison_index += 1
        representation = name or f"comparison_{comparison_index}"
        pytorch_dtype = dtype_name(pytorch_value)
        jax_dtype = dtype_name(jax_value)
        dtype_match = pytorch_dtype == jax_dtype
        metrics = difference_metrics(
            pytorch_value,
            jax_value,
            name=representation,
        )
        precision_report_session.record(
            request.node.nodeid,
            dtype_policy,
            representation,
            {
                **metrics,
                "pytorch_dtype": pytorch_dtype,
                "jax_dtype": jax_dtype,
                "dtype_match": dtype_match,
            },
        )

        failures = []
        if not dtype_match:
            failures.append(
                f"dtype mismatch: PyTorch {pytorch_dtype}, JAX {jax_dtype}"
            )
        for metric, threshold in (
            ("relative_L2", relative_L2_threshold),
            ("relative_Linf", relative_Linf_threshold),
        ):
            if metrics[metric] > threshold:
                failures.append(
                    f"{metric} {metrics[metric]:.6g} exceeds {threshold:.6g}"
                )
        assert not failures, f"{representation}: {'; '.join(failures)}"

    return compare
