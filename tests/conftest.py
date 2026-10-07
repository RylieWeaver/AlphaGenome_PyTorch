# External
import os
import sys
from pathlib import Path
import pytest
import torch

# Internal
from alphagenome_pt import HeadName, small_alphagenome, synthetic_metadata
from .deepmind_equivalence.precision import EQUIVALENCE_TEST_POLICIES

from .helpers import DNA_SEQUENCE


def pytest_exception_interact(node, report):
    """
    Let torchrun stop all peers when a parallelism test fails on one rank.
    
    NOTE: "pytest_exception_interact" is a recognized pytest hook name,
    which is how this is caught and activated.
    """
    if (
        node.path.name != "test_parallelism.py"
        or "TORCHELASTIC_RUN_ID" not in os.environ
        or not report.failed
    ):
        return

    # NOTE: Pytest normally catches failures and keeps the worker alive. We want
    # to exit before teardown because destroying process groups can hang while peers
    # wait in collectives. Torchrun observes this failed worker and stops the others.
    #
    # Related hard-exit patterns:
    # - pytest-timeout's timeout_timer: dump diagnostics, flush, then os._exit(1).
    #   https://github.com/pytest-dev/pytest-timeout/blob/2.3.1/pytest_timeout.py
    # - vLLM's subprocess-test helper discussion: print traceback, then os._exit(1).
    #   https://github.com/vllm-project/vllm/issues/7053
    # Torchrun stops the other workers when one fails:
    # https://docs.pytorch.org/docs/stable/elastic/run.html#failure-modes
    print(
        f"\n[rank {os.environ['RANK']}] {report.nodeid}\n{report.longreprtext}",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    os._exit(1)


def pytest_addoption(parser):
    group = parser.getgroup("equivalence")
    group.addoption(
        "--run-equivalence",
        action="store_true",
        help="Run the checkpoint-backed JAX/PyTorch equivalence test.",
    )
    group.addoption(
        "--equivalence-policy",
        choices=EQUIVALENCE_TEST_POLICIES,
        help="Run only one precision policy instead of all equivalence policies.",
    )
    group.addoption(
        "--checkpoint-equivalence-device",
        default="cpu",
        help="Device for checkpoint-backed tests, such as cpu, cuda, or cuda:1.",
    )
    group.addoption(
        "--equivalence-sequence-length",
        type=int,
        default=4096,
        help=(
            "Model input length in base pairs; lengths below 131072 skip only "
            "the checkpoint-backed loss comparison."
        ),
    )
    group.addoption(
        "--equivalence-report",
        default="tests/deepmind_equivalence/report.csv",
        help="CSV destination for JAX/PyTorch numerical difference metrics.",
    )
    group.addoption(
        "--equivalence-dna",
        nargs="+",
        type=Path,
        metavar="PATH",
        help=(
            "One DNA file reused for both organisms, or human and mouse DNA "
            "files in that order."
        ),
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    model = small_alphagenome(
        synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,)),
        max_seq_len=len(DNA_SEQUENCE),
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
    )
    model.eval()
    return model
