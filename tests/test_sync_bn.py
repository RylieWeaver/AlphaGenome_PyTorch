# External
import pytest

# Internal
from alphagenome_pt import small_alphagenome
from alphagenome_pt.layers import BatchNorm


@pytest.mark.parametrize("sync_bn", [True, False])
def test_alphagenome_sync_bn_wires_through(sync_bn: bool):
    model = small_alphagenome(sync_bn=sync_bn)
    batch_norms = [
        module for module in model.modules() if isinstance(module, BatchNorm)
    ]

    assert batch_norms
    assert all(module.sync is sync_bn for module in batch_norms)
