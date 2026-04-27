import pytest

from alphagenome_pt import AlphaGenome, AlphaGenomeConfig
from alphagenome_pt.utils import RMSBatchNorm

from .helpers import build_metadata


def make_cfg(sync_bn: bool) -> AlphaGenomeConfig:
    metadata = build_metadata({"masked_language_modeling": {}}, organisms=("human",))
    return AlphaGenomeConfig(
        max_seq_len=2048,
        num_channels=64,
        transformer_layers=1,
        num_q_heads=4,
        num_kv_heads=1,
        qk_head_dim=16,
        v_head_dim=16,
        pair_channels=16,
        pair_heads=8,
        sync_bn=sync_bn,
        metadata=metadata,
    )


@pytest.mark.parametrize("sync_bn", [True, False])
def test_alphagenome_sync_bn_wires_through(sync_bn: bool):
    model = AlphaGenome(make_cfg(sync_bn))
    batch_norms = [
        module for module in model.modules() if isinstance(module, RMSBatchNorm)
    ]

    assert batch_norms
    assert all(module.sync is sync_bn for module in batch_norms)
