from dataclasses import replace

import pytest
import torch

from alphagenome_pt import (
    HeadName,
    Metadata,
    get_dtype_policy,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)
from alphagenome_pt.precision import FLOAT64_DTYPE_POLICY, dtype_policy_context
from alphagenome_pt.checkpoint import load_deepmind_state
from alphagenome_pt.serialization import _load_metadata, _save_metadata, _save_state_dict


### TESTS ###
def test_policy_rejects_compute_uptype_narrower_than_compute_dtype():
    with pytest.raises(ValueError, match="at least as precise"):
        replace(FLOAT64_DTYPE_POLICY, compute_uptype=torch.float32)


def test_checkpoint_assignment_restores_parameter_dtype(tmp_path):
    source = small_alphagenome(dtype_policy="float32")
    checkpoint_path = tmp_path / "state.safetensors"
    _save_state_dict(source.state_dict(), checkpoint_path)

    target = small_alphagenome(dtype_policy="float64")
    load_deepmind_state(
        target,
        local_dir=tmp_path,
        local_filename=checkpoint_path.name,
        assign=True,
    )

    floating_state = [
        value
        for value in target.state_dict().values()
        if value.is_floating_point()
    ]
    assert floating_state
    assert all(value.dtype == torch.float64 for value in floating_state)


@pytest.mark.parametrize("policy_name", ("deepmind", "bfloat16", "float32", "float64"))
def test_json_metadata_preserves_track_means_precision(tmp_path, policy_name):
    metadata = synthetic_metadata((HeadName.RNA_SEQ,), organisms=("human",))
    head_metadata = metadata.metadata["heads"]["rna_seq"]
    # This value rounds to 1.0 if the JSON list passes through FP32.
    expected = torch.full_like(head_metadata["means"], 1.00000001, dtype=torch.float64)
    head_metadata["means"] = expected
    path = tmp_path / "metadata.json"
    _save_metadata(metadata.metadata, path)

    policy = get_dtype_policy(policy_name)
    with dtype_policy_context(policy, "cpu"):
        model = small_alphagenome(
            Metadata(_load_metadata(path)),
            max_seq_len=2_048,
            num_channels=16,
            transformer_layers=1,
            dtype_policy=policy_name,
        )

    torch.testing.assert_close(
        model._heads["rna_seq"]._track_means,
        expected.to(policy.parameter_dtype),
        rtol=0, atol=0,
    )


@pytest.mark.parametrize(
    ("policy_name", "caller_dtype"),
    (
        ("deepmind", torch.float64),
        ("float32", torch.bfloat16),
        ("float64", torch.float32),
    ),
)
def test_model_policy_normalizes_input_and_output_dtypes(
    policy_name, caller_dtype
):
    policy = get_dtype_policy(policy_name)
    model = small_alphagenome(
        synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,)),
        max_seq_len=2_048,
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
        dtype_policy=policy_name,
    )
    model.eval()
    encoder_input_dtypes = []
    handle = model.sequence_encoder.register_forward_pre_hook(
        lambda _module, inputs: encoder_input_dtypes.append(inputs[0].dtype)
    )

    try:
        embeddings = model.embed(
            torch.zeros(1, 2_048, 4, dtype=caller_dtype)
        )
    finally:
        handle.remove()

    assert encoder_input_dtypes == [policy.input_dtype]
    assert all(
        value.dtype == policy.output_dtype
        for value in (
            embeddings.embeddings_1bp,
            embeddings.embeddings_128bp,
            embeddings.embeddings_pair,
        )
    )


@pytest.mark.parametrize("policy_name", ("deepmind", "float32", "float64"))
def test_model_loss_returns_output_dtype(policy_name):
    policy = get_dtype_policy(policy_name)
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(
        metadata,
        max_seq_len=2_048,
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
        dtype_policy=policy_name,
    )
    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    result = model.loss(batch)

    assert result.total.dtype == policy.output_dtype
    assert all(
        leaf.numerator.dtype == policy.output_dtype
        and leaf.denominator.dtype == policy.output_dtype
        and leaf.value.dtype == policy.output_dtype
        for _, leaf in result.tree.iter_leaves()
    )
