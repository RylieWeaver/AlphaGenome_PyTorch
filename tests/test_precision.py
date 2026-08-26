from dataclasses import replace

import pytest
import torch

from alphagenome_pt import (
    HeadName,
    get_dtype_policy,
    small_alphagenome,
    synthetic_batch,
    synthetic_metadata,
)
from alphagenome_pt.precision import (
    DEEPMIND_DTYPE_POLICY,
    FLOAT32_DTYPE_POLICY,
    FLOAT64_DTYPE_POLICY,
    dot_with_dtype_policy,
    dtype_policy_context,
)


### HELPERS ###
def _dot_inputs():
    left = torch.linspace(-1, 1, 24).reshape(2, 3, 4)
    right = torch.linspace(-0.5, 0.5, 40).reshape(2, 4, 5)
    return left, right


### TESTS ###
def test_policy_rejects_compute_uptype_narrower_than_compute_dtype():
    with pytest.raises(ValueError, match="at least as precise"):
        replace(FLOAT64_DTYPE_POLICY, compute_uptype=torch.float32)


def test_float32_policy_overrides_outer_bfloat16_autocast():
    left, right = _dot_inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        with dtype_policy_context(FLOAT32_DTYPE_POLICY, "cpu"):
            actual = dot_with_dtype_policy(left, right)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, torch.bmm(left, right), rtol=0, atol=0)


def test_deepmind_policy_uses_bfloat16_operands_and_float32_output():
    left, right = _dot_inputs()

    with dtype_policy_context(DEEPMIND_DTYPE_POLICY, "cpu"):
        actual = dot_with_dtype_policy(left, right)
    expected = torch.bmm(left.bfloat16().float(), right.bfloat16().float())

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def test_float64_model_loss_uses_float64_policy():
    metadata = synthetic_metadata((HeadName.MASKED_LANGUAGE_MODELING,))
    model = small_alphagenome(
        metadata,
        max_seq_len=2_048,
        num_channels=16,
        transformer_layers=1,
        sync_bn=False,
        dtype_policy="float64",
    )
    batch = synthetic_batch(metadata, seq_len=model.max_seq_len)

    result = model.loss(batch)

    assert result.total.dtype == torch.float64
