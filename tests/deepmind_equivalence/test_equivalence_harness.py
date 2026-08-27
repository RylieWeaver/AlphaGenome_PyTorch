"""Tests for the JAX/PyTorch equivalence harness."""

# External
import numpy as np
import pytest
import torch

pytest.importorskip("jax")
pytest.importorskip("alphagenome_research")

import jax
import jax.numpy as jnp
import haiku as hk
from alphagenome_research.model import attention as jax_attention
from alphagenome_research.model import heads as jax_heads
from alphagenome_research.model import layers as jax_layers
from alphagenome_research.model import losses as jax_losses

pytestmark = pytest.mark.usefixtures("jax_cpu_device")

# Internal
from .precision import (
    jax_dot_algorithm,
    jax_dtype,
    jax_mixed_precision_policy,
    use_jax_compute_uptype_policy,
)
from .utils import difference_metrics, normal_values


NORMALIZATION_SHAPE = (2, 4, 8)
DOT_LEFT_SHAPE = (2, 16, 8)
DOT_RIGHT_SHAPE = (2, 8, 8)


def test_haiku_policy_uses_parameter_dtype_for_params_and_state(
    pt_dtype_policy,
):
    class Probe(hk.Module):
        def __call__(self):
            hk.get_parameter(
                "parameter", (), jnp.float32, init=jnp.ones
            )
            hk.get_state(
                "ema", (), jax_layers.jnp.float32, init=jnp.ones
            )

    transformed = hk.transform_with_state(lambda: Probe(name="probe")())
    with (
        hk.mixed_precision.push_policy(
            Probe, jax_mixed_precision_policy(pt_dtype_policy)
        ),
        use_jax_compute_uptype_policy(pt_dtype_policy),
    ):
        params, state = transformed.init(jax.random.PRNGKey(0))

    expected_dtype = jax_dtype(pt_dtype_policy.parameter_dtype)
    assert params["probe"]["parameter"].dtype == expected_dtype
    assert state["probe"]["ema"].dtype == expected_dtype


def test_policy_overrides_layer_norm_statistics_dtype(pt_dtype_policy):
    values = jnp.asarray(normal_values(NORMALIZATION_SHAPE, seed=2)).astype(
        jax_dtype(pt_dtype_policy.compute_dtype)
    )
    original_layers_jax_numpy = jax_layers.jnp
    original_attention_jax_numpy = jax_attention.jnp

    with use_jax_compute_uptype_policy(pt_dtype_policy):
        actual = jax_layers.jnp.mean(
            values,
            axis=-1,
            dtype=jax_layers.jnp.float32,
            keepdims=True,
        )

    expected = jnp.mean(
        values,
        axis=-1,
        dtype=jax_dtype(pt_dtype_policy.compute_uptype),
        keepdims=True,
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jax_dtype(
        pt_dtype_policy.compute_uptype
    )
    assert jax_layers.jnp is original_layers_jax_numpy
    assert jax_attention.jnp is original_attention_jax_numpy


def test_policy_overrides_preferred_element_type(pt_dtype_policy):
    dtype = jax_dtype(pt_dtype_policy.compute_dtype)
    left = jnp.asarray(normal_values(DOT_LEFT_SHAPE, seed=0), dtype=dtype)
    right = jnp.asarray(normal_values(DOT_RIGHT_SHAPE, seed=1), dtype=dtype)
    original_jax_numpy = jax_heads.jnp

    with use_jax_compute_uptype_policy(pt_dtype_policy):
        assert jax_heads.jnp.float32 == jax_dtype(
            pt_dtype_policy.compute_uptype
        )
        actual = jax_heads.jnp.einsum(
            "bqi,bik->bqk",
            left,
            right,
            preferred_element_type=jax_heads.jnp.float32,
        )

    expected = jnp.einsum(
        "bqi,bik->bqk",
        left,
        right,
        preferred_element_type=jax_dtype(pt_dtype_policy.compute_uptype),
    )
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == jax_dtype(pt_dtype_policy.compute_uptype)
    assert jax_heads.jnp is original_jax_numpy


def test_policy_replaces_core_float32_dtypes(pt_dtype_policy):
    expected = jax_dtype(pt_dtype_policy.compute_uptype)

    with use_jax_compute_uptype_policy(pt_dtype_policy):
        assert jax_attention.jnp.float32 == expected
        assert jax_heads.jnp.float32 == expected
        assert jax_layers.jnp.float32 == expected
        assert jax_losses.jnp.float32 == expected


def test_policy_replaces_ssu_float16_dtype(pt_dtype_policy):
    expected = (
        jnp.float16
        if pt_dtype_policy.compute_dtype == torch.bfloat16
        else jax_dtype(pt_dtype_policy.compute_dtype)
    )

    with use_jax_compute_uptype_policy(pt_dtype_policy):
        assert jax_heads.jnp.float16 == expected


def test_policy_overrides_reference_dot_algorithm(pt_dtype_policy):
    left = jnp.asarray(normal_values(DOT_LEFT_SHAPE, seed=0)).astype(
        jax_dtype(pt_dtype_policy.compute_dtype)
    )
    right = jnp.asarray(normal_values(DOT_RIGHT_SHAPE, seed=1)).astype(
        jax_dtype(pt_dtype_policy.compute_dtype)
    )
    original_jax_numpy = jax_attention.jnp

    with use_jax_compute_uptype_policy(pt_dtype_policy):
        actual = jax_attention.jnp.einsum(
            "bqi,bik->bqk",
            left,
            right,
            precision=jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
        )

    expected = jnp.einsum(
        "bqi,bik->bqk",
        left,
        right,
        precision=jax_dot_algorithm(pt_dtype_policy),
    )
    np.testing.assert_array_equal(actual, expected)
    assert jax_attention.jnp is original_jax_numpy
