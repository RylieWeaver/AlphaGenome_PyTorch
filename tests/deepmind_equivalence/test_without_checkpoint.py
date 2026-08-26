"""JAX/PyTorch parity tests that do not require published checkpoints.

These comparisons intentionally run on CPU with small model/data sizes.
"""

# External
from dataclasses import replace
import numpy as np
import pytest
import torch

pytest.importorskip("jax")
pytest.importorskip("haiku")
pytest.importorskip("alphagenome_research")

import haiku as hk
import jax
import jax.numpy as jnp
from alphagenome_research.model import attention as jax_attention
from alphagenome_research.model import convolutions as jax_convolutions
from alphagenome_research.model import heads as jax_heads
from alphagenome_research.model import layers as jax_layers
from alphagenome_research.model import losses as jax_losses
from alphagenome_research.model import splicing as jax_splicing

pytestmark = pytest.mark.usefixtures("jax_cpu_device")

# Internal
from alphagenome_pt import attention as torch_attention
from alphagenome_pt import convolutions as torch_convolutions
from alphagenome_pt import heads as torch_heads
from alphagenome_pt import layers as torch_layers
from alphagenome_pt import losses as torch_losses
from alphagenome_pt import splicing as torch_splicing
from alphagenome_pt.precision import (
    dot_with_dtype_policy,
    dtype_policy_context,
)
from .precision import (
    equivalence_criteria,
    jax_dot_algorithm,
    jax_dtype,
    jax_mixed_precision_policy,
    use_jax_compute_uptype_policy,
)
from .utils import linspace_values, normal_values


### HELPERS ###
BATCH_SIZE = 2
SEQUENCE_LENGTH = 32
CHANNELS = 8
NUM_ORGANISMS = 2
NUM_HEADS = 3
NUM_TRACKS = 3
NUM_CLASSES = 4
NUM_SPLICE_SITE_CLASSES = 5

SEQUENCE_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, CHANNELS)
ATTENTION_PROJECTION_SHAPE = (BATCH_SIZE, CHANNELS, CHANNELS)
TRACK_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, NUM_TRACKS)
TRACK_MASK_SHAPE = (BATCH_SIZE, 1, NUM_TRACKS)
PAIR_SHAPE = (BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, SEQUENCE_LENGTH)
ROTARY_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, NUM_HEADS, CHANNELS)
CLASSIFICATION_SHAPE = (BATCH_SIZE, SEQUENCE_LENGTH, NUM_CLASSES)
SPLICE_SITE_SHAPE = (
    BATCH_SIZE, SEQUENCE_LENGTH, NUM_SPLICE_SITE_CLASSES
)
RELATIVE_DIAGONAL_SHAPE = (
    BATCH_SIZE, NUM_HEADS, SEQUENCE_LENGTH, 2 * SEQUENCE_LENGTH
)

WEIGHT_AND_BIAS_BOUND = 1.0 / np.sqrt(CHANNELS)
WEIGHT_AND_BIAS_RANGE = (-WEIGHT_AND_BIAS_BOUND, WEIGHT_AND_BIAS_BOUND)
MULTIPLICATIVE_SCALE_RANGE = (0.5, 1.5)

TRACK_SCALING_VALUE_RANGE = (0.5, 20.0)
TRACK_MEAN_RANGE = (0.5, 2.0)
MSE_TARGET_RANGE = (-1.0, 1.0)
MSE_PREDICTION_RANGE = (-1.5, 1.5)
# NOTE: Counts are expected to be non-negative and can be large
COUNT_TARGET_RANGE = (0.0, 1000.0)
POSITIVE_PREDICTION_RANGE = (1.0, 900.0)
LOGIT_RANGE = (-3.0, 3.0)
PROBABILITY_TARGET_RANGE = (0.0, 1.0)


def _torch_array(
    values: np.ndarray,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.as_tensor(values, dtype=dtype)


def _jax_array(
    values: np.ndarray,
    dtype: torch.dtype,
) -> jax.Array:
    return jnp.asarray(values, dtype=jax_dtype(dtype))


# NOTE: Haiku must run the transformed function once to initialize its parameters.
def _initialize_haiku(
    function, inputs, module_type, pt_dtype_policy, output_dtype
):
    module_pt_dtype_policy = replace(
        pt_dtype_policy, output_dtype=output_dtype
    )
    jax_dtype_policy = jax_mixed_precision_policy(module_pt_dtype_policy)

    def policy_function(values):
        with (
            hk.mixed_precision.push_policy(
                module_type, jax_dtype_policy
            ),
            use_jax_compute_uptype_policy(pt_dtype_policy),
        ):
            return function(values)

    transformed = hk.without_apply_rng(hk.transform(policy_function))
    params = transformed.init(
        jax.random.PRNGKey(0),
        _jax_array(inputs, pt_dtype_policy.compute_dtype),
    )
    return transformed, hk.data_structures.to_mutable_dict(params)


##### TESTS #####

### SHARED MODULES ###
# NOTE: gelu is used for multiple representation types but we use the sequence shape here
def test_gelu_activations(pt_dtype_policy, record_and_assert_close):
    values = linspace_values(SEQUENCE_SHAPE, -4, 4)
    torch_values = _torch_array(values, pt_dtype_policy.compute_dtype)
    jax_values = _jax_array(values, pt_dtype_policy.compute_dtype)
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_1702 = torch_layers.GELU_1702()(torch_values)
        torch_tanh = torch.nn.GELU(approximate="tanh")(torch_values)
    jax_1702 = jax_layers.gelu(jax_values)
    jax_tanh = jax.nn.gelu(jax_values)

    record_and_assert_close(
        torch_1702,
        jax_1702,
        name="gelu_1702",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )
    record_and_assert_close(
        torch_tanh,
        jax_tanh,
        name="gelu_tanh",
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


@pytest.mark.parametrize("rms_norm", [False, True])
def test_layer_norm(
    rms_norm, pt_dtype_policy, record_and_assert_close
):
    values = normal_values(SEQUENCE_SHAPE)
    transformed, params = _initialize_haiku(
        lambda x: jax_layers.LayerNorm(rms_norm=rms_norm)(x),
        values,
        jax_layers.LayerNorm,
        pt_dtype_policy,
        pt_dtype_policy.compute_dtype,
    )
    scale = linspace_values((CHANNELS,), *MULTIPLICATIVE_SCALE_RANGE)
    offset = linspace_values((CHANNELS,), *WEIGHT_AND_BIAS_RANGE)
    params["layer_norm"].update(
        scale=_jax_array(scale, pt_dtype_policy.parameter_dtype),
        offset=_jax_array(offset, pt_dtype_policy.parameter_dtype),
    )

    torch_layer = torch_layers.LayerNorm(CHANNELS, rms_norm=rms_norm).to(
        pt_dtype_policy.parameter_dtype
    )
    with torch.no_grad():
        torch_layer.scale.copy_(
            _torch_array(scale, pt_dtype_policy.parameter_dtype)
        )
        torch_layer.offset.copy_(
            _torch_array(offset, pt_dtype_policy.parameter_dtype)
        )
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_layer(
            _torch_array(values, pt_dtype_policy.compute_dtype)
        )
    jax_output = transformed.apply(
        params, _jax_array(values, pt_dtype_policy.compute_dtype)
    )

    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### ENCODER / DECODER ###
def test_standardized_convolution(
    pt_dtype_policy, record_and_assert_close
):
    values = normal_values(SEQUENCE_SHAPE)
    transformed, params = _initialize_haiku(
        lambda x: jax_convolutions.StandardizedConv1D(CHANNELS, 3)(x),
        values,
        jax_convolutions.StandardizedConv1D,
        pt_dtype_policy,
        pt_dtype_policy.compute_dtype,
    )
    weights = linspace_values(
        (3, CHANNELS, CHANNELS), *WEIGHT_AND_BIAS_RANGE
    )
    scale = linspace_values((CHANNELS,), *MULTIPLICATIVE_SCALE_RANGE)
    bias = linspace_values((CHANNELS,), *WEIGHT_AND_BIAS_RANGE)
    params["standardized_conv1_d"].update(
        w=_jax_array(weights, pt_dtype_policy.parameter_dtype),
        scale=_jax_array(
            scale[None, None], pt_dtype_policy.parameter_dtype
        ),
        bias=_jax_array(bias, pt_dtype_policy.parameter_dtype),
    )

    torch_layer = torch_convolutions.StandardizedConv1d(CHANNELS, CHANNELS, 3).to(
        pt_dtype_policy.parameter_dtype
    )
    with torch.no_grad():
        torch_layer.weight.copy_(
            _torch_array(
                weights.transpose(2, 1, 0),
                pt_dtype_policy.parameter_dtype,
            )
        )
        torch_layer.scale.copy_(
            _torch_array(
                scale[:, None, None],
                pt_dtype_policy.parameter_dtype,
            )
        )
        torch_layer.bias.copy_(
            _torch_array(bias, pt_dtype_policy.parameter_dtype)
        )
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_layer(
            _torch_array(values, pt_dtype_policy.compute_dtype).permute(0, 2, 1)
        )
    torch_output = torch_output.permute(0, 2, 1)
    jax_output = transformed.apply(
        params, _jax_array(values, pt_dtype_policy.compute_dtype)
    )

    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### TRANSFORMER ###
def test_rotary_embedding(pt_dtype_policy, record_and_assert_close):
    values = normal_values(ROTARY_SHAPE)
    positions = np.tile(
        np.arange(SEQUENCE_LENGTH), (BATCH_SIZE, 1)
    )
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_attention.apply_rope(
            _torch_array(values, pt_dtype_policy.compute_dtype),
            torch.from_numpy(positions),
            max_position=8192,
        )
    jax_output = jax_attention.apply_rope(
        _jax_array(values, pt_dtype_policy.compute_dtype),
        jnp.asarray(positions),
        max_position=8192,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### PAIR ###
# NOTE: No need to test multiple precisions because the resulting representation
# is always in {1, 0, -1} (identical across precisions).
def test_relative_position_features(record_and_assert_close):
    device = torch.device("cpu")
    positions = np.arange(
        -(SEQUENCE_LENGTH // 2), SEQUENCE_LENGTH // 2, dtype=np.float32
    )
    unsigned = jax_attention._central_mask_features(
        distances=jnp.abs(positions), feature_size=CHANNELS // 2, seq_length=512
    )
    jax_features = jnp.concatenate(
        [unsigned, jnp.sign(positions)[..., None] * unsigned], axis=-1
    )
    torch_features = torch_attention.central_mask_features(
        torch.from_numpy(positions).to(device),
        feature_size=CHANNELS,
        max_sequence_length=512,
        device=device,
    )
    record_and_assert_close(
        torch_features,
        jax_features,
        dtype_policy="float32",
        **equivalence_criteria("float32", "exact"),
    )

    diagonals = linspace_values(RELATIVE_DIAGONAL_SHAPE, dtype=np.float32)
    torch_pairs = torch_attention._shift(torch.from_numpy(diagonals))
    jax_pairs = jax_attention._shift(
        jnp.asarray(diagonals), SEQUENCE_LENGTH, SEQUENCE_LENGTH
    )
    assert tuple(torch_pairs.shape) == PAIR_SHAPE
    assert tuple(jax_pairs.shape) == PAIR_SHAPE
    record_and_assert_close(
        torch_pairs,
        jax_pairs,
        dtype_policy="float32",
        **equivalence_criteria("float32", "exact"),
    )


### PREDICTION HEADS ###
def test_multi_organism_linear(
    pt_dtype_policy, record_and_assert_close
):
    values = normal_values(SEQUENCE_SHAPE)
    organisms = np.arange(BATCH_SIZE, dtype=np.int32) % NUM_ORGANISMS
    transformed, params = _initialize_haiku(
        lambda x: jax_heads._MultiOrganismLinear(CHANNELS, NUM_ORGANISMS)(
            x, jnp.asarray(organisms)
        ),
        values,
        jax_heads._MultiOrganismLinear,
        pt_dtype_policy,
        pt_dtype_policy.compute_uptype,
    )
    weights = linspace_values(
        (NUM_ORGANISMS, CHANNELS, CHANNELS), *WEIGHT_AND_BIAS_RANGE
    )
    bias = linspace_values(
        (NUM_ORGANISMS, CHANNELS), *WEIGHT_AND_BIAS_RANGE
    )
    params["multi_organism_linear"].update(
        w=_jax_array(weights, pt_dtype_policy.parameter_dtype),
        b=_jax_array(bias, pt_dtype_policy.parameter_dtype),
    )

    torch_layer = torch_heads.MultiOrganismLinear(CHANNELS, CHANNELS, NUM_ORGANISMS).to(
        pt_dtype_policy.parameter_dtype
    )
    with torch.no_grad():
        torch_layer.weight.copy_(
            _torch_array(weights, pt_dtype_policy.parameter_dtype)
        )
        torch_layer.bias.copy_(
            _torch_array(bias, pt_dtype_policy.parameter_dtype)
        )
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_layer(
            _torch_array(values, pt_dtype_policy.compute_dtype),
            torch.from_numpy(organisms),
        )
    with use_jax_compute_uptype_policy(pt_dtype_policy):
        jax_output = transformed.apply(
            params, _jax_array(values, pt_dtype_policy.compute_dtype)
        )

    assert torch_output.dtype == pt_dtype_policy.compute_uptype
    assert jax_output.dtype == jax_dtype(pt_dtype_policy.compute_uptype)
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


@pytest.mark.parametrize("apply_squashing", [False, True])
@pytest.mark.parametrize(
    "representation, torch_scaling, jax_scaling",
    [
        pytest.param(
            "predictions",
            torch_heads.predictions_scaling,
            jax_heads.predictions_scaling,
            id="predictions",
        ),
        pytest.param(
            "targets",
            torch_heads.targets_scaling,
            jax_heads.targets_scaling,
            id="targets",
        ),
    ],
)
def test_track_scaling(
    apply_squashing,
    representation,
    torch_scaling,
    jax_scaling,
    pt_dtype_policy,
    record_and_assert_close,
):
    values = linspace_values(TRACK_SHAPE, *TRACK_SCALING_VALUE_RANGE)
    means = linspace_values((BATCH_SIZE, NUM_TRACKS), *TRACK_MEAN_RANGE)
    resolution = 128
    torch_values = _torch_array(values, pt_dtype_policy.compute_dtype)
    torch_means = _torch_array(means, pt_dtype_policy.compute_dtype)
    jax_values = _jax_array(values, pt_dtype_policy.compute_dtype)
    jax_means = _jax_array(means, pt_dtype_policy.compute_dtype)
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_scaling(
            torch_values,
            torch_means,
            resolution=resolution,
            apply_squashing=apply_squashing,
        )
    jax_output = jax_scaling(
        jax_values,
        jax_means,
        resolution=resolution,
        apply_squashing=apply_squashing,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        name=representation,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### LOSSES ###
# Each loss receives a nontrivial mask. This verifies that JAX and PyTorch agree
# on which values contribute, not only on the unmasked loss formula.
def test_mse_loss(
    pt_dtype_policy, jax_compute_uptype_policy, record_and_assert_close
):
    targets = linspace_values(TRACK_SHAPE, *MSE_TARGET_RANGE)
    predictions = linspace_values(TRACK_SHAPE, *MSE_PREDICTION_RANGE)
    mask = np.ones(TRACK_MASK_SHAPE, dtype=bool)
    mask[..., -1] = False
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_losses.mse(
            y_true=_torch_array(targets, pt_dtype_policy.compute_dtype),
            y_pred=_torch_array(predictions, pt_dtype_policy.compute_dtype),
            mask=torch.from_numpy(mask),
        )
    jax_output = jax_losses.mse(
        y_true=_jax_array(targets, pt_dtype_policy.compute_dtype),
        y_pred=_jax_array(predictions, pt_dtype_policy.compute_dtype),
        mask=jnp.asarray(mask),
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_poisson_loss(
    pt_dtype_policy, jax_compute_uptype_policy, record_and_assert_close
):
    targets = linspace_values(TRACK_SHAPE, *COUNT_TARGET_RANGE)
    predictions = linspace_values(TRACK_SHAPE, *POSITIVE_PREDICTION_RANGE)
    mask = np.ones(TRACK_MASK_SHAPE, dtype=bool)
    mask[..., -1] = False
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_losses.poisson_loss(
            y_true=_torch_array(targets, pt_dtype_policy.compute_dtype),
            y_pred=_torch_array(predictions, pt_dtype_policy.compute_dtype),
            mask=torch.from_numpy(mask),
        )
    jax_output = jax_losses.poisson_loss(
        y_true=_jax_array(targets, pt_dtype_policy.compute_dtype),
        y_pred=_jax_array(predictions, pt_dtype_policy.compute_dtype),
        mask=jnp.asarray(mask),
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_multinomial_loss(
    pt_dtype_policy, jax_compute_uptype_policy, record_and_assert_close
):
    targets = linspace_values(TRACK_SHAPE, *COUNT_TARGET_RANGE)
    predictions = linspace_values(TRACK_SHAPE, *POSITIVE_PREDICTION_RANGE)
    mask = np.ones(TRACK_MASK_SHAPE, dtype=bool)
    mask[..., -1] = False
    multinomial_resolution = 4
    positional_weight = 5.0
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_losses.multinomial_loss(
            y_true=_torch_array(targets, pt_dtype_policy.compute_dtype),
            y_pred=_torch_array(predictions, pt_dtype_policy.compute_dtype),
            mask=torch.from_numpy(mask),
            min_zero=False,
            multinomial_resolution=multinomial_resolution,
            positional_weight=positional_weight,
        )
    jax_output = jax_losses.multinomial_loss(
        y_true=_jax_array(targets, pt_dtype_policy.compute_dtype),
        y_pred=_jax_array(predictions, pt_dtype_policy.compute_dtype),
        mask=jnp.asarray(mask),
        multinomial_resolution=multinomial_resolution,
        positional_weight=positional_weight,
    )
    for name in ("loss", "loss_total", "loss_positional"):
        record_and_assert_close(
            torch_output[name],
            jax_output[name],
            name=name,
            dtype_policy=pt_dtype_policy.name,
            **equivalence_criteria(
                pt_dtype_policy.name, "module"
            ),
        )


def test_cross_entropy_from_logits(
    pt_dtype_policy, jax_compute_uptype_policy, record_and_assert_close
):
    logits = linspace_values(CLASSIFICATION_SHAPE, *LOGIT_RANGE)
    labels = np.arange(BATCH_SIZE * SEQUENCE_LENGTH).reshape(
        BATCH_SIZE, SEQUENCE_LENGTH
    ) % NUM_CLASSES
    targets = np.eye(NUM_CLASSES, dtype=np.float32)[labels]
    mask = np.ones(CLASSIFICATION_SHAPE, dtype=bool)
    mask[..., -1] = False
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_losses.cross_entropy_loss_from_logits(
            y_pred_logits=_torch_array(logits, pt_dtype_policy.compute_dtype),
            y_true=_torch_array(targets, pt_dtype_policy.compute_dtype),
            mask=torch.from_numpy(mask),
            axis=-1,
        )
    jax_output = jax_losses.cross_entropy_loss_from_logits(
        y_pred_logits=_jax_array(logits, pt_dtype_policy.compute_dtype),
        y_true=_jax_array(targets, pt_dtype_policy.compute_dtype),
        mask=jnp.asarray(mask),
        axis=-1,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_binary_cross_entropy_from_logits(
    pt_dtype_policy, jax_compute_uptype_policy, record_and_assert_close
):
    logits = linspace_values(TRACK_SHAPE, *LOGIT_RANGE)
    targets = linspace_values(TRACK_SHAPE, *PROBABILITY_TARGET_RANGE)
    mask = np.ones(TRACK_MASK_SHAPE, dtype=bool)
    mask[..., -1] = False
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_losses.binary_crossentropy_from_logits(
            y_pred=_torch_array(logits, pt_dtype_policy.compute_dtype),
            y_true=_torch_array(targets, pt_dtype_policy.compute_dtype),
            mask=torch.from_numpy(mask),
        )
    jax_output = jax_losses.binary_crossentropy_from_logits(
        y_pred=_jax_array(logits, pt_dtype_policy.compute_dtype),
        y_true=_jax_array(targets, pt_dtype_policy.compute_dtype),
        mask=jnp.asarray(mask),
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_cross_entropy_on_counts(
    pt_dtype_policy, jax_compute_uptype_policy, record_and_assert_close
):
    targets = linspace_values(TRACK_SHAPE, *COUNT_TARGET_RANGE)
    predictions = linspace_values(TRACK_SHAPE, *POSITIVE_PREDICTION_RANGE)
    mask = np.ones(TRACK_SHAPE, dtype=bool)
    mask[..., -1] = False
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = torch_losses.cross_entropy_loss(
            y_true=_torch_array(targets, pt_dtype_policy.compute_dtype),
            y_pred=_torch_array(predictions, pt_dtype_policy.compute_dtype),
            mask=torch.from_numpy(mask),
            axis=1,
        )
    jax_output = jax_losses.cross_entropy_loss(
        y_true=_jax_array(targets, pt_dtype_policy.compute_dtype),
        y_pred=_jax_array(predictions, pt_dtype_policy.compute_dtype),
        mask=jnp.asarray(mask),
        axis=1,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


### TASK OPERATIONS ###
def test_splice_site_positions(record_and_assert_close):
    predictions = linspace_values(SPLICE_SITE_SHAPE, 0.05, 0.95, dtype=np.float32)
    k = 3
    pad_to_length = 5
    threshold = 0.4
    torch_output = torch_splicing.generate_splice_site_positions(
        torch.from_numpy(predictions),
        None,
        None,
        k=k,
        pad_to_length=pad_to_length,
        threshold=threshold,
    )
    jax_output = jax_splicing.generate_splice_site_positions(
        jnp.asarray(predictions),
        None,
        None,
        k=k,
        pad_to_length=pad_to_length,
        threshold=threshold,
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy="float32",
        **equivalence_criteria("float32", "exact"),
    )


### ETC ###
def test_dot_implementations(
    pt_dtype_policy, record_and_assert_close
):
    left = normal_values(SEQUENCE_SHAPE, seed=0)
    right = normal_values(ATTENTION_PROJECTION_SHAPE, seed=1)
    with dtype_policy_context(pt_dtype_policy, "cpu"):
        torch_output = dot_with_dtype_policy(
            _torch_array(left, pt_dtype_policy.compute_dtype),
            _torch_array(right, pt_dtype_policy.compute_dtype),
        )
    jax_operand_dtype = jax_dtype(pt_dtype_policy.compute_dtype)
    jax_output = jnp.einsum(
        "bqi,bik->bqk",
        jnp.asarray(left, dtype=jax_operand_dtype),
        jnp.asarray(right, dtype=jax_operand_dtype),
        precision=jax_dot_algorithm(pt_dtype_policy),
    )
    record_and_assert_close(
        torch_output,
        jax_output,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "module"),
    )


def test_policy_compute_operand_cast(
    pt_dtype_policy, record_and_assert_close
):
    values = linspace_values(SEQUENCE_SHAPE, -8, 8)
    pytorch_values = _torch_array(
        values, pt_dtype_policy.compute_dtype
    )
    jax_values = _jax_array(
        values, pt_dtype_policy.compute_dtype
    )
    record_and_assert_close(
        pytorch_values,
        jax_values,
        dtype_policy=pt_dtype_policy.name,
        **equivalence_criteria(pt_dtype_policy.name, "exact"),
    )
