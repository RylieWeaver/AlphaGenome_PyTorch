# External
from contextlib import contextmanager
import numpy as np
import torch


def jax_device(device):
    """Resolve a PyTorch-style device string to the corresponding JAX device."""
    import jax

    platform, separator, index = device.partition(":")
    if platform == "cuda":
        platform = "gpu"
    devices = jax.devices(platform)
    device_index = int(index) if separator else 0
    if device_index >= len(devices):
        raise RuntimeError(
            f"JAX has no {device!r} device; found {len(devices)} {platform} device(s)"
        )
    return devices[device_index]


def linspace_values(shape, low=-1.0, high=1.0, *, dtype=np.float64):
    """Create deterministic values evenly spaced across a shape.

    Note that this is in fp64.
    """
    return np.linspace(low, high, np.prod(shape), dtype=dtype).reshape(shape)


def normal_values(shape, *, seed=0, dtype=np.float64):
    """Draw deterministic standard-normal values from NumPy's default RNG."""
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


def one_hot_dna_values(batch_size, sequence_length):
    """Create deterministic DNA cycling through the four one-hot bases."""
    bases = np.arange(sequence_length) % 4
    sequence = np.eye(4, dtype=np.float64)[bases]
    return np.broadcast_to(
        sequence, (batch_size, sequence_length, 4)
    ).copy()


@contextmanager
def use_jax_junction_padding_mask():
    """Make the JAX junction head respect padded organism metadata.

    The released head counts the length of already padded junction metadata,
    so it treats all 367 mouse tissues as valid instead of the real 90. The
    public inference wrapper later removes those tracks, but the raw research
    loss does not. This temporary replacement uses the metadata padding field
    so checkpoint prediction and loss comparisons cover only real tissues.
    """
    import jax.numpy as jnp
    from alphagenome_research.model import heads

    original_method = (
        heads.SpliceSitesJunctionHead.get_multi_organism_track_mask
    )

    def padding_aware_mask(head):
        track_masks = []
        for organism in head._metadata:
            padding = np.asarray(
                head._metadata[organism].padding[head._output_type]
            )
            tissue_mask = np.logical_not(padding)
            track_masks.append(
                np.concatenate([tissue_mask, tissue_mask])
            )
        return jnp.stack(track_masks).astype(bool)

    try:
        heads.SpliceSitesJunctionHead.get_multi_organism_track_mask = (
            padding_aware_mask
        )
        yield
    finally:
        heads.SpliceSitesJunctionHead.get_multi_organism_track_mask = (
            original_method
        )


def _numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        # NumPy has no native bfloat16 dtype. Widening both frameworks' BF16
        # outputs to float32 preserves their represented values for comparison.
        if value.dtype == torch.bfloat16:
            value = value.float()
    value = np.asarray(value)
    return value.astype(np.float32) if str(value.dtype) == "bfloat16" else value


def difference_metrics(pytorch_value, jax_value, *, name="value"):
    """Validate comparability and summarize error relative to the JAX reference."""
    pytorch_value = _numpy(pytorch_value)
    jax_value = _numpy(jax_value)
    assert pytorch_value.shape == jax_value.shape, (
        f"{name} shape mismatch: "
        f"PyTorch {pytorch_value.shape}, JAX {jax_value.shape}"
    )

    floating = (
        np.issubdtype(pytorch_value.dtype, np.floating)
        and np.issubdtype(jax_value.dtype, np.floating)
    )
    if not floating:
        np.testing.assert_array_equal(pytorch_value, jax_value, err_msg=name)
        reference_abs = np.abs(jax_value.astype(np.float64))
        return {
            "relative_L2": 0.0,
            "relative_Linf": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "reference_max_abs": float(reference_abs.max(initial=0)),
            "reference_mean_abs": (
                float(reference_abs.mean()) if reference_abs.size else 0.0
            ),
            "exact_fraction": 1.0,
            "num_values": int(pytorch_value.size),
        }

    for nonfinite_type, detector in (
        ("NaN", np.isnan),
        ("positive infinity", np.isposinf),
        ("negative infinity", np.isneginf),
    ):
        assert np.array_equal(detector(pytorch_value), detector(jax_value)), (
            f"{name} has mismatched {nonfinite_type} locations"
        )

    valid = np.isfinite(jax_value)
    assert valid.any(), f"{name} has no finite values to compare"
    pytorch_finite = pytorch_value[valid].astype(np.float64)
    jax_finite = jax_value[valid].astype(np.float64)
    difference = np.abs(pytorch_finite - jax_finite)
    reference_abs = np.abs(jax_finite)
    difference_norm = np.linalg.norm(pytorch_finite - jax_finite)
    difference_max = difference.max(initial=0)
    reference_norm = np.linalg.norm(jax_finite)
    reference_max = reference_abs.max(initial=0)
    minimum_scale = np.finfo(np.float64).tiny
    relative_L2 = (
        difference_norm / reference_norm
        if reference_norm > minimum_scale
        else 0.0 if difference_norm == 0 else np.inf
    )
    relative_Linf = (
        difference_max / reference_max
        if reference_max > minimum_scale
        else 0.0 if difference_max == 0 else np.inf
    )
    return {
        "relative_L2": float(relative_L2),
        "relative_Linf": float(relative_Linf),
        "max_abs": float(difference_max),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
        "reference_max_abs": float(reference_max),
        "reference_mean_abs": (
            float(reference_abs.mean()) if reference_abs.size else 0.0
        ),
        "exact_fraction": float(np.mean(pytorch_finite == jax_finite)),
        "num_values": int(difference.size),
    }
