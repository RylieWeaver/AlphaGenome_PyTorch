# External
from contextlib import contextmanager
import torch

# Internal
from alphagenome_pt.precision import dtype_policy_context


EQUIVALENCE_TEST_POLICIES = ("deepmind", "float32", "float64")

# FORMULA: Threshold is the product of precision, composition, metric,
# and representation coefficients.
_PRECISION_COEFFICIENTS = {
    "deepmind": 1e-2,
    "float32": 1e-6,
    "float64": 1e-8,
}
_COMPOSITION_COEFFICIENTS = {
    "module": 1,
    "architecture": 2,
    "full_model": 5,
}
_METRIC_COEFFICIENTS = {
    "relative_L2": 1,
    "relative_Linf": 2,
}
_REPRESENTATION_COEFFICIENTS = {
    "default": 1,
    "junction": 2.5,
    "descaled": 2.5,
}

EQUIVALENCE_THRESHOLDS = {
    policy_name: {
        "exact": dict.fromkeys(_METRIC_COEFFICIENTS, 0),
        **{
            composition: {
                metric: (
                    precision_coefficient
                    * composition_coefficient
                    * metric_coefficient
                )
                for metric, metric_coefficient in _METRIC_COEFFICIENTS.items()
            }
            for composition, composition_coefficient in (
                _COMPOSITION_COEFFICIENTS.items()
            )
        },
    }
    for policy_name, precision_coefficient in _PRECISION_COEFFICIENTS.items()
}

# Resulting thresholds:
# {
#     "deepmind": {
#         "exact": {"relative_L2": 0, "relative_Linf": 0},
#         "module": {"relative_L2": 1e-2, "relative_Linf": 2e-2},
#         "architecture": {"relative_L2": 2e-2, "relative_Linf": 4e-2},
#         "full_model": {
#             "default": {"relative_L2": 5e-2, "relative_Linf": 1e-1},
#             "junction": {"relative_L2": 1.25e-1, "relative_Linf": 2.5e-1},
#             "descaled": {"relative_L2": 1.25e-1, "relative_Linf": 2.5e-1},
#         },
#     },
#     "float32": {
#         "exact": {"relative_L2": 0, "relative_Linf": 0},
#         "module": {"relative_L2": 1e-6, "relative_Linf": 2e-6},
#         "architecture": {"relative_L2": 2e-6, "relative_Linf": 4e-6},
#         "full_model": {
#             "default": {"relative_L2": 5e-6, "relative_Linf": 1e-5},
#             "junction": {"relative_L2": 1.25e-5, "relative_Linf": 2.5e-5},
#             "descaled": {"relative_L2": 1.25e-5, "relative_Linf": 2.5e-5},
#         },
#     },
#     "float64": {
#         "exact": {"relative_L2": 0, "relative_Linf": 0},
#         "module": {"relative_L2": 1e-8, "relative_Linf": 2e-8},
#         "architecture": {"relative_L2": 2e-8, "relative_Linf": 4e-8},
#         "full_model": {
#             "default": {"relative_L2": 5e-8, "relative_Linf": 1e-7},
#             "junction": {"relative_L2": 1.25e-7, "relative_Linf": 2.5e-7},
#             "descaled": {"relative_L2": 1.25e-7, "relative_Linf": 2.5e-7},
#         },
#     },
# }


def jax_dtype(torch_dtype):
    """Return the JAX dtype corresponding to a supported PyTorch dtype."""
    import jax.numpy as jnp

    pt2jax_types = {
        torch.bfloat16: jnp.bfloat16,
        torch.float16: jnp.float16,
        torch.float32: jnp.float32,
        torch.float64: jnp.float64,
    }
    try:
        return pt2jax_types[torch_dtype]
    except KeyError as error:
        raise ValueError(
            f"JAX cannot represent PyTorch dtype {torch_dtype}."
        ) from error


def torch_dtype(jax_dtype):
    """Return the PyTorch dtype corresponding to a supported JAX dtype."""
    import jax.numpy as jnp

    jax2pt_types = {
        jnp.bfloat16: torch.bfloat16,
        jnp.float16: torch.float16,
        jnp.float32: torch.float32,
        jnp.float64: torch.float64,
    }
    try:
        return jax2pt_types[jax_dtype]
    except KeyError as error:
        raise ValueError(
            f"PyTorch cannot represent JAX dtype {jax_dtype}."
        ) from error


def equivalence_criteria(
    dtype_policy, composition, representation="default"
):
    """Return relative-error criteria for one precision tier and representation."""
    try:
        thresholds = EQUIVALENCE_THRESHOLDS[dtype_policy][composition]
        representation_coefficient = _REPRESENTATION_COEFFICIENTS[
            representation
        ]
    except KeyError as error:
        raise ValueError(
            f"Unsupported equivalence criterion: "
            f"dtype_policy={dtype_policy!r}, composition={composition!r}, "
            f"representation={representation!r}."
        ) from error
    return {
        f"{metric}_threshold": threshold * representation_coefficient
        for metric, threshold in thresholds.items()
    }


@contextmanager
def use_dtype_policy(model, pt_dtype_policy):
    """Temporarily apply one PyTorch dtype policy to a loaded model."""
    original_pt_dtype_policy = model.dtype_policy
    original_dtype = next(model.parameters()).dtype
    try:
        model.dtype_policy = pt_dtype_policy
        model.to(dtype=pt_dtype_policy.parameter_dtype)
        with dtype_policy_context(pt_dtype_policy, model.device.type):
            yield
    finally:
        model.dtype_policy = original_pt_dtype_policy
        model.to(dtype=original_dtype)


def jax_mixed_precision_policy(pt_dtype_policy):
    """Create a JAX policy from a PyTorch dtype policy."""
    import jax
    import jmp

    pt_dtypes = (
        pt_dtype_policy.parameter_dtype,
        pt_dtype_policy.compute_dtype,
        pt_dtype_policy.output_dtype,
    )
    jax_dtype_names = tuple(
        jax_dtype(dtype).__name__ for dtype in pt_dtypes
    )
    if torch.float64 in pt_dtypes:
        jax.config.update("jax_enable_x64", True)
    return jmp.get_policy(
        "params="
        f"{jax_dtype_names[0]},compute={jax_dtype_names[1]},"
        f"output={jax_dtype_names[2]}"
    )


def jax_dot_algorithm(pt_dtype_policy):
    """Select JAX's explicit dot algorithm from a PyTorch dtype policy."""
    import jax

    pt_dtypes_to_jax_algorithms = {
        (torch.bfloat16, torch.bfloat16): "BF16_BF16_BF16",
        (torch.bfloat16, torch.float32): "BF16_BF16_F32",
        (torch.float16, torch.float16): "F16_F16_F16",
        (torch.float16, torch.float32): "F16_F16_F32",
        (torch.float32, torch.float32): "F32_F32_F32",
        (torch.float64, torch.float64): "F64_F64_F64",
    }
    pt_dot_dtypes = (
        pt_dtype_policy.compute_dtype,
        pt_dtype_policy.compute_uptype,
    )
    try:
        algorithm_name = pt_dtypes_to_jax_algorithms[
            pt_dot_dtypes
        ]
        return getattr(jax.lax.DotAlgorithmPreset, algorithm_name)
    except KeyError as error:
        raise ValueError(
            "JAX has no configured dot algorithm for operand/output dtypes "
            f"{pt_dot_dtypes}."
        ) from error


@contextmanager
def use_jax_compute_uptype_policy(pt_dtype_policy):
    """Redirect non-uniform JAX precision choices through the PyTorch policy.

    Every explicit ``jnp.float32`` in the four core reference modules maps to
    ``compute_uptype``. This covers:

    - attention: logits, post-bias casts, preferred contraction outputs, and
      sequence-to-pair relative positions;
    - heads: pooling and gene reductions, MultiOrganismLinear preferred output,
      classification/SSU upcasts, junction RoPE values, and junction losses;
    - layers: RMSBatchNorm EMA state and statistics plus LayerNorm statistics;
    - losses: operand casts, reductions, totals, and log-softmax calculations.

    The sole explicit ``jnp.float16`` in these modules is the SSU prediction
    output. It remains FP16 for BF16 compute (preserving the reference's
    intermediate rounding), but maps to ``compute_dtype`` for FP32/FP64.

    Fixed ``BF16_BF16_F32`` einsum presets are redirected separately because
    replacing ``jnp.float32`` cannot alter a JAX dot-algorithm enum. Ordinary
    computation and mixed-precision autocasts remain unchanged.
    """
    import jax
    from alphagenome_research.model import attention, heads, layers, losses

    source_algorithm = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    target_algorithm = jax_dot_algorithm(pt_dtype_policy)
    target_dtype = jax_dtype(pt_dtype_policy.compute_uptype)
    target_compute_dtype = jax_dtype(pt_dtype_policy.compute_dtype)
    target_float16_dtype = (
        jax_dtype(torch.float16)
        if pt_dtype_policy.compute_dtype == torch.bfloat16
        else target_compute_dtype
    )

    class PolicyNumpy:
        def __init__(
            self,
            numpy_module,
            *,
            replace_float16=False,
        ):
            self._numpy_module = numpy_module
            self._replace_float16 = replace_float16

        def __getattr__(self, name):
            if name == "float32":
                return target_dtype
            if self._replace_float16 and name == "float16":
                return target_float16_dtype
            return getattr(self._numpy_module, name)

        def einsum(self, *args, **kwargs):
            if kwargs.get("precision") == source_algorithm:
                kwargs = {**kwargs, "precision": target_algorithm}
            return self._numpy_module.einsum(*args, **kwargs)

    module_options = (
        (attention, False),
        (heads, True),
        (layers, False),
        (losses, False),
    )
    original_numpy_modules = tuple(
        module.jnp for module, _ in module_options
    )
    try:
        for module, replace_float16 in module_options:
            module.jnp = PolicyNumpy(
                module.jnp,
                replace_float16=replace_float16,
            )
        yield
    finally:
        for (module, _), original_numpy_module in zip(
            module_options, original_numpy_modules
        ):
            module.jnp = original_numpy_module
