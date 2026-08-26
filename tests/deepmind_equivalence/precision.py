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
    "float32": 1e-5,
    "float64": 1e-7,
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
#         "module": {"relative_L2": 1e-5, "relative_Linf": 2e-5},
#         "architecture": {"relative_L2": 2e-5, "relative_Linf": 4e-5},
#         "full_model": {
#             "default": {"relative_L2": 5e-5, "relative_Linf": 1e-4},
#             "junction": {"relative_L2": 1.25e-4, "relative_Linf": 2.5e-4},
#             "descaled": {"relative_L2": 1.25e-4, "relative_Linf": 2.5e-4},
#         },
#     },
#     "float64": {
#         "exact": {"relative_L2": 0, "relative_Linf": 0},
#         "module": {"relative_L2": 1e-7, "relative_Linf": 2e-7},
#         "architecture": {"relative_L2": 2e-7, "relative_Linf": 4e-7},
#         "full_model": {
#             "default": {"relative_L2": 5e-7, "relative_Linf": 1e-6},
#             "junction": {"relative_L2": 1.25e-6, "relative_Linf": 2.5e-6},
#             "descaled": {"relative_L2": 1.25e-6, "relative_Linf": 2.5e-6},
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
    """Replace fixed JAX AlphaGenome compute uptypes with the policy.

    The reference uses FP32 normalization statistics, head and loss casts,
    reductions, and preferred element types. It also fixes four attention and
    two splice-junction einsums to BF16_BF16_F32. This proxy redirects those
    explicit choices to compute_uptype while leaving ordinary computation and
    autocasts unchanged.

    Essentially, this is a way for us to slice into JAX modules for upcasting
    changes that are needed to comparison with the PyTorch policy.

    The targeted modules are:
    - alphagenome_research.model.attention: explicit dot algorithms and outputs
    - alphagenome_research.model.heads: casts, reductions, and contractions
    - alphagenome_research.model.layers: LayerNorm and RMSBatchNorm statistics
    - alphagenome_research.model.losses: casts and reductions
    """
    import jax
    from alphagenome_research.model import attention, heads, layers, losses

    source_algorithm = jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    target_algorithm = jax_dot_algorithm(pt_dtype_policy)
    source_dtype = jax_dtype(torch.float32)
    target_dtype = jax_dtype(pt_dtype_policy.compute_uptype)

    class PolicyNumpy:
        def __init__(
            self,
            numpy_module,
            *,
            replace_float32=False,
            replace_mean_dtype=False,
        ):
            self._numpy_module = numpy_module
            self._replace_float32 = replace_float32
            self._replace_mean_dtype = replace_mean_dtype

        def __getattr__(self, name):
            if self._replace_float32 and name == "float32":
                return target_dtype
            return getattr(self._numpy_module, name)

        def mean(self, *args, **kwargs):
            if (
                self._replace_mean_dtype
                and kwargs.get("dtype") == source_dtype
            ):
                kwargs = {**kwargs, "dtype": target_dtype}
            return self._numpy_module.mean(*args, **kwargs)

        def einsum(self, *args, **kwargs):
            replacements = {}
            if kwargs.get("precision") == source_algorithm:
                replacements["precision"] = target_algorithm
            if kwargs.get("preferred_element_type") == source_dtype:
                replacements["preferred_element_type"] = target_dtype
            return self._numpy_module.einsum(
                *args, **{**kwargs, **replacements}
            )

    module_options = (
        (attention, False, False),
        (heads, True, False),
        (layers, False, True),
        (losses, True, False),
    )
    original_numpy_modules = tuple(
        module.jnp for module, _, _ in module_options
    )
    try:
        for module, replace_float32, replace_mean_dtype in module_options:
            module.jnp = PolicyNumpy(
                module.jnp,
                replace_float32=replace_float32,
                replace_mean_dtype=replace_mean_dtype,
            )
        yield
    finally:
        for (module, _, _), original_numpy_module in zip(
            module_options, original_numpy_modules
        ):
            module.jnp = original_numpy_module
