# External
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass, replace
import torch


@dataclass(frozen=True)
class DtypePolicy:
    name: str
    parameter_dtype: torch.dtype
    input_dtype: torch.dtype
    compute_dtype: torch.dtype
    compute_uptype: torch.dtype  # must be at least as precise as compute_dtype
    output_dtype: torch.dtype

    def __post_init__(self):
        precision = {
            torch.float16: 16,
            torch.bfloat16: 16,
            torch.float32: 32,
            torch.float64: 64,
        }
        try:
            compute_precision = precision[self.compute_dtype]
            for name in ("input_dtype", "compute_uptype"):
                if precision[getattr(self, name)] < compute_precision:
                    raise ValueError(
                        f"{name} must be at least as precise as compute_dtype."
                    )
        except KeyError as error:
            raise ValueError(
                f"Unsupported policy dtype {error.args[0]}."
            ) from error

    # cast_tree is a recursive helper for cast_output
    def _cast_tree(self, value, dtype):
        from .losses import LossLeaf, MetricTree

        if isinstance(value, torch.Tensor):
            return value.to(dtype) if value.is_floating_point() else value
        if isinstance(value, LossLeaf):
            return LossLeaf(self._cast_tree(value.value, dtype))
        if isinstance(value, MetricTree):
            return MetricTree(self._cast_tree(value.children, dtype))
        if isinstance(value, dict):
            return {
                key: self._cast_tree(child, dtype)
                for key, child in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._cast_tree(child, dtype) for child in value)
        if isinstance(value, list):
            return [self._cast_tree(child, dtype) for child in value]
        if is_dataclass(value) and not isinstance(value, type):
            return replace(
                value,
                **{
                    field.name: self._cast_tree(getattr(value, field.name), dtype)
                    for field in fields(value)
                },
            )
        # DataBatch has optional fields and retains the original DNA strings.
        if value is None or isinstance(value, str):
            return value
        raise TypeError(
            "DtypePolicy cannot cast an unsupported tree leaf of type "
            f"{type(value).__name__}."
        )

    def cast_output(self, value):
        return self._cast_tree(value, self.output_dtype)


# Uniform policies
BF16_DTYPE_POLICY = DtypePolicy(
    name="bfloat16",
    parameter_dtype=torch.bfloat16,
    input_dtype=torch.bfloat16,
    compute_dtype=torch.bfloat16,
    compute_uptype=torch.float32,
    output_dtype=torch.bfloat16,
)

FLOAT32_DTYPE_POLICY = DtypePolicy(
    name="float32",
    parameter_dtype=torch.float32,
    input_dtype=torch.float32,
    compute_dtype=torch.float32,
    compute_uptype=torch.float32,
    output_dtype=torch.float32,
)

FLOAT64_DTYPE_POLICY = DtypePolicy(
    name="float64",
    parameter_dtype=torch.float64,
    input_dtype=torch.float64,
    compute_dtype=torch.float64,
    compute_uptype=torch.float64,
    output_dtype=torch.float64,
)


# Mixed policies
FP32_PARAMS_BF16_COMPUTE_FP32_COMPUTE_UPTYPE_BF16_OUTPUT_POLICY = DtypePolicy(
    # NOTE: same as deepmind
    name="fp32_params_bf16_compute_fp32_compute_uptype_bf16_output",
    parameter_dtype=torch.float32,
    input_dtype=torch.float32,
    compute_dtype=torch.bfloat16,
    compute_uptype=torch.float32,
    output_dtype=torch.bfloat16,
)

DEEPMIND_DTYPE_POLICY = DtypePolicy(
    name="deepmind",
    parameter_dtype=torch.float32,
    input_dtype=torch.float32,
    compute_dtype=torch.bfloat16,
    compute_uptype=torch.float32,
    output_dtype=torch.bfloat16,
)


_DTYPE_POLICIES = {
    policy.name: policy
    for policy in (
        BF16_DTYPE_POLICY,
        FLOAT32_DTYPE_POLICY,
        FLOAT64_DTYPE_POLICY,
        FP32_PARAMS_BF16_COMPUTE_FP32_COMPUTE_UPTYPE_BF16_OUTPUT_POLICY,
        DEEPMIND_DTYPE_POLICY,
    )
}


def get_dtype_policy(name: str) -> DtypePolicy:
    try:
        return _DTYPE_POLICIES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_DTYPE_POLICIES))
        raise ValueError(
            f"Unknown dtype policy {name!r}; expected one of: {choices}."
        ) from error


_ACTIVE_DTYPE_POLICY = ContextVar(
    "alphagenome_dtype_policy",
    default=DEEPMIND_DTYPE_POLICY,
)


@contextmanager
def dtype_policy_context(policy: DtypePolicy, device_type: str):
    """Unify torch autocast and our specially casted operations into one context."""
    token = _ACTIVE_DTYPE_POLICY.set(policy)
    use_autocast = policy.compute_dtype in (torch.float16, torch.bfloat16)
    try:
        with torch.autocast(
            device_type=device_type,
            dtype=policy.compute_dtype,
            enabled=use_autocast,
        ):
            yield
    # NOTE: Customary to reset after contextual execution, even
    # though we'll probably keep using the same policy.
    finally:
        _ACTIVE_DTYPE_POLICY.reset(token)


def dot_with_dtype_policy(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Apply the active policy's dtypes to a batched dot product."""
    policy = _ACTIVE_DTYPE_POLICY.get()
    left = left.to(policy.compute_dtype)
    right = right.to(policy.compute_dtype)
    if (
        left.is_cuda
        and policy.compute_dtype in (torch.float16, torch.bfloat16)
        and policy.compute_uptype == torch.float32
    ):
        try:
            return torch.bmm(left, right, out_dtype=policy.compute_uptype)
        except TypeError:
            pass

    # Fallback: widen compute-dtype operands to the compute uptype.
    with torch.autocast(device_type=left.device.type, enabled=False):
        return torch.bmm(
            left.to(policy.compute_uptype),
            right.to(policy.compute_uptype),
        )
