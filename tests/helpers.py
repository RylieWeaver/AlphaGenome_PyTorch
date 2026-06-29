# External
import torch


def assert_finite_scalars(scalars: dict[str, torch.Tensor]) -> None:
    for key, value in scalars.items():
        assert torch.isfinite(value), key
