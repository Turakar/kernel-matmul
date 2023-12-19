import math
import os
import warnings
import torch
from torch import Tensor

from kernel_matmul import _BLOCK_SIZE

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="pkg_resources is deprecated as an API"
    )
    import torch.utils.cpp_extension


def _load_native():
    debug = os.environ.get("KERNEL_MATMUL_COMPILE_DEBUG", "false") == "true"
    if debug:
        name = "km___ranges___debug"
        flags = ["-g"]
    else:
        name = "km___ranges"
        flags = ["-O3"]
    return torch.utils.cpp_extension.load(
        name=name,
        sources=[
            os.path.join("native", "ranges", "ranges.cpp"),
        ],
        extra_cflags=flags,
        verbose=False,
    )


_native = _load_native()


def make_ranges(
    cutoff: float | None,
    x1: Tensor,
    x2: Tensor | None = None,
    *,
    block_size: int = _BLOCK_SIZE,
    align: bool = False,
) -> tuple[Tensor, Tensor]:
    single = x1.dim() == 1
    if single:
        x1 = x1.unsqueeze(0)
        if x2 is not None:
            x2 = x2.unsqueeze(0)

    if cutoff is None:
        rows = int(math.ceil(x1.shape[1] / block_size))
        start = torch.zeros(x1.shape[0], rows, dtype=torch.int32, device=x1.device)
        end = torch.full(
            (x1.shape[0], rows),
            x2.shape[1] if x2 is not None else x1.shape[1],
            dtype=torch.int32,
            device=x1.device,
        )
    elif x2 is None:
        start, end = _native.make_ranges_symmetric(x1, cutoff, block_size)
    else:
        start, end = _native.make_ranges(x1, x2, cutoff, block_size, align)

    if single:
        start = start.squeeze(0)
        end = end.squeeze(0)
    return start, end


def transpose_ranges(
    start: Tensor, end: Tensor, x1_size: int, x2_size: int, *, block_size: int = _BLOCK_SIZE
) -> tuple[Tensor, Tensor]:
    single = start.dim() == 1
    if single:
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)

    if torch.any(start % block_size != 0) or torch.any(
        torch.logical_and(end % block_size != 0, end != x2_size)
    ):
        raise ValueError("Ranges must be block-aligned")

    start, end = _native.transpose_ranges(start, end, x1_size, x2_size, block_size)

    if single:
        start = start.squeeze(0)
        end = end.squeeze(0)
    return start, end
