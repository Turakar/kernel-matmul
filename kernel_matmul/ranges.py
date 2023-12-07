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
    cutoff: float | None, x1: Tensor, x2: Tensor | None = None, *, block_size: int = _BLOCK_SIZE
) -> tuple[Tensor, Tensor]:
    if cutoff is None:
        rows = int(math.ceil(x1.shape[0] / block_size))
        start = torch.zeros(rows, dtype=torch.int32, device=x1.device)
        end = torch.full((rows,), x2.shape[0], dtype=torch.int32, device=x2.device)
        return start, end
    if x2 is None:
        return _native.make_ranges_symmetric(x1, cutoff, block_size)
    else:
        return _native.make_ranges(x1, x2, cutoff, block_size)
