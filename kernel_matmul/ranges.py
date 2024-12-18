import math
import os
import warnings
import torch
from torch import Tensor

from kernel_matmul import _BLOCK_SIZE
from kernel_matmul.compile import get_native_module_path

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
    module_path = get_native_module_path("ranges")
    return torch.utils.cpp_extension.load(
        name=name,
        sources=[
            os.path.join(module_path, "ranges.cpp"),
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
    """Compute the sparsity pattern for a given cutoff and input.

    This function uses an optimized C++ implementation on the CPU.

    Args:
        cutoff (float | None): The cutoff distance. If None, a dense pattern will be returned.
        x1 (Tensor): The first input tensor of shape (batch..., n1).
        x2 (Tensor | None, optional): The second input tensor of shape (batch..., n2).
            If None, the sparsity pattern will be symmetric. Defaults to None.
        block_size (int, optional): _description_. Defaults to _BLOCK_SIZE.
        align (bool, optional): _description_. If the pattern is aligned, its transpose can be
            computed exactly. Defaults to False.

    Returns:
        tuple[Tensor, Tensor]: The start and end indices of the blocks in the sparsity pattern.
    """
    batch_shape = x1.shape[:-1]
    if len(batch_shape) == 0:
        x1 = x1.unsqueeze(0)
        if x2 is not None:
            x2 = x2.unsqueeze(0)
    elif len(batch_shape) > 1:
        x1 = x1.reshape(-1, x1.shape[-1])
        if x2 is not None:
            x2 = x2.reshape(-1, x2.shape[-1])

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

    start = start.reshape(*batch_shape, start.shape[-1])
    end = end.reshape(*batch_shape, end.shape[-1])
    return start, end


def transpose_ranges(
    start: Tensor, end: Tensor, x1_size: int, x2_size: int, *, block_size: int = _BLOCK_SIZE
) -> tuple[Tensor, Tensor]:
    """Transpose a sparsity pattern.

    Args:
        start (Tensor): Start indices.
        end (Tensor): End indices.
        x1_size (int): Size of x1 (n1).
        x2_size (int): Size of x2 (n2).
        block_size (int, optional): _description_. Defaults to _BLOCK_SIZE.

    Raises:
        ValueError: If the original pattern is not aligned.

    Returns:
        tuple[Tensor, Tensor]: The start and end indices of the transposed pattern.
    """
    if torch.any(torch.logical_and(start % block_size != 0, start != x2_size)) or torch.any(
        torch.logical_and(end % block_size != 0, end != x2_size)
    ):
        raise ValueError("Ranges must be block-aligned")

    batch_shape = start.shape[:-1]
    if len(batch_shape) == 0:
        start = start.unsqueeze(0)
        end = end.unsqueeze(0)
    elif len(batch_shape) > 1:
        start = start.reshape(-1, start.shape[-1])
        end = end.reshape(-1, end.shape[-1])

    start, end = _native.transpose_ranges(start, end, x1_size, x2_size, block_size)

    start = start.reshape(*batch_shape, start.shape[-1])
    end = end.reshape(*batch_shape, end.shape[-1])
    return start, end


class RangesCache:
    """Cache for sparsity patterns for a fixed x = x1 = x2 and variable cutoff.

    Can be used like a dictionary with cutoffs as keys.
    """

    _cache: dict[float | None, tuple[Tensor, Tensor]]

    def __init__(self, x: Tensor, align: bool = False):
        self._x = x
        self._cache = {}
        self._align = align

    @property
    def x(self):
        return self._x

    @property
    def align(self):
        return self._align

    def __getitem__(self, cutoff: float | None) -> tuple[Tensor, Tensor]:
        entry = self._cache.get(cutoff)
        if entry is None:
            entry = make_ranges(cutoff, self._x, align=self._align)
            self._cache[cutoff] = entry
        return entry
