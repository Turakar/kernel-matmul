import math
import multiprocessing

import torch
from kernel_matmul.linear_operator import KernelMatmulLinearOperator
from kernel_matmul.ranges import make_ranges


from gpytorch.kernels import Kernel
from torch import Size, Tensor


import abc


class KernelMatmulKernel(Kernel, abc.ABC):
    """Base class for KernelMatmul-based GPyTorch kernels."""

    has_lengthscale = False

    def __init__(
        self,
        kernel_type: str,
        cutoff: float | None,
        epsilon: float | None,
        batch_shape: Size,
        compile_pool: multiprocessing.pool.Pool | None = None,
    ):
        super().__init__(batch_shape=batch_shape)

        if cutoff is not None and epsilon is not None:
            raise ValueError("Cannot have both a fixed cutoff and an epsilon for dynamic cutoff!")

        self._kernel_type = kernel_type
        self._cutoff = cutoff
        self._epsilon = epsilon
        self._compile_pool = compile_pool

    @abc.abstractmethod
    def _get_params(self) -> Tensor:
        ...

    @abc.abstractmethod
    def _get_largest_lengthscale(self) -> float:
        ...

    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> KernelMatmulLinearOperator | Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("last_dim_is_batch not supported for KernelMatmulKernel")

        params = self._get_params()

        cutoff = None
        x1_eq_x2 = torch.equal(x1, x2)
        if self._cutoff is not None:
            cutoff = self._cutoff
        if self._epsilon is not None:
            cutoff = (
                math.sqrt(2)
                * torch.special.erfinv(torch.tensor(1 - self._epsilon)).item()
                * self._get_largest_lengthscale()
            )
        if x1_eq_x2:
            start, end = make_ranges(cutoff, x1.squeeze(-1), align=True)
        else:
            start, end = make_ranges(cutoff, x1.squeeze(-1), x2.squeeze(-1), align=True)

        operator = KernelMatmulLinearOperator(
            x1,
            x2,
            params,
            start,
            end,
            kernel_type=self._kernel_type,
            symmetric=x1_eq_x2,
            compile_pool=self._compile_pool,
        )

        if diag:
            return operator.diagonal()

        return operator
