from typing import List
import torch
from torch import Tensor
from linear_operator import LinearOperator
from torch._C import Size
from kernel_matmul import ranges
from kernel_matmul.configurations import (
    MatmulSingleConfiguration,
    MatmulBwdConfiguration,
    DenseConfiguration,
    DenseBwdConfiguration,
)

from kernel_matmul.native_function import NativeFunction


class _MatmulOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, rhs, params, start, end, native_matmul, native_matmul_bwd):
        ctx.save_for_backward(x1, x2, rhs, params, start, end)
        ctx.native_matmul_bwd = native_matmul_bwd
        return native_matmul(x1, x2, rhs, params, start, end)

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, rhs, params, start, end = ctx.saved_tensors
        grad = ctx.native_matmul_bwd(x1, x2, rhs, params, start, end, grad_output)
        return None, None, None, grad, None, None, None, None


class _DenseOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, params, start, end, native_dense, native_dense_bwd):
        ctx.save_for_backward(x1, x2, params, start, end)
        ctx.native_dense_bwd = native_dense_bwd
        return native_dense(x1, x2, params, start, end)

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, params, start, end = ctx.saved_tensors
        grad = ctx.native_dense_bwd(x1, x2, params, start, end, grad_output)
        return None, None, grad, None, None, None, None


class KernelMatmulLinearOperator(LinearOperator):
    def __init__(
        self,
        x1: Tensor,
        x2: Tensor,
        params: Tensor,
        start: Tensor,
        end: Tensor,
        kernel_type: str | None = None,
    ):
        super().__init__(x1, x2, params, start, end, kernel_type=kernel_type)

        if kernel_type is None:
            raise ValueError("kernel_type must be specified")
        if x1.dim() < 2 or x2.dim() < 2 or x1.size(-1) != 1 or x2.size(-1) != 1:
            raise ValueError("x1 and x2 must have shapes (..., M, 1) and (..., N, 1), respectively")

        symmetric = torch.equal(x1, x2)

        x1_batch_shape = x1.shape[:-2]
        x2_batch_shape = x2.shape[:-2]
        start_batch_shape = start.shape[:-1]
        end_batch_shape = end.shape[:-1]
        batch_shape = torch.broadcast_shapes(
            x1_batch_shape, x2_batch_shape, start_batch_shape, end_batch_shape
        )
        x1 = x1.expand(*batch_shape, *x1.shape[-2:])
        x2 = x2.expand(*batch_shape, *x2.shape[-2:])
        start = start.expand(*batch_shape, *start.shape[-1:])
        end = end.expand(*batch_shape, *end.shape[-1:])

        self._x1 = x1
        self._x2 = x2
        self._params = params
        self._start = start
        self._end = end
        self._kernel_type = kernel_type
        self._batch_shape = x1_batch_shape
        self._symmetric = symmetric

        self._native_matmul = NativeFunction("matmul", MatmulSingleConfiguration(kernel_type))
        self._native_matmul_bwd = NativeFunction("matmul_bwd", MatmulBwdConfiguration(kernel_type))
        self._native_dense = NativeFunction("dense", DenseConfiguration(kernel_type))
        self._native_dense_bwd = NativeFunction("dense_bwd", DenseBwdConfiguration(kernel_type))

    def _get_args_for_native(
        self, batch_shape: torch.Size
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        def transform(x: Tensor, non_batch_dims: int) -> Tensor:
            non_batch_shape = x.shape[-non_batch_dims:]
            return x.expand(*batch_shape, *non_batch_shape)

        return (
            transform(self._x1.squeeze(-1), 1),
            transform(self._x2.squeeze(-1), 1),
            transform(self._params, 1),
            transform(self._start, 1),
            transform(self._end, 1),
        )

    def _matmul(self, rhs: Tensor) -> Tensor:
        # Determine broadcasted batch shape
        non_batch_shape = torch.Size((self._x2.size(-2), rhs.size(-1)))
        expected_shape = torch.Size((*self._batch_shape, *non_batch_shape))
        broadcast = torch.broadcast_shapes(expected_shape, rhs.shape)
        batch_shape = broadcast[: -len(non_batch_shape)]

        # Broadcast inputs
        rhs_ = rhs.expand(*broadcast)
        x1, x2, params, start, end = self._get_args_for_native(batch_shape)

        # Calculate
        result = _MatmulOperator.apply(
            x1, x2, rhs_, params, start, end, self._native_matmul, self._native_matmul_bwd
        )

        # Reshape
        return result.reshape(*batch_shape, *result.shape[-2:])

    def _size(self) -> torch.Size:
        return torch.Size((*self._batch_shape, self._x1.size(-2), self._x2.size(-2)))

    def _transpose_nonbatch(self) -> LinearOperator:
        if self._symmetric:
            return self
        else:
            x1, x2, _, start, end = self._get_args_for_native(self._batch_shape)
            start, end = ranges.transpose_ranges(start, end, x1.size(-1), x2.size(-1))
            start = start.reshape(*self._batch_shape, -1)
            end = end.reshape(*self._batch_shape, -1)
            return KernelMatmulLinearOperator(
                self._x2, self._x1, self._params, start, end, kernel_type=self._kernel_type
            )

    def to_dense(self) -> Tensor:
        x1, x2, params, start, end = self._get_args_for_native(self._batch_shape)
        result = _DenseOperator.apply(
            x1, x2, params, start, end, self._native_dense, self._native_dense_bwd
        )
        return result.reshape(*self.shape)

    def _expand_batch(self: LinearOperator, batch_shape: Size | List[int]) -> LinearOperator:
        return KernelMatmulLinearOperator(
            self._x1.expand(*batch_shape, *self._x1.shape[-2:]),
            self._x2.expand(*batch_shape, *self._x2.shape[-2:]),
            self._params.expand(*batch_shape, *self._params.shape[-1:]),
            self._start.expand(*batch_shape, *self._start.shape[-1:]),
            self._end.expand(*batch_shape, *self._end.shape[-1:]),
            kernel_type=self._kernel_type,
        )