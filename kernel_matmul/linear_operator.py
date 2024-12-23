import multiprocessing
from typing import List, Tuple
import torch
from torch import Tensor
from linear_operator import LinearOperator
from torch import Size, dtype
from kernel_matmul import ranges
from kernel_matmul.configurations import (
    MatmulAutotuneConfiguration,
    MatmulBwdConfiguration,
    DenseConfiguration,
    DenseBwdConfiguration,
    BilinearDerivativeConfiguration,
    IndexConfiguration,
    IndexBwdConfiguration,
)
from linear_operator.utils.generic import _to_helper

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


class _IndexOperator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        x2,
        params,
        start,
        end,
        native_index,
        native_index_bwd,
        row_index,
        col_index,
        *batch_indices,
    ):
        ctx.save_for_backward(x1, x2, params, start, end, row_index, col_index, *batch_indices)
        ctx.native_index_bwd = native_index_bwd
        return native_index(x1, x2, params, start, end, batch_indices, row_index, col_index)

    @staticmethod
    def backward(ctx, grad_output):
        x1, x2, params, start, end, row_index, col_index = ctx.saved_tensors[:8]
        batch_indices = ctx.saved_tensors[8:]
        grad = ctx.native_index_bwd(
            x1, x2, params, start, end, batch_indices, row_index, col_index, grad_output
        )
        return None, None, grad, None, None, None, None, None, None, *[None for _ in batch_indices]


class KernelMatmulLinearOperator(LinearOperator):
    """LinearOperator implementation for KernelMatmul.

    A KernelMatmulLinearOperator is defined by the inputs x1 and x2, the kernel function and its
    parameters, and the start and end indices of the kernel function. The kernel function is
    evaluated for each pair of points in x1 and x2, and the result is a matrix of shape
    (..., M, N), where M is the number of points in x1 and N is the number of points in x2.
    """

    def __init__(
        self,
        x1: Tensor,
        x2: Tensor,
        params: Tensor,
        start: Tensor,
        end: Tensor,
        kernel_type: str | None = None,
        symmetric: bool | None = None,
        compile_pool: multiprocessing.pool.Pool | None = None,
    ):
        """Create a KernelMatmulLinearOperator.

        Args:
            x1 (Tensor): Input x1 of shape (..., M, 1).
            x2 (Tensor): Input x2 of shape (..., N, 1).
            params (Tensor): Parameters of the kernel function of shape (..., params).
            start (Tensor): Start indices of the sparsity pattern of shape (..., blocks).
            end (Tensor): End indices of the sparsity pattern of shape (..., blocks).
            kernel_type (str | None, optional): Kernel function name. Required.
            symmetric (bool | None, optional): Whether this kernel matrix is symmetric.
                Computed automatically if not provided.
            compile_pool (multiprocessing.pool.Pool | None, optional): Compile worker pool.
                Defaults to no multiprocessing or the global compile pool, if active.

        Raises:
            ValueError: For invalid arguments.
        """
        super().__init__(
            x1,
            x2,
            params,
            start,
            end,
            kernel_type=kernel_type,
            symmetric=symmetric,
            compile_pool=compile_pool,
        )

        if kernel_type is None:
            raise ValueError("kernel_type must be specified")
        if x1.dim() < 2 or x2.dim() < 2 or x1.size(-1) != 1 or x2.size(-1) != 1:
            raise ValueError("x1 and x2 must have shapes (..., M, 1) and (..., N, 1), respectively")

        if symmetric is None:
            symmetric = torch.equal(x1, x2)

        x1_batch_shape = x1.shape[:-2]
        x2_batch_shape = x2.shape[:-2]
        params_batch_shape = params.shape[:-1]
        start_batch_shape = start.shape[:-1]
        end_batch_shape = end.shape[:-1]
        batch_shape = torch.broadcast_shapes(
            x1_batch_shape, x2_batch_shape, params_batch_shape, start_batch_shape, end_batch_shape
        )
        x1 = x1.expand(*batch_shape, *x1.shape[-2:])
        x2 = x2.expand(*batch_shape, *x2.shape[-2:])
        params = params.expand(*batch_shape, *params.shape[-1:])
        start = start.expand(*batch_shape, *start.shape[-1:])
        end = end.expand(*batch_shape, *end.shape[-1:])

        self._x1 = x1
        self._x2 = x2
        self._params = params
        self._start = start
        self._end = end
        self._kernel_type = kernel_type
        self._batch_shape = batch_shape
        self._symmetric = symmetric

        self._native_matmul = NativeFunction(
            "matmul", MatmulAutotuneConfiguration(kernel_type), compile_pool=compile_pool
        )
        self._native_matmul_bwd = NativeFunction("matmul_bwd", MatmulBwdConfiguration(kernel_type))
        self._native_dense = NativeFunction("dense", DenseConfiguration(kernel_type))
        self._native_dense_bwd = NativeFunction("dense_bwd", DenseBwdConfiguration(kernel_type))
        self._native_bilinear_derivative = NativeFunction(
            "bilinear_derivative", BilinearDerivativeConfiguration(kernel_type)
        )
        self._native_index = NativeFunction("index", IndexConfiguration(kernel_type))
        self._native_index_bwd = NativeFunction("index_bwd", IndexBwdConfiguration(kernel_type))

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
            start_t, end_t = ranges.transpose_ranges(
                self._start, self._end, self._x1.size(-2), self._x2.size(-2)
            )
            return KernelMatmulLinearOperator(
                self._x2, self._x1, self._params, start_t, end_t, kernel_type=self._kernel_type
            )

    def to_dense(self) -> Tensor:
        x1, x2, params, start, end = self._get_args_for_native(self._batch_shape)
        result = _DenseOperator.apply(
            x1, x2, params, start, end, self._native_dense, self._native_dense_bwd
        )
        return result

    def _expand_batch(self: LinearOperator, batch_shape: Size | List[int]) -> LinearOperator:
        return KernelMatmulLinearOperator(
            self._x1.expand(*batch_shape, *self._x1.shape[-2:]),
            self._x2.expand(*batch_shape, *self._x2.shape[-2:]),
            self._params.expand(*batch_shape, *self._params.shape[-1:]),
            self._start.expand(*batch_shape, *self._start.shape[-1:]),
            self._end.expand(*batch_shape, *self._end.shape[-1:]),
            kernel_type=self._kernel_type,
        )

    def to(self: LinearOperator, *args, **kwargs) -> LinearOperator:
        device, dtype = _to_helper(*args, **kwargs)
        if (device is not None and device.type != "cuda") or (
            dtype is not None and dtype != torch.float32
        ):
            raise NotImplementedError("KernelMatmulLinearOperator only supports CUDA and float32")
        return super().to(*args, **kwargs)

    def type(self: LinearOperator, dtype: dtype) -> LinearOperator:
        if dtype != torch.float32:
            raise NotImplementedError("KernelMatmulLinearOperator only supports CUDA and float32")
        return super().type(dtype)

    def _bilinear_derivative(
        self, left_vecs: Tensor, right_vecs: Tensor
    ) -> Tuple[Tensor | None, ...]:
        # Determine broadcasted batch shape
        left_vecs_batch = left_vecs.shape[:-2]
        right_vecs_batch = right_vecs.shape[:-2]
        batch_shape = torch.broadcast_shapes(left_vecs_batch, right_vecs_batch, self._batch_shape)

        # Broadcast inputs
        left_vecs = left_vecs.expand(*batch_shape, *left_vecs.shape[-2:])
        right_vecs = right_vecs.expand(*batch_shape, *right_vecs.shape[-2:])
        x1, x2, params, start, end = self._get_args_for_native(batch_shape)

        # Calculate
        result = self._native_bilinear_derivative(x1, x2, left_vecs, right_vecs, params, start, end)

        # Reshape
        grad = result.reshape(*batch_shape, result.shape[-1])

        return None, None, grad, None, None

    def _get_indices(self, row_index: Tensor, col_index: Tensor, *batch_indices: Tensor) -> Tensor:
        x1, x2, params, start, end = self._get_args_for_native(self._batch_shape)
        shape = torch.broadcast_shapes(
            row_index.shape, col_index.shape, *(b.shape for b in batch_indices)
        )

        def prepare(index):
            return index.expand(*shape).to(device=x1.device, dtype=torch.int)

        return _IndexOperator.apply(
            x1,
            x2,
            params,
            start,
            end,
            self._native_index,
            self._native_index_bwd,
            prepare(row_index),
            prepare(col_index),
            *(prepare(b) for b in batch_indices),
        )
