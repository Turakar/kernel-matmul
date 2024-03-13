from gpytorch.kernels import Kernel
from torch import Tensor, Size
from linear_operator.operators import LinearOperator


class SumKernel(Kernel):
    has_lengthscale = False

    def __init__(self, base_kernel: Kernel, dim: int = -1):
        super().__init__()
        self.base_kernel = base_kernel
        self.dim = dim

    def forward(self, x1, x2, diag: bool = False, **params) -> tuple[Tensor, LinearOperator]:
        base = self.base_kernel.forward(x1, x2, diag=diag, **params)
        if self.dim >= 0:
            dim = self.dim
        elif diag:
            dim = self.dim - 1
        else:
            dim = self.dim - 2
        return base.sum(dim=dim)

    @property
    def batch_shape(self) -> Size:
        base_shape = list(self.base_kernel.batch_shape)
        del base_shape[self.dim]
        return Size(base_shape)
