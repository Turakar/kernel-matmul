import multiprocessing
from kernel_matmul.gpytorch.base import KernelMatmulKernel


import torch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import Prior
from torch import Size, nn


class RBFKernelMatmulKernel(KernelMatmulKernel):
    """RBF kernel

    kernel_value = outputscale * exp(-0.5 * (x1 - x2)^2 / lengthscale^2)
    """

    def __init__(
        self,
        cutoff: float | None = None,
        epsilon: float | None = None,
        batch_shape: Size = torch.Size(()),
        lengthscale_constraint: Interval | None = None,
        lengthscale_prior: Prior | None = None,
        outputscale_constraint: Interval | None = None,
        outputscale_prior: Prior | None = None,
        compile_pool: multiprocessing.pool.Pool | None = None,
    ):
        super().__init__(
            "rbf",
            cutoff,
            epsilon,
            batch_shape,
            compile_pool=compile_pool,
        )

        if lengthscale_constraint is None:
            lengthscale_constraint = GreaterThan(1e-6)
        if outputscale_constraint is None:
            outputscale_constraint = GreaterThan(1e-6)

        self.raw_lengthscale = nn.Parameter(torch.zeros(batch_shape))
        self.register_constraint("raw_lengthscale", lengthscale_constraint)
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, x: m.initialize(lengthscale=x),
            )
        self.raw_outputscale = nn.Parameter(torch.zeros(batch_shape))
        self.register_constraint("raw_outputscale", outputscale_constraint)
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior",
                outputscale_prior,
                lambda m: m.outputscale,
                lambda m, x: m.initialize(outputscale=x),
            )

    @property
    def lengthscale(self) -> torch.Tensor:
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value: torch.Tensor) -> None:
        value = value.expand(self.raw_lengthscale.shape)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    @property
    def outputscale(self) -> torch.Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value: torch.Tensor) -> None:
        value = value.expand(self.raw_outputscale.shape)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def _get_params(self) -> torch.Tensor:
        return torch.stack([self.lengthscale, self.outputscale], dim=-1)

    def _get_largest_lengthscale(self) -> float:
        return self.lengthscale.max().item()
