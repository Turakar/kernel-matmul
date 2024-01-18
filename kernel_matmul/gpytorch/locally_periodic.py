from kernel_matmul.gpytorch.base import KernelMatmulKernel


import torch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.priors import Prior
from torch import Size, Tensor, nn


class LocallyPeriodicKernelMatmulKernel(KernelMatmulKernel):
    def __init__(
        self,
        cutoff: float | None = None,
        epsilon: float | None = None,
        batch_shape: Size = torch.Size(()),
        lengthscale_rbf_constraint: Interval | None = GreaterThan(1e-6),
        lengthscale_rbf_prior: Prior | None = None,
        lengthscale_periodic_constraint: Interval | None = GreaterThan(1e-6),
        lengthscale_periodic_prior: Prior | None = None,
        outputscale_constraint: Interval | None = GreaterThan(1e-6),
        outputscale_prior: Prior | None = None,
        period_length_constraint: Interval | None = GreaterThan(1e-6),
        period_length_prior: Prior | None = None,
    ):
        super().__init__(
            "spectral",
            cutoff,
            epsilon,
            batch_shape,
        )

        self.raw_lengthscale_rbf = nn.Parameter(torch.zeros(batch_shape))
        self.register_constraint("raw_lengthscale_rbf", lengthscale_rbf_constraint)
        if lengthscale_rbf_prior is not None:
            self.register_prior(
                "lengthscale_rbf_prior",
                lengthscale_rbf_prior,
                lambda m: m.lengthscale_periodic,
                lambda m, x: m.initialize(lengthscale_rbf=x),
            )
        self.raw_lengthscale_periodic = nn.Parameter(torch.zeros(batch_shape))
        self.register_constraint("raw_lengthscale_periodic", lengthscale_periodic_constraint)
        if lengthscale_periodic_prior is not None:
            self.register_prior(
                "lengthscale_periodic_prior",
                lengthscale_periodic_prior,
                lambda m: m.lengthscale_periodic,
                lambda m, x: m.initialize(lengthscale_periodic=x),
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
        self.raw_period_length = nn.Parameter(torch.zeros(batch_shape))
        self.register_constraint("raw_period_length", period_length_constraint)
        if period_length_prior is not None:
            self.register_prior(
                "period_length_prior",
                period_length_prior,
                lambda m: m.period_length,
                lambda m, x: m.initialize(period_length=x),
            )

    @property
    def lengthscale_rbf(self) -> torch.Tensor:
        return self.raw_lengthscale_rbf_constraint.transform(self.raw_lengthscale_rbf)

    @lengthscale_rbf.setter
    def lengthscale_rbf(self, value: torch.Tensor) -> None:
        value = value.expand(self.raw_lengthscale_rbf.shape)
        self.initialize(
            raw_lengthscale_rbf=self.raw_lengthscale_rbf_constraint.inverse_transform(value)
        )

    @property
    def lengthscale_periodic(self) -> torch.Tensor:
        return self.raw_lengthscale_periodic_constraint.transform(self.raw_lengthscale_periodic)

    @lengthscale_periodic.setter
    def lengthscale_periodic(self, value: torch.Tensor) -> None:
        value = value.expand(self.raw_lengthscale_periodic.shape)
        self.initialize(
            raw_lengthscale_periodic=self.raw_lengthscale_periodic_constraint.inverse_transform(
                value
            )
        )

    @property
    def outputscale(self) -> torch.Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value: torch.Tensor) -> None:
        value = value.expand(self.raw_outputscale.shape)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    @property
    def period_length(self) -> torch.Tensor:
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value: torch.Tensor) -> None:
        value = value.expand(self.raw_period_length.shape)
        self.initialize(
            raw_period_length=self.raw_period_length_constraint.inverse_transform(value)
        )

    def _get_params(self) -> Tensor:
        return torch.stack(
            [self.lengthscale_rbf, self.lengthscale_periodic, self.period_length, self.outputscale],
            dim=-1,
        )

    def _get_largest_lengthscale(self) -> float:
        return max(
            self.lengthscale_rbf.max().item(),
            self.lengthscale_periodic.max().item(),
        )
