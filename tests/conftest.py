from dataclasses import dataclass
import pytest
from torch import Tensor
import torch
from gpytorch.kernels import (
    ScaleKernel,
    RBFKernel,
    SpectralMixtureKernel,
    ProductKernel,
    PeriodicKernel,
)

from kernel_matmul.make_ranges import make_ranges_block


@dataclass
class ExampleData:
    x1: Tensor
    x2: Tensor
    rhs: Tensor
    params: Tensor
    start: Tensor
    end: Tensor
    kernel_type: str


@pytest.fixture(
    params=[
        dict(m=128, n=128, b=1, k=1, cutoff=2.0, kernel_type="rbf"),
        dict(m=128, n=128, b=1, k=1, cutoff=2.0, kernel_type="spectral"),
        dict(m=128, n=128, b=1, k=1, cutoff=2.0, kernel_type="locally_periodic"),
    ]
)
def example_data(request) -> ExampleData:
    m = request.param["m"]
    n = request.param["n"]
    b = request.param["b"]
    k = request.param["k"]
    cutoff = request.param["cutoff"]
    kernel_type = request.param["kernel_type"]

    device = torch.device("cuda:0")
    x1 = torch.sort(torch.rand(m, device=device, dtype=torch.float32) * 10)[0]
    x2 = torch.sort(torch.rand(n, device=device, dtype=torch.float32) * 10)[0]
    rhs = torch.randn(n, k, device=device, dtype=torch.float32)
    rhs = rhs / torch.linalg.norm(rhs, dim=0, keepdim=True)

    start, end = make_ranges_block(cutoff, x1, x2)

    if kernel_type == "rbf":
        lengthscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        outputscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        params = torch.stack([lengthscale, outputscale], dim=0)
    elif kernel_type == "spectral":
        lengthscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        frequency = 0.5 + torch.rand(b, device=device, dtype=torch.float32) * 0.5
        outputscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        params = torch.stack([lengthscale, frequency, outputscale], dim=0)
    elif kernel_type == "locally_periodic":
        lengthscale_rbf = 1 + torch.rand(b, device=device, dtype=torch.float32)
        lengthscale_periodic = 1 + torch.rand(b, device=device, dtype=torch.float32)
        period_length = 1 + torch.rand(b, device=device, dtype=torch.float32)
        outputscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        params = torch.stack(
            [lengthscale_rbf, lengthscale_periodic, period_length, outputscale], dim=0
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return ExampleData(
        x1=x1,
        x2=x2,
        rhs=rhs,
        params=params,
        start=start,
        end=end,
        kernel_type=kernel_type,
    )


@pytest.fixture
def reference_kernel(example_data: ExampleData) -> Tensor:
    kernel_type = example_data.kernel_type
    params = example_data.params
    batch_shape = example_data.params.shape[1:]
    if kernel_type == "rbf":
        lengthscale = params[0]
        outputscale = params[1]
        gpytorch_kernel = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)
        gpytorch_kernel.to(example_data.x1.device)
        gpytorch_kernel.outputscale = outputscale
        gpytorch_kernel.base_kernel.lengthscale = lengthscale
        kernel = gpytorch_kernel(example_data.x1[:, None], example_data.x2[:, None]).to_dense()
    elif kernel_type == "spectral":
        lengthscale = params[0]
        frequency = params[1]
        outputscale = params[2]
        gpytorch_kernel = SpectralMixtureKernel(num_mixtures=1, batch_shape=batch_shape)
        gpytorch_kernel.to(example_data.x1.device)
        gpytorch_kernel.mixture_scales = torch.sqrt(
            1 / (4 * torch.pi**2 * lengthscale[..., None, None, None] ** 2)
        )
        gpytorch_kernel.mixture_means = frequency[..., None, None, None]
        gpytorch_kernel.mixture_weights = outputscale[..., None]
        kernel = gpytorch_kernel(example_data.x1[:, None], example_data.x2[:, None]).to_dense()
    elif kernel_type == "locally_periodic":
        lengthscale_rbf = params[0]
        lengthscale_periodic = params[1]
        period_length = params[2]
        outputscale = params[3]
        gpytorch_kernel = ScaleKernel(
            ProductKernel(
                RBFKernel(batch_shape=batch_shape),
                PeriodicKernel(batch_shape=batch_shape),
            ),
            batch_shape=batch_shape,
        )
        gpytorch_kernel.to(example_data.x1.device)
        gpytorch_kernel.outputscale = outputscale
        gpytorch_kernel.base_kernel.kernels[0].lengthscale = lengthscale_rbf
        gpytorch_kernel.base_kernel.kernels[1].lengthscale = lengthscale_periodic
        gpytorch_kernel.base_kernel.kernels[1].period_length = period_length
        kernel = gpytorch_kernel(example_data.x1[:, None], example_data.x2[:, None]).to_dense()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    return kernel


@pytest.fixture
def debug_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_MATMUL_COMPILE_DEBUG", "true")
