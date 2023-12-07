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
    Kernel,
)

from kernel_matmul.ranges import make_ranges
from kernel_matmul import _BLOCK_SIZE


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
        pytest.param(param, id=f"example{i}")
        for i, param in enumerate(
            [
                dict(m=128, n=128, b=1, k=1, cutoff=2.0, kernel_type="rbf"),
                dict(m=128, n=128, b=1, k=1, cutoff=2.0, kernel_type="spectral"),
                dict(m=128, n=128, b=1, k=1, cutoff=2.0, kernel_type="locally_periodic"),
                dict(m=1000, n=1200, b=5, k=11, cutoff=2.0, kernel_type="rbf"),
                dict(m=1000, n=1200, b=5, k=11, cutoff=2.0, kernel_type="spectral"),
                dict(m=1000, n=1200, b=5, k=11, cutoff=2.0, kernel_type="locally_periodic"),
                dict(m=401, n=301, b=10, k=11, cutoff=None, kernel_type="rbf"),
                dict(m=401, n=301, b=10, k=11, cutoff=None, kernel_type="spectral"),
                dict(m=401, n=301, b=10, k=11, cutoff=None, kernel_type="locally_periodic"),
            ]
        )
    ],
)
def example_data(request: pytest.FixtureRequest) -> ExampleData:
    square = request.node.get_closest_marker("square", False)
    m = request.param["m"]
    n = request.param["n"] if not square else m
    b = request.param["b"]
    k = request.param["k"]
    cutoff = request.param["cutoff"]
    kernel_type = request.param["kernel_type"]

    device = torch.device("cuda:0")
    x1 = torch.sort(torch.rand(m, device=device, dtype=torch.float32) * 10)[0]
    x2 = torch.sort(torch.rand(n, device=device, dtype=torch.float32) * 10)[0]
    rhs = torch.randn(n, k, device=device, dtype=torch.float32)
    rhs = rhs / torch.linalg.norm(rhs, dim=0, keepdim=True)

    start, end = make_ranges(cutoff, x1, x2)

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

    params.requires_grad = True

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
    def with_grads(kernel: Kernel, name: str, value: Tensor) -> None:
        raw_name = f"raw_{name}"
        constraint = kernel.constraint_for_parameter_name(raw_name)
        if constraint is not None:
            value = constraint.inverse_transform(value)
        assert getattr(kernel, raw_name).shape == value.shape, name
        delattr(kernel, raw_name)
        setattr(kernel, raw_name, value)

    kernel_type = example_data.kernel_type
    params = example_data.params
    batch_shape = example_data.params.shape[1:]
    if kernel_type == "rbf":
        lengthscale = params[0]
        outputscale = params[1]
        gpytorch_kernel = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)
        gpytorch_kernel.to(example_data.x1.device)
        with_grads(gpytorch_kernel, "outputscale", outputscale)
        with_grads(gpytorch_kernel.base_kernel, "lengthscale", lengthscale[:, None, None])
        kernel = gpytorch_kernel(example_data.x1[:, None], example_data.x2[:, None]).to_dense()
    elif kernel_type == "spectral":
        lengthscale = params[0]
        frequency = params[1]
        outputscale = params[2]
        gpytorch_kernel = SpectralMixtureKernel(num_mixtures=1, batch_shape=batch_shape)
        gpytorch_kernel.to(example_data.x1.device)
        with_grads(
            gpytorch_kernel,
            "mixture_scales",
            torch.sqrt(1 / (4 * torch.pi**2 * lengthscale[..., None, None, None] ** 2)),
        )
        with_grads(gpytorch_kernel, "mixture_means", frequency[..., None, None, None])
        with_grads(gpytorch_kernel, "mixture_weights", outputscale[..., None])
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
        with_grads(gpytorch_kernel, "outputscale", outputscale)
        with_grads(
            gpytorch_kernel.base_kernel.kernels[0], "lengthscale", lengthscale_rbf[:, None, None]
        )
        with_grads(
            gpytorch_kernel.base_kernel.kernels[1],
            "lengthscale",
            lengthscale_periodic[:, None, None],
        )
        with_grads(
            gpytorch_kernel.base_kernel.kernels[1], "period_length", period_length[:, None, None]
        )
        kernel = gpytorch_kernel(example_data.x1[:, None], example_data.x2[:, None]).to_dense()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    masked_kernel = torch.zeros_like(kernel)
    for i in range(example_data.start.shape[0]):
        rows = slice(i * _BLOCK_SIZE, (i + 1) * _BLOCK_SIZE)
        columns = slice(example_data.start[i], example_data.end[i])
        masked_kernel[..., rows, columns] = kernel[..., rows, columns]
    return masked_kernel


@pytest.fixture(params=[True, False], ids=["debug", "release"])
def debug_build(request, monkeypatch: pytest.MonkeyPatch) -> bool:
    debug = request.param
    monkeypatch.setenv("KERNEL_MATMUL_COMPILE_DEBUG", "true" if debug else "false")
    return debug


@pytest.fixture
def kernel_define(example_data: ExampleData) -> str:
    return {
        "rbf": "KERNEL_RBF",
        "spectral": "KERNEL_SPECTRAL",
        "locally_periodic": "KERNEL_LOCALLY_PERIODIC",
    }[example_data.kernel_type]
