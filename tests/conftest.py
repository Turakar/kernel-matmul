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
from kernel_matmul.configurations import get_kernel_type_define
from kernel_matmul.util import dict_product


@dataclass
class ExampleData:
    x1: Tensor
    x2: Tensor
    rhs: Tensor
    params: Tensor
    start: Tensor
    end: Tensor
    kernel_type: str
    cutoff: float | None


@pytest.fixture(
    params=[
        pytest.param(param, id=f"example{i}")
        for i, param in enumerate(
            dict_product(
                [
                    dict(m=128, n=128, b=1, batch=1, k=1, cutoff=2.0),
                    dict(m=600, n=800, b=3, batch=1, k=11, cutoff=2.0),
                    dict(m=500, n=500, b=1, batch=3, k=11, cutoff=2.0),
                    dict(m=301, n=201, b=4, batch=3, k=11, cutoff=None),
                ],
                [
                    dict(kernel_type="rbf"),
                    dict(kernel_type="spectral"),
                    dict(kernel_type="locally_periodic"),
                ],
            )
        )
    ],
)
def example_data(request: pytest.FixtureRequest) -> ExampleData:
    square = request.node.get_closest_marker("square") is not None
    align = request.node.get_closest_marker("align") is not None
    m = request.param["m"]
    n = request.param["n"] if not square else m
    b = request.param["b"]
    batch = request.param["batch"]
    k = request.param["k"]
    cutoff = request.param["cutoff"]
    kernel_type = request.param["kernel_type"]

    device = torch.device("cuda:0")
    tkwargs = dict(device=device, dtype=torch.float32)

    x1 = torch.sort(torch.rand(batch, m, **tkwargs) * 10, dim=-1)[0]
    if square:
        x2 = x1
    else:
        x2 = torch.sort(torch.rand(batch, n, **tkwargs) * 10, dim=-1)[0]
    start, end = make_ranges(cutoff, x1, x2, align=align)
    rhs = torch.randn(batch, n, k, **tkwargs)
    rhs = rhs / torch.linalg.norm(rhs, dim=-2, keepdim=True)
    rhs = rhs.unsqueeze(1).expand(batch, b, n, k)

    if kernel_type == "rbf":
        lengthscale = 1 + torch.rand(batch, b, **tkwargs)
        outputscale = 1 + torch.rand(batch, b, **tkwargs)
        params = torch.stack([lengthscale, outputscale], dim=-1)
    elif kernel_type == "spectral":
        lengthscale = 1 + torch.rand(batch, b, **tkwargs)
        frequency = 0.5 + torch.rand(batch, b, **tkwargs) * 0.5
        outputscale = 1 + torch.rand(batch, b, **tkwargs)
        params = torch.stack([lengthscale, frequency, outputscale], dim=-1)
    elif kernel_type == "locally_periodic":
        lengthscale_rbf = 1 + torch.rand(batch, b, **tkwargs)
        lengthscale_periodic = 1 + torch.rand(batch, b, **tkwargs)
        period_length = 1 + torch.rand(batch, b, **tkwargs)
        outputscale = 1 + torch.rand(batch, b, **tkwargs)
        params = torch.stack(
            [lengthscale_rbf, lengthscale_periodic, period_length, outputscale], dim=-1
        )
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    params.requires_grad = True

    data = ExampleData(
        x1=x1,
        x2=x2,
        rhs=rhs,
        params=params,
        start=start,
        end=end,
        kernel_type=kernel_type,
        cutoff=cutoff,
    )
    if request.cls is not None:
        request.cls.example_data = data
    return data


@pytest.fixture
def reference_kernel(request: pytest.FixtureRequest, example_data: ExampleData) -> Tensor:
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
    batch_shape = torch.Size((params.shape[0], params.shape[1]))
    x1_ = example_data.x1[..., None, :, None]
    x2_ = example_data.x2[..., None, :, None]
    if kernel_type == "rbf":
        lengthscale = params[..., 0]
        outputscale = params[..., 1]
        gpytorch_kernel = ScaleKernel(RBFKernel(batch_shape=batch_shape), batch_shape=batch_shape)
        gpytorch_kernel.to(example_data.x1.device)
        with_grads(gpytorch_kernel, "outputscale", outputscale)
        with_grads(gpytorch_kernel.base_kernel, "lengthscale", lengthscale[..., None, None])
        kernel = gpytorch_kernel(x1_, x2_).to_dense()
    elif kernel_type == "spectral":
        lengthscale = params[..., 0]
        frequency = params[..., 1]
        outputscale = params[..., 2]
        gpytorch_kernel = SpectralMixtureKernel(num_mixtures=1, batch_shape=batch_shape)
        gpytorch_kernel.to(example_data.x1.device)
        with_grads(
            gpytorch_kernel,
            "mixture_scales",
            torch.sqrt(1 / (4 * torch.pi**2 * lengthscale[..., None, None, None] ** 2)),
        )
        with_grads(gpytorch_kernel, "mixture_means", frequency[..., None, None, None])
        with_grads(gpytorch_kernel, "mixture_weights", outputscale[..., None])
        kernel = gpytorch_kernel(x1_, x2_).to_dense()
    elif kernel_type == "locally_periodic":
        lengthscale_rbf = params[..., 0]
        lengthscale_periodic = params[..., 1]
        period_length = params[..., 2]
        outputscale = params[..., 3]
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
            gpytorch_kernel.base_kernel.kernels[0], "lengthscale", lengthscale_rbf[..., None, None]
        )
        with_grads(
            gpytorch_kernel.base_kernel.kernels[1],
            "lengthscale",
            lengthscale_periodic[..., None, None],
        )
        with_grads(
            gpytorch_kernel.base_kernel.kernels[1], "period_length", period_length[..., None, None]
        )
        kernel = gpytorch_kernel(x1_, x2_).to_dense()
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    masked_kernel = torch.zeros_like(kernel)
    for batch in range(example_data.start.shape[0]):
        for i in range(example_data.start.shape[1]):
            rows = slice(i * _BLOCK_SIZE, (i + 1) * _BLOCK_SIZE)
            columns = slice(example_data.start[batch, i], example_data.end[batch, i])
            masked_kernel[batch, :, rows, columns] = kernel[batch, :, rows, columns]

    if request.cls is not None:
        request.cls.reference_kernel = masked_kernel

    return masked_kernel


@pytest.fixture(params=[True, False], ids=["debug", "release"])
def build_type(request, monkeypatch: pytest.MonkeyPatch) -> bool:
    debug = request.param
    monkeypatch.setenv("KERNEL_MATMUL_COMPILE_DEBUG", "true" if debug else "false")
    return debug


@pytest.fixture
def release_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_MATMUL_COMPILE_DEBUG", "false")


@pytest.fixture
def kernel_define(example_data: ExampleData) -> str:
    return get_kernel_type_define(example_data.kernel_type)
