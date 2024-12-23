from dataclasses import dataclass
import itertools
import random
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


@pytest.fixture(autouse=True)
def seed(request) -> int:
    rng = random.Random(f"{request.node.nodeid}_{getattr(request.node, 'execution_count', 0)}")
    seed = rng.randint(0, 0xFFFF_FFFF_FFFF_FFFF)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


@pytest.fixture(
    params=[
        pytest.param(param, id=f"example{i}")
        for i, param in enumerate(
            dict_product(
                [
                    dict(m=128, n=128, batch=(), unsqueeze=None, k=1, cutoff=2.0),
                    dict(m=600, n=800, batch=(), unsqueeze=None, k=11, cutoff=2.0),
                    dict(m=500, n=500, batch=(3,), unsqueeze=None, k=11, cutoff=2.0),
                    dict(m=301, n=201, batch=(3,), unsqueeze=0, k=11, cutoff=None),
                    dict(m=301, n=199, batch=(2, 3), unsqueeze=1, k=11, cutoff=2.0),
                    dict(m=129, n=130, batch=(), unsqueeze=None, k=65, cutoff=2.0),
                ],
                [
                    dict(kernel_type="rbf"),
                    dict(kernel_type="spectral"),
                    dict(kernel_type="locally_periodic"),
                    # Compact kernels do not work with matmul_bwd, for now.
                    # dict(kernel_type="compact"),
                ],
            )
        )
    ],
)
def example_data(request: pytest.FixtureRequest) -> ExampleData:
    square = request.node.get_closest_marker("square") is not None
    align = request.node.get_closest_marker("align") is not None
    stable = request.node.get_closest_marker("stable") is not None

    if stable and not square:
        pytest.skip("Stable test only implemented for square matrices")

    m = request.param["m"]
    n = request.param["n"] if not square else m
    batch = request.param["batch"]
    unsqueeze = request.param["unsqueeze"]
    k = request.param["k"]
    cutoff = request.param["cutoff"] if not stable else None
    kernel_type = request.param["kernel_type"]

    if unsqueeze is not None:
        squeeze_batch = (*batch[:unsqueeze], *batch[unsqueeze + 1 :])
    else:
        squeeze_batch = batch

    tkwargs = dict(device=torch.device("cuda:0"), dtype=torch.float32)

    if stable:
        x1 = torch.linspace(0, 10, m, **tkwargs).expand(*squeeze_batch, m)
        x2 = x1
    else:
        x1 = torch.sort(torch.rand(*squeeze_batch, m, **tkwargs) * 10, dim=-1)[0]
        if square:
            x2 = x1
        else:
            x2 = torch.sort(torch.rand(*squeeze_batch, n, **tkwargs) * 10, dim=-1)[0]

    if square:
        start, end = make_ranges(cutoff, x1, align=align)
    else:
        start, end = make_ranges(cutoff, x1, x2, align=align)

    rhs = torch.randn(*squeeze_batch, n, k, **tkwargs)
    rhs = rhs / torch.linalg.norm(rhs, dim=-2, keepdim=True)

    if unsqueeze is not None:
        x1 = x1.unsqueeze(unsqueeze).expand(*batch, m)
        x2 = x2.unsqueeze(unsqueeze).expand(*batch, n)
        start = start.unsqueeze(unsqueeze).expand(*batch, -1)
        end = end.unsqueeze(unsqueeze).expand(*batch, -1)
        rhs = rhs.unsqueeze(unsqueeze).expand(*batch, n, k)

    lengthscale_factor = 0.01 if stable else 1.0

    if kernel_type == "rbf":
        lengthscale = (1 + torch.rand(batch, **tkwargs)) * lengthscale_factor
        outputscale = 1 + torch.rand(batch, **tkwargs)
        params = torch.stack([lengthscale, outputscale], dim=-1)
    elif kernel_type == "spectral":
        lengthscale = (1 + torch.rand(batch, **tkwargs)) * lengthscale_factor
        frequency = 0.5 + torch.rand(batch, **tkwargs) * 0.5
        outputscale = 1 + torch.rand(batch, **tkwargs)
        params = torch.stack([lengthscale, frequency, outputscale], dim=-1)
    elif kernel_type == "locally_periodic":
        lengthscale_rbf = (1 + torch.rand(batch, **tkwargs)) * lengthscale_factor
        lengthscale_periodic = 1 + torch.rand(batch, **tkwargs) * lengthscale_factor
        period_length = 1 + torch.rand(batch, **tkwargs)
        outputscale = 1 + torch.rand(batch, **tkwargs)
        params = torch.stack(
            [lengthscale_rbf, lengthscale_periodic, period_length, outputscale], dim=-1
        )
    elif kernel_type == "compact":
        num_orders = 2
        cutoff = torch.rand((*batch, 1), **tkwargs) + 1.0
        orders = torch.stack(
            [torch.randint(1, 5, batch, **tkwargs), torch.randint(5, 10, batch, **tkwargs)], dim=-1
        )
        weights = torch.randn((*batch, num_orders, num_orders), **tkwargs)
        weights = weights @ weights.mT
        weights = weights / weights.norm(dim=(-2, -1), keepdim=True)
        params = torch.cat([cutoff, orders, weights.reshape(*batch, -1)], dim=-1)
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
    batch_shape = params.shape[:-1]
    x1_ = example_data.x1[..., None]
    x2_ = example_data.x2[..., None]
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
    elif kernel_type == "compact":
        num_orders = 2
        cutoff = params[..., 0]
        orders = params[..., 1 : 1 + num_orders].int()
        weights = params[..., 1 + num_orders :].reshape(*batch_shape, num_orders, num_orders)
        dist = example_data.x1[..., :, None] - example_data.x2[..., None, :]
        dist = (dist / cutoff[..., None, None]).abs().clamp(max=1.0)
        orders_plus = orders[..., None, None, :, None] + orders[..., None, None, None, :]
        orders_minus = orders[..., None, None, :, None] - orders[..., None, None, None, :]
        cos_term = torch.cos(torch.pi * orders_plus * dist[..., None, None])
        sinc_term = torch.special.sinc(orders_minus * (1 - dist)[..., None, None])
        base = cos_term * sinc_term * (1 - dist)[..., None, None]
        kernel = torch.einsum("...ij,...ji->...", weights[..., None, None, :, :], base)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    masked_kernel = torch.zeros_like(kernel)
    batch_indices = itertools.product(*[range(s) for s in batch_shape])
    for batch in batch_indices:
        for i in range(example_data.start.shape[-1]):
            rows = slice(i * _BLOCK_SIZE, (i + 1) * _BLOCK_SIZE)
            columns = slice(example_data.start[*batch, i], example_data.end[*batch, i])
            masked_kernel[*batch, rows, columns] = kernel[*batch, rows, columns]

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
