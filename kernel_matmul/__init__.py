import os

import torch
import torch.utils.cpp_extension
from gpytorch.kernels.kernel import sq_dist
from torch import Tensor

_BLOCK_SIZE = 128


def load_extension(
    name: str,
    sources: list[str],
    extra_cflags: list[str] | None = None,
    extra_cuda_cflags: list[str] | None = None,
) -> None:
    if extra_cflags is None:
        extra_cflags = []
    if extra_cuda_cflags is None:
        extra_cuda_cflags = []
    debug = os.environ.get("KERNEL_MATMUL_COMPILE_DEBUG", "false") == "true"
    verbose = os.environ.get("KERNEL_MATMUL_COMPILE_VERBOSE", "false") == "true"
    if debug:
        extra_cflags.append("-g")
        extra_cuda_cflags += ["-g", "-G"]
    else:
        extra_cflags.append("-O3")
    return torch.utils.cpp_extension.load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=verbose,
    )


native = load_extension(
    name="kernel_matmul",
    sources=["native/kernel_matmul.cpp", "native/kernel_matmul.cu"],
    extra_cuda_cflags=[f"-DKERNEL_MATMUL_BLOCK_SIZE={_BLOCK_SIZE}"],
)


def main():
    m = 1000
    n = 1000
    b = 5
    k = 11
    kernel_type = "spectral"
    (
        x1,
        x2,
        rhs,
        params,
        start,
        end,
    ) = make_example_data(m, n, b, k, kernel_type)
    kernel = reference_kernel(x1, x2, params, start, end, kernel_type)
    result = native.kernel_matmul_spectral(x1, x2, rhs, params, start, end)
    reference = kernel @ rhs
    assert torch.allclose(reference, result, atol=2e-3)


def make_example_data(
    m: int, n: int, b: int, k: int, kernel_type: str
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    device = torch.device("cuda:0")
    x1 = torch.linspace(0, 10, m, device=device, dtype=torch.float32)
    x2 = torch.linspace(0, 10, n, device=device, dtype=torch.float32)
    rhs = torch.randn(n, k, device=device, dtype=torch.float32)
    start = torch.full(((m + _BLOCK_SIZE - 1) // _BLOCK_SIZE,), 0, device=device, dtype=torch.int32)
    end = torch.full(((m + _BLOCK_SIZE - 1) // _BLOCK_SIZE,), n, device=device, dtype=torch.int32)
    if kernel_type == "locally_periodic":
        lengthscale_rbf = 1 + torch.rand(b, device=device, dtype=torch.float32)
        lengthscale_periodic = 1 + torch.rand(b, device=device, dtype=torch.float32)
        period_length = 1 + torch.rand(b, device=device, dtype=torch.float32)
        outputscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        params = torch.stack(
            [lengthscale_rbf, lengthscale_periodic, period_length, outputscale], dim=0
        )
    elif kernel_type == "rbf":
        lengthscale_rbf = 1 + torch.rand(b, device=device, dtype=torch.float32)
        outputscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        params = torch.stack([lengthscale_rbf, outputscale], dim=0)
    elif kernel_type == "spectral":
        lengthscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        period_length = 1 + torch.rand(b, device=device, dtype=torch.float32)
        outputscale = 1 + torch.rand(b, device=device, dtype=torch.float32)
        params = torch.stack([lengthscale, 1 / period_length, outputscale], dim=0)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    params.requires_grad = True
    return (
        x1,
        x2,
        rhs,
        params,
        start,
        end,
    )


def reference_kernel(
    x1: Tensor,
    x2: Tensor,
    params: Tensor,
    start: Tensor,
    end: Tensor,
    kernel_type: str,
) -> Tensor:
    lengthscale_rbf = params[0] ** 2
    outputscale = params[-1]
    distance_sq = sq_dist(x1.unsqueeze(-1), x2.unsqueeze(-1))
    rbf = 0.5 * distance_sq.unsqueeze(0) / lengthscale_rbf[:, None, None]
    if kernel_type == "locally_periodic":
        lengthscale_periodic = params[1]
        period_length = params[2]
        distance = torch.sqrt(distance_sq.clamp_min(1e-30))
        periodic = (
            2
            * (torch.sin(distance.unsqueeze(0) / period_length[:, None, None]) ** 2)
            / lengthscale_periodic[:, None, None]
        )
        kernel = outputscale[:, None, None] * torch.exp(-(rbf + periodic))
    elif kernel_type == "rbf":
        kernel = outputscale[:, None, None] * torch.exp(-rbf)
    elif kernel_type == "spectral":
        frequency = params[1]
        distance = torch.sqrt(distance_sq.clamp_min(1e-30))
        cosine = torch.cos(2 * torch.pi * frequency[:, None, None] * distance.unsqueeze(0))
        kernel = outputscale[:, None, None] * torch.exp(-rbf) * cosine
    else:
        raise ValueError(f"Unknown kernel type {kernel_type}")
    start = torch.repeat_interleave(start, _BLOCK_SIZE)[: len(x1)]
    end = torch.repeat_interleave(end, _BLOCK_SIZE)[: len(x1)]
    for i, (s, e) in enumerate(zip(start, end)):
        kernel[:, i, :s] = 0
        kernel[:, i, e:] = 0
    return kernel


if __name__ == "__main__":
    main()
