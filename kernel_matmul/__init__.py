import os
from typing import Any
import warnings

import torch
from gpytorch.kernels.kernel import sq_dist
from torch import Tensor

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="pkg_resources is deprecated as an API"
    )
    import torch.utils.cpp_extension

_BLOCK_SIZE = 128


def load_native(
    name: str,
    defines: list[dict[str, Any]] | None = None,
) -> None:
    debug = os.environ.get("KERNEL_MATMUL_COMPILE_DEBUG", "false") == "true"
    print_sizes = os.environ.get("KERNEL_MATMUL_COMPILE_PRINT_SIZES", "false") == "true"
    verbose = os.environ.get("KERNEL_MATMUL_COMPILE_VERBOSE", "false") == "true"

    if defines is None:
        defines = {}

    source_ext = (".cpp", ".cu")
    common_path = os.path.join("native", "common")
    sources = [
        os.path.join(common_path, filename)
        for filename in os.listdir(common_path)
        if filename.endswith(source_ext)
    ]
    module_path = os.path.join("native", name)
    sources += [
        os.path.join(module_path, filename)
        for filename in os.listdir(module_path)
        if filename.endswith(source_ext)
    ]

    torch_name = f"km__{name}"
    flags = []
    cpp_flags = []
    cuda_flags = []
    for k, v in defines.items():
        if v is not None:
            flags.append(f"-DKM_{k}={v}")
            torch_name += f"__{k}__{v}"
        else:
            flags.append(f"-DKM_{k}")
            torch_name += f"__{k}"
    if debug:
        flags.append("-DKM_DEBUG_GPU_ASSERT")
        cpp_flags.append("-g")
        cuda_flags += ["-g", "-G"]
        torch_name += "__debug"
    else:
        cpp_flags.append("-O3")
    if print_sizes:
        flags.append("-DKM_DEBUG_PRINT_SIZES")
        torch_name += "__print_sizes"
    return torch.utils.cpp_extension.load(
        name=torch_name,
        sources=sources,
        extra_cflags=cpp_flags + flags,
        extra_cuda_cflags=cuda_flags + flags,
        verbose=verbose,
    )


def main():
    native = load_native(
        name="matmul",
        defines={
            "BLOCK_SIZE": _BLOCK_SIZE,
            "MATMUL_THREADS": 64,
            "MATMUL_PER_THREAD": 2,
            "MATMUL_K_BLOCK_SIZE": 11,
            "KERNEL_SPECTRAL": None,
        },
    )

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
    result = native.call(x1, x2, rhs, params, start, end)
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
