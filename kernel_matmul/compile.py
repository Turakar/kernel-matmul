import os
from typing import Any, Callable
import warnings

import torch


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="pkg_resources is deprecated as an API"
    )
    import torch.utils.cpp_extension


Defines = dict[str, Any]


def load_native(
    name: str,
    defines: Defines | None = None,
) -> None:
    # Get compile options from environment variables
    debug = os.environ.get("KERNEL_MATMUL_COMPILE_DEBUG", "false") == "true"
    print_sizes = os.environ.get("KERNEL_MATMUL_COMPILE_PRINT_SIZE", "false") == "true"
    verbose = os.environ.get("KERNEL_MATMUL_COMPILE_VERBOSE", "false") == "true"

    # Default values
    if defines is None:
        defines = {}

    # Find sources
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

    # Determine compilation arguments
    torch_name = f"km___{name}"
    flags = []
    cpp_flags = []
    cuda_flags = []

    # Add preprocessor defines
    define_order = sorted(list(defines.keys()))
    for k in define_order:
        v = defines[k]
        if v is not None:
            flags.append(f"-DKM_{k}={v}")
            torch_name += f"___{k}__{v}"
        else:
            flags.append(f"-DKM_{k}")
            torch_name += f"___{k}"

    # Add additional flags
    if debug:
        flags.append("-DKM_DEBUG_GPU_ASSERT")
        cpp_flags.append("-g")
        cuda_flags += ["-g", "-G"]
        torch_name += "___debug"
    else:
        cpp_flags.append("-O3")
    if print_sizes:
        flags.append("-DKM_DEBUG_PRINT_SIZE")
        torch_name += "___print_size"

    # Compile and load
    return torch.utils.cpp_extension.load(
        name=torch_name,
        sources=sources,
        extra_cflags=cpp_flags + flags,
        extra_cuda_cflags=cuda_flags + flags,
        verbose=verbose,
    )


def find_best(
    name: str,
    args: list[Any],
    candidates: list[Defines],
    return_timings: bool = False,
    num_measurements: int = 1,
) -> dict[str, Any] | tuple[dict[str, Any], list[float]]:
    timings = []
    for candidate in candidates:
        module = load_native(name, defines=candidate)
        config_timings = cuda_time(lambda: module.call(*args), num_measurements=num_measurements)
        timings.append(sum(config_timings) / len(config_timings))
    best = candidates[timings.index(min(timings))]
    if return_timings:
        return best, timings
    else:
        return best


def cuda_time(fn: Callable, num_measurements: int | None = None) -> float | list[float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    single = num_measurements is None
    if single:
        num_measurements = 1
    timings = []
    for _ in range(num_measurements):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    if single:
        return timings[0]
    else:
        return timings
