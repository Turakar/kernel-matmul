import multiprocessing
import multiprocessing.pool
import os
from typing import Any, Callable
import warnings
from contextlib import contextmanager

import torch


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="pkg_resources is deprecated as an API"
    )
    import torch.utils.cpp_extension


Defines = dict[str, Any]


_global_compile_pool: multiprocessing.pool.Pool | None = None


@contextmanager
def compile_pool(processes: int):
    """Context manager to create a compile worker pool.

    This opens a process-wide compile pool for the duration of the context.
    The global compile pool is then used for all compilations.

    Args:
        processes (int): Number of worker processes.

    Raises:
        RuntimeError: If a global compile pool is already activated.
    """
    global _global_compile_pool
    if _global_compile_pool is not None:
        raise RuntimeError("compile_pool is already active")
    _global_compile_pool = multiprocessing.get_context("spawn").Pool(processes)
    try:
        yield
    finally:
        _global_compile_pool.close()
        _global_compile_pool.join()
        _global_compile_pool = None


def load_native(
    name: str,
    defines: Defines | None = None,
) -> None:
    """Compiles a module if necessary and loads it.

    Each module and defines combination is compiled only once and cached on disk.

    Args:
        name (str): Name of the module (directory in native/).
        defines (Defines | None, optional): Preprocessor defines, as a dict. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Get compile options from environment variables
    debug = os.environ.get("KERNEL_MATMUL_COMPILE_DEBUG", "false") == "true"
    print_sizes = os.environ.get("KERNEL_MATMUL_COMPILE_PRINT_SIZE", "false") == "true"
    verbose = os.environ.get("KERNEL_MATMUL_COMPILE_VERBOSE", "false") == "true"

    # Default values
    if defines is None:
        defines = {}

    # Find sources
    source_ext = (".cpp", ".cu")
    common_path = get_native_module_path("common")
    sources = [
        os.path.join(common_path, filename)
        for filename in os.listdir(common_path)
        if filename.endswith(source_ext)
    ]
    module_path = get_native_module_path(name)
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


def get_native_module_path(name: str) -> str:
    """Get the path to a native module.

    Args:
        name (str): Name of the module.

    Returns:
        str: Path to the module.
    """
    return os.path.join(os.path.dirname(__file__), "..", "native", name)


def _precompile(name: str, candidate: Defines) -> None:
    """
    Compile a candidate if it is not already compiled.
    This method intentionally returns nothing to allow calling by multiprocessing.
    """
    load_native(name, defines=candidate)


def find_best(
    name: str,
    args: list[Any],
    candidates: list[Defines],
    return_timings: bool = False,
    num_measurements: int = 1,
    compile_pool: multiprocessing.pool.Pool | None = None,
) -> dict[str, Any] | tuple[dict[str, Any], list[float]]:
    """Autotuning of a native module.

    Mainly used for the KernelMatmul module.

    Args:
        name (str): Name of the module.
        args (list[Any]): Arguments to the module (e.g., x and RHS).
        candidates (list[Defines]): List of preprocessor defines sets to test.
        return_timings (bool, optional): Whether to return the timings. Defaults to False.
        num_measurements (int, optional): Number of measurements per candidate. Defaults to 1.
        compile_pool (multiprocessing.pool.Pool | None, optional): Compile pool to use.
            If None is given, it uses a global compile if it is open. Defaults to None.

    Returns:
        dict[str, Any] | tuple[dict[str, Any], list[float]]: If return_timings is False, the best
            candidate is returned. Otherwise, the best candidate and all timings are returned.
    """

    # Compile all candidates in parallel
    if compile_pool is None and _global_compile_pool is not None:
        compile_pool = _global_compile_pool
    if compile_pool is not None:
        results = []
        for candidate in candidates:
            results.append(
                compile_pool.apply_async(_precompile, (name, candidate), error_callback=print)
            )
        for result in results:
            # Join worker processes.
            # Without a timeout, KeyboardInterrupt is not propagated.
            # See https://stackoverflow.com/a/1408476/353337
            result.get(60 * 60 * 24)

    # Measure timings
    timings = []
    for candidate in candidates:
        module = load_native(name, defines=candidate)
        config_timings = cuda_time(lambda: module.call(*args), num_measurements=num_measurements)
        timings.append(sum(config_timings) / len(config_timings))

    # Return best candidate and maybe all timings
    best = candidates[timings.index(min(timings))]
    if return_timings:
        return best, timings
    else:
        return best


def cuda_time(fn: Callable, num_measurements: int | None = None) -> float | list[float]:
    """Measure the time of a CUDA kernel.

    Args:
        fn (Callable): Function to measure.
        num_measurements (int | None, optional): Number of measurements. Defaults to None.

    Returns:
        float | list[float]: Time in milliseconds.
    """
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
