import abc
import itertools
from kernel_matmul.compile import Defines
from kernel_matmul import _BLOCK_SIZE


class Configuration(abc.ABC):
    @abc.abstractmethod
    def make_candidates(self, args: tuple) -> list[Defines]:
        ...

    @abc.abstractmethod
    def cache_key(self, args: tuple) -> str:
        ...


class SingleConfiguration(Configuration):
    def make_candidates(self, args: tuple) -> list[Defines]:
        return [self.make_config(args)]

    @abc.abstractmethod
    def make_config(self, args: tuple) -> Defines:
        ...


class MatmulSingleConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, rhs, params, start, end = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "MATMUL_K_BLOCK_SIZE": min(rhs.shape[-1], 32),
            get_kernel_type_define(self.kernel_type): None,
            "MATMUL_THREADS": 64,
            "MATMUL_PER_THREAD": 2,
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, rhs, params, start, end = args
        return f"{x1.dim() - 1}_{rhs.shape[-1]}"


class MatmulAutotuneConfiguration(Configuration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_candidates(self, args: tuple) -> list[Defines]:
        x1, x2, rhs, params, start, end = args
        fixed = {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            get_kernel_type_define(self.kernel_type): None,
        }
        configs = [
            dict(MATMUL_THREADS=32, MATMUL_PER_THREAD=4),
            dict(MATMUL_THREADS=64, MATMUL_PER_THREAD=2),
            dict(MATMUL_THREADS=128, MATMUL_PER_THREAD=1),
        ]
        k = rhs.shape[-1]
        if k <= 32:
            k_configs = [dict(MATMUL_K_BLOCK_SIZE=k)]
        else:
            k_configs = [dict(MATMUL_K_BLOCK_SIZE=x) for x in [8, 16, 32, 64]]
        return [
            fixed | config | k_config for config, k_config in itertools.product(configs, k_configs)
        ]

    def cache_key(self, args: tuple) -> str:
        x1, x2, rhs, params, start, end = args
        return f"{x1.dim() - 1}_{rhs.shape[-1]}"


class BilinearDerivativeConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, left_vectors, right_vectors, params, start, end = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            get_kernel_type_define(self.kernel_type): None,
            "BILINEAR_DERIVATIVE_THREAD_DIM": 16,
            "BILINEAR_DERIVATIVE_PER_THREAD": 8,
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, left_vectors, right_vectors, params, start, end = args
        return f"{x1.dim() - 1}"


class RowConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, row, params, start, end = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            get_kernel_type_define(self.kernel_type): None,
            "ROW_THREAD_DIM": 64,
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, row, params, start, end = args
        return f"{x1.dim() - 1}"


class DenseConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, params, start, end = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            get_kernel_type_define(self.kernel_type): None,
            "DENSE_THREAD_DIM": 16,
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, params, start, end = args
        return f"{x1.dim() - 1}"


class DenseBwdConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, params, start, end, out_grad = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            get_kernel_type_define(self.kernel_type): None,
            "DENSE_BWD_THREAD_DIM": 16,
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, params, start, end, out_grad = args
        return f"{x1.dim() - 1}"


class MatmulBwdConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, rhs, params, start, end, out_grad = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            get_kernel_type_define(self.kernel_type): None,
            "MATMUL_BWD_THREAD_DIM": 16,
            "MATMUL_BWD_PER_THREAD": 8,
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, rhs, params, start, end, out_grad = args
        return f"{x1.dim() - 1}"


def get_kernel_type_define(kernel_type: str) -> str:
    return {
        "rbf": "KERNEL_RBF",
        "spectral": "KERNEL_SPECTRAL",
        "locally_periodic": "KERNEL_LOCALLY_PERIODIC",
    }[kernel_type]
