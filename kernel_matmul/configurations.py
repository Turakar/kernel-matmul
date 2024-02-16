import abc
import itertools
from kernel_matmul.compile import Defines
from kernel_matmul import _BLOCK_SIZE
from kernel_matmul.util import dict_product
from typing import Any
import math
from torch import Tensor


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
            "MATMUL_THREADS": 64,
            "MATMUL_PER_THREAD": 2,
            "MATMUL_COL_BLOCKS": 1,
            "MATMUL_USE_SHM": 1,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, rhs, params, start, end = args
        return augment_cache_key(f"{x1.dim() - 1}_{rhs.shape[-1]}", self.kernel_type, params)


class MatmulAutotuneConfiguration(Configuration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_candidates(self, args: tuple) -> list[Defines]:
        x1, x2, rhs, params, start, end = args
        fixed = {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            **get_kernel_type_defines(self.kernel_type, params),
        }
        if self.kernel_type == "compact":
            num_orders = int(math.sqrt(params.shape[-1] - 0.75))
            fixed["KERNEL_COMPACT_NUM_ORDERS"] = num_orders
        configs = dict_product(
            [
                dict(MATMUL_THREADS=32, MATMUL_PER_THREAD=4),
                dict(MATMUL_THREADS=64, MATMUL_PER_THREAD=2),
                dict(MATMUL_THREADS=128, MATMUL_PER_THREAD=1),
            ],
            [
                dict(MATMUL_COL_BLOCKS=1),
                dict(MATMUL_COL_BLOCKS=4),
                dict(MATMUL_COL_BLOCKS=8),
                dict(MATMUL_COL_BLOCKS=16),
            ],
            [
                # dict(MATMUL_USE_SHM=0),
                dict(MATMUL_USE_SHM=1),
            ],
        )
        k = rhs.shape[-1]
        if k <= 32:
            k_configs = [dict(MATMUL_K_BLOCK_SIZE=k)]
        else:
            k_configs = [dict(MATMUL_K_BLOCK_SIZE=x) for x in [16, 32, 64]]
        return [
            fixed | config | k_config for config, k_config in itertools.product(configs, k_configs)
        ]

    def cache_key(self, args: tuple) -> str:
        x1, x2, rhs, params, start, end = args
        return augment_cache_key(f"{x1.dim() - 1}_{rhs.shape[-1]}", self.kernel_type, params)


class BilinearDerivativeConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, left_vectors, right_vectors, params, start, end = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "BILINEAR_DERIVATIVE_THREAD_DIM": 16,
            "BILINEAR_DERIVATIVE_PER_THREAD": 8,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, left_vectors, right_vectors, params, start, end = args
        return augment_cache_key(f"{x1.dim() - 1}", self.kernel_type, params)


class IndexConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, params, start, end, batch_indices, row_index, col_index = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "INDEX_THREAD_DIM": 64,
            "INDEX_BATCH_DIM": row_index.dim() - 1,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, params, start, end, batch_indices, row_index, col_index = args
        return augment_cache_key(f"{x1.dim() - 1}_{row_index.dim() - 1}", self.kernel_type, params)


class IndexBwdConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, params, start, end, batch_indices, row_index, col_index, out_grad = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "INDEX_BWD_THREAD_DIM": 64,
            "INDEX_BWD_BATCH_DIM": row_index.dim() - 1,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, params, start, end, batch_indices, row_index, col_index, out_grad = args
        return augment_cache_key(f"{x1.dim() - 1}_{row_index.dim() - 1}", self.kernel_type, params)


class DenseConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, params, start, end = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "DENSE_THREAD_DIM": 16,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, params, start, end = args
        return augment_cache_key(f"{x1.dim() - 1}", self.kernel_type, params)


class DenseBwdConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, params, start, end, out_grad = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "DENSE_BWD_THREAD_DIM": 16,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, params, start, end, out_grad = args
        return augment_cache_key(f"{x1.dim() - 1}", self.kernel_type, params)


class MatmulBwdConfiguration(SingleConfiguration):
    def __init__(self, kernel_type: str):
        self.kernel_type = kernel_type

    def make_config(self, args: tuple) -> Defines:
        x1, x2, rhs, params, start, end, out_grad = args
        return {
            "BLOCK_SIZE": _BLOCK_SIZE,
            "BATCH_DIM": x1.dim() - 1,
            "MATMUL_BWD_THREAD_DIM": 16,
            "MATMUL_BWD_PER_THREAD": 8,
            **get_kernel_type_defines(self.kernel_type, params),
        }

    def cache_key(self, args: tuple) -> str:
        x1, x2, rhs, params, start, end, out_grad = args
        return augment_cache_key(f"{x1.dim() - 1}", self.kernel_type, params)


def get_kernel_type_defines(kernel_type: str, params: Tensor) -> dict[str, Any]:
    kernel_type_define = {
        "rbf": "KERNEL_RBF",
        "spectral": "KERNEL_SPECTRAL",
        "locally_periodic": "KERNEL_LOCALLY_PERIODIC",
        "compact": "KERNEL_COMPACT",
    }[kernel_type]
    defines = {kernel_type_define: None}
    if kernel_type == "compact":
        num_orders = int(math.sqrt(params.shape[-1] - 0.75))
        defines.update(NUM_ORDERS=num_orders)
    return defines


def augment_cache_key(key: str, kernel_type: str, params: Tensor) -> str:
    if kernel_type == "compact":
        key += f"_{params.shape[-1]}"
    return key
