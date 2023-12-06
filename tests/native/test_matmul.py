from torch import Tensor
import torch
from kernel_matmul import _BLOCK_SIZE, load_native
from tests.conftest import ExampleData


def test_matmul(
    example_data: ExampleData, reference_kernel: Tensor, debug_build: bool, kernel_define: str
) -> None:
    native = load_native(
        name="matmul",
        defines={
            "BLOCK_SIZE": _BLOCK_SIZE,
            "MATMUL_THREADS": 64,
            "MATMUL_PER_THREAD": 2,
            "MATMUL_K_BLOCK_SIZE": example_data.rhs.shape[-1],
            kernel_define: None,
        },
    )
    result = native.call(
        example_data.x1,
        example_data.x2,
        example_data.rhs,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    reference = reference_kernel @ example_data.rhs
    assert torch.allclose(reference, result, atol=2e-4)
