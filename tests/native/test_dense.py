import pytest
from kernel_matmul import load_native, _BLOCK_SIZE
from torch import Tensor
from tests.conftest import ExampleData
import torch


@pytest.mark.square(True)
def test_diagonal(
    example_data: ExampleData, reference_kernel: Tensor, kernel_define: str, debug_build: bool
) -> None:
    native = load_native(
        name="dense",
        defines={
            "BLOCK_SIZE": _BLOCK_SIZE,
            "DENSE_THREAD_DIM": 64,
            kernel_define: None,
        },
    )
    result = native.call(
        example_data.x1,
        example_data.x2,
        -1,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    reference = reference_kernel.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(reference, result, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("row", [0, 133, 256])
def test_row(
    example_data: ExampleData,
    reference_kernel: Tensor,
    kernel_define: str,
    debug_build: bool,
    row: int,
) -> None:
    if row >= example_data.x1.shape[0]:
        pytest.skip("Row index out of bounds")
    native = load_native(
        name="dense",
        defines={
            "BLOCK_SIZE": _BLOCK_SIZE,
            "DENSE_THREAD_DIM": 64,
            kernel_define: None,
        },
    )
    result = native.call(
        example_data.x1,
        example_data.x2,
        row,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    reference = reference_kernel[:, row, :]
    assert torch.allclose(reference, result, atol=2e-4, rtol=2e-4)
