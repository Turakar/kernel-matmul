import pytest
from kernel_matmul.compile import load_native
from torch import Tensor
from kernel_matmul.configurations import RowConfiguration
from tests.conftest import ExampleData
import torch


@pytest.mark.square()
def test_diagonal(example_data: ExampleData, reference_kernel: Tensor, build_type: bool) -> None:
    args = (
        example_data.x1,
        example_data.x2,
        -1,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    config = RowConfiguration(example_data.kernel_type)
    defines = config.make_config(args)
    native = load_native(
        name="row",
        defines=defines,
    )
    result = native.call(*args)
    reference = reference_kernel.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(reference, result, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("row", [0, 133, 256])
def test_row(
    example_data: ExampleData,
    reference_kernel: Tensor,
    build_type: bool,
    row: int,
) -> None:
    if row >= example_data.x1.shape[-1]:
        pytest.skip("Row index out of bounds")
    args = (
        example_data.x1,
        example_data.x2,
        row,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    config = RowConfiguration(example_data.kernel_type)
    defines = config.make_config(args)
    native = load_native(
        name="row",
        defines=defines,
    )
    result = native.call(*args)
    reference = reference_kernel[..., row, :]
    assert torch.allclose(reference, result, atol=2e-4, rtol=2e-4)
