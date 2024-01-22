import pytest
from kernel_matmul.compile import load_native
from torch import Tensor
from kernel_matmul.configurations import IndexConfiguration
from tests.conftest import ExampleData
import torch


@pytest.mark.square()
@pytest.mark.parametrize("shape", [(100,), (5, 101)])
def test_index(
    example_data: ExampleData, reference_kernel: Tensor, build_type: bool, shape: tuple[int, ...]
) -> None:
    shape = torch.Size(shape)
    batch_indices = tuple(
        torch.randint(0, size, shape, dtype=torch.int, device=reference_kernel.device)
        for size in reference_kernel.shape[:-2]
    )
    row_index = torch.randint(
        0, reference_kernel.shape[-2], shape, dtype=torch.int, device=reference_kernel.device
    )
    col_index = torch.randint(
        0, reference_kernel.shape[-1], shape, dtype=torch.int, device=reference_kernel.device
    )

    args = (
        example_data.x1,
        example_data.x2,
        example_data.params,
        example_data.start,
        example_data.end,
        batch_indices,
        row_index,
        col_index,
    )
    config = IndexConfiguration(example_data.kernel_type)
    defines = config.make_config(args)
    native = load_native(
        name="index",
        defines=defines,
    )
    result = native.call(*args)
    reference = reference_kernel[*batch_indices, row_index, col_index]
    assert torch.allclose(reference, result, atol=2e-4, rtol=2e-4)
