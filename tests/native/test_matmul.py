from torch import Tensor
import torch
from kernel_matmul.compile import load_native
from kernel_matmul.configurations import MatmulSingleConfiguration
from tests.conftest import ExampleData


def test_matmul(
    example_data: ExampleData,
    reference_kernel: Tensor,
    build_type: bool,
) -> None:
    args = (
        example_data.x1,
        example_data.x2,
        example_data.rhs,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    config = MatmulSingleConfiguration(example_data.kernel_type)
    defines = config.make_config(args)
    print(defines)
    assert "MATMUL_COL_BLOCKS" in defines
    defines["MATMUL_COL_BLOCKS"] = 2
    native = load_native("matmul", defines)
    result = native.call(*args)
    reference = reference_kernel @ example_data.rhs
    print(example_data.params)
    assert torch.allclose(reference, result, atol=2e-4)
