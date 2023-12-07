from torch import Tensor
import torch
from kernel_matmul.native_function import NativeFunction
from kernel_matmul.configurations import MatmulAutotuneConfiguration, MatmulSingleConfiguration

from tests.conftest import ExampleData


def test_native_function_default(
    example_data: ExampleData,
    reference_kernel: Tensor,
    release_build: None,
) -> None:
    native = NativeFunction("matmul", MatmulSingleConfiguration(example_data.kernel_type))
    result = native(
        example_data.x1,
        example_data.x2,
        example_data.rhs,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    reference = reference_kernel @ example_data.rhs
    assert torch.allclose(reference, result, atol=2e-4)


def test_native_function_autotune(
    example_data: ExampleData,
    reference_kernel: Tensor,
    release_build: None,
) -> None:
    native = NativeFunction(
        "matmul", MatmulAutotuneConfiguration(example_data.kernel_type), verbose=True
    )
    result = native(
        example_data.x1,
        example_data.x2,
        example_data.rhs,
        example_data.params,
        example_data.start,
        example_data.end,
    )
    reference = reference_kernel @ example_data.rhs
    assert torch.allclose(reference, result, atol=2e-4)
