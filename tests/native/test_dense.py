import torch
from torch import Tensor
from kernel_matmul.compile import load_native
from kernel_matmul.configurations import DenseConfiguration

from tests.conftest import ExampleData


def test_dense(example_data: ExampleData, reference_kernel: Tensor, build_type: bool) -> None:
    args = (
        example_data.x1,
        example_data.x2,
        example_data.params,
        example_data.start,
        example_data.end,
    )

    config = DenseConfiguration(example_data.kernel_type)
    defines = config.make_config(args)

    with torch.no_grad():
        native = load_native(
            name="dense",
            defines=defines,
        )
        result = native.call(*args)

    assert torch.allclose(reference_kernel, result, atol=2e-4, rtol=2e-4)
