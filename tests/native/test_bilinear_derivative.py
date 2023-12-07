import torch
from torch import Tensor
from kernel_matmul.compile import load_native
from kernel_matmul.configurations import BilinearDerivativeConfiguration

from tests.conftest import ExampleData


def test_bilinear_derivative(
    example_data: ExampleData, reference_kernel: Tensor, kernel_define: str, build_type: bool
) -> None:
    x1 = example_data.x1
    x2 = example_data.x2
    params = example_data.params
    start = example_data.start
    end = example_data.end

    left_vectors = torch.randn(x1.shape[0], 5, device=x1.device)
    left_vectors = left_vectors / torch.linalg.norm(left_vectors)
    right_vectors = torch.randn(x2.shape[0], 5, device=x1.device)
    right_vectors = right_vectors / torch.linalg.norm(right_vectors)

    args = (
        x1,
        x2,
        left_vectors,
        right_vectors,
        params,
        start,
        end,
    )

    (
        left_vectors.T[:, None, :]  # d x 1 x m
        @ reference_kernel.sum(dim=0, keepdim=True)  # 1 x m x n
        @ right_vectors.T[:, :, None]  # d x n x 1
    ).sum().backward()
    reference_grads = params.grad.clone()

    config = BilinearDerivativeConfiguration(example_data.kernel_type)
    defines = config.make_config(args)

    with torch.no_grad():
        native = load_native(
            name="bilinear_derivative",
            defines=defines,
        )
        result = native.call(*args)

    assert torch.allclose(reference_grads, result, atol=2e-3, rtol=2e-3)
