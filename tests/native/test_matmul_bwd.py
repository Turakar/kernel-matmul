import torch
from torch import Tensor
from kernel_matmul.compile import load_native
from kernel_matmul.configurations import MatmulBwdConfiguration

from tests.conftest import ExampleData


def test_matmul_bwd(example_data: ExampleData, reference_kernel: Tensor, build_type: bool) -> None:
    x1 = example_data.x1
    x2 = example_data.x2
    rhs = example_data.rhs
    params = example_data.params
    start = example_data.start
    end = example_data.end

    out_grad = torch.randn(
        *params.shape[:-1],
        x1.shape[-1],
        rhs.shape[-1],
        device=x1.device,
        dtype=x1.dtype,
    )
    out_grad = out_grad / torch.linalg.norm(out_grad, dim=(-2, -1), keepdim=True)

    args = (
        x1,
        x2,
        rhs,
        params,
        start,
        end,
        out_grad,
    )

    (reference_kernel @ rhs).backward(gradient=out_grad)
    reference_grads = params.grad.clone()

    config = MatmulBwdConfiguration(example_data.kernel_type)
    defines = config.make_config(args)

    with torch.no_grad():
        native = load_native(
            name="matmul_bwd",
            defines=defines,
        )
        result = native.call(*args)

    assert torch.allclose(reference_grads, result, atol=2e-3, rtol=2e-3)
