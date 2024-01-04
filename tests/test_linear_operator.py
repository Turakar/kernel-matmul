import pytest
from kernel_matmul.linear_operator import KernelMatmulLinearOperator
from torch import Tensor

from linear_operator.test.linear_operator_test_case import RectangularLinearOperatorTestCase


@pytest.mark.usefixtures("example_data")
@pytest.mark.align()
@pytest.mark.flaky(reruns=3)
class TestKernelMatmulLinearOperator(RectangularLinearOperatorTestCase):
    def create_linear_op(self) -> KernelMatmulLinearOperator:
        return KernelMatmulLinearOperator(
            self.example_data.x1.unsqueeze(-1),
            self.example_data.x2.unsqueeze(-1),
            self.example_data.params,
            self.example_data.start,
            self.example_data.end,
            kernel_type=self.example_data.kernel_type,
        )

    def evaluate_linear_op(self, linear_op: KernelMatmulLinearOperator) -> Tensor:
        return linear_op.to_dense()
