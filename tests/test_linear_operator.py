import unittest
import linear_operator
import pytest
import torch
from kernel_matmul.linear_operator import KernelMatmulLinearOperator
from torch import Tensor
from linear_operator.operators import AddedDiagLinearOperator, DiagLinearOperator

from linear_operator.test.linear_operator_test_case import (
    RectangularLinearOperatorTestCase,
    LinearOperatorTestCase,
)


@pytest.mark.usefixtures("example_data")
@pytest.mark.align()
@pytest.mark.flaky(reruns=3)
class TestKernelMatmulLinearOperatorRectangular(RectangularLinearOperatorTestCase):
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


@pytest.mark.usefixtures("example_data")
@pytest.mark.align()
@pytest.mark.square()
@pytest.mark.stable()
@pytest.mark.flaky(reruns=3)
class TestKernelMatmulLinearOperatorSquare(LinearOperatorTestCase):
    @property
    def tolerances(self):
        tolerances = super().tolerances
        return tolerances | {
            "root_decomposition": dict(atol=0.01, rtol=0.05),
            "diagonalization": dict(atol=0.01, rtol=0.05),
            "sqrt_inv_matmul": dict(atol=5e-3, rtol=1e-2),
        }

    def assertTrue(self, *args, **kwargs):
        if not hasattr(self, "mock_test_case"):
            self.mock_test_case = unittest.TestCase()
        return self.mock_test_case.assertTrue(*args, **kwargs)

    def assertFalse(self, *args, **kwargs):
        if not hasattr(self, "mock_test_case"):
            self.mock_test_case = unittest.TestCase()
        return self.mock_test_case.assertFalse(*args, **kwargs)

    def assertRaises(self, *args, **kwargs):
        if not hasattr(self, "mock_test_case"):
            self.mock_test_case = unittest.TestCase()
        return self.mock_test_case.assertRaises(*args, **kwargs)

    def assertRaisesRegex(self, *args, **kwargs):
        if not hasattr(self, "mock_test_case"):
            self.mock_test_case = unittest.TestCase()
        return self.mock_test_case.assertRaisesRegex(*args, **kwargs)

    def assertWarnsRegex(self, *args, **kwargs):
        if not hasattr(self, "mock_test_case"):
            self.mock_test_case = unittest.TestCase()
        return self.mock_test_case.assertWarnsRegex(*args, **kwargs)

    def assertLess(self, *args, **kwargs):
        if not hasattr(self, "mock_test_case"):
            self.mock_test_case = unittest.TestCase()
        return self.mock_test_case.assertLess(*args, **kwargs)

    def create_linear_op(self) -> KernelMatmulLinearOperator:
        km = KernelMatmulLinearOperator(
            self.example_data.x1.unsqueeze(-1),
            self.example_data.x2.unsqueeze(-1),
            self.example_data.params,
            self.example_data.start,
            self.example_data.end,
            kernel_type=self.example_data.kernel_type,
        )
        noise = (
            torch.rand(
                (
                    *self.example_data.x1.shape[:-1],
                    km.size(-1),
                ),
                dtype=km.dtype,
                device=km.device,
            )
            * 0.5
            + 0.75
        )
        linear_op = AddedDiagLinearOperator(km, DiagLinearOperator(noise))
        return linear_op

    def evaluate_linear_op(self, linear_op: AddedDiagLinearOperator) -> Tensor:
        km = linear_op.linear_ops[0]
        diagonal = linear_op.linear_ops[1]._diag
        evaluated = km.to_dense() + torch.diag_embed(diagonal)
        return evaluated

    def test_to_double(self) -> None:
        with pytest.raises(RuntimeError, match="Could not convert <.*> to double"):
            return super().test_to_double()

    def test_double(self):
        with pytest.raises(
            NotImplementedError, match="KernelMatmulLinearOperator only supports CUDA and float32"
        ):
            return super().test_double()

    def test_float(self) -> None:
        pytest.skip("KernelMatmulLinearOperator only supports CUDA and float32")

    def test_half(self):
        with pytest.raises(
            NotImplementedError, match="KernelMatmulLinearOperator only supports CUDA and float32"
        ):
            return super().test_half()

    def test_root_decomposition(self, cholesky=False):
        with linear_operator.settings.max_root_decomposition_size(1000):
            return super().test_root_decomposition(cholesky)

    def test_diagonalization(self, symeig=False):
        with linear_operator.settings.max_root_decomposition_size(1000):
            return super().test_diagonalization(symeig)
