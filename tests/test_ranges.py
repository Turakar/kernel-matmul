import math
import torch
import pytest
from torch import Tensor

from kernel_matmul import ranges


@pytest.fixture(params=[-1.0, 0.0, 1.5], ids=lambda x: f"offset_{x}")
def x_data(request) -> tuple[Tensor, Tensor]:
    offset = request.param
    batch = 2
    x1 = torch.sort(torch.rand(batch, 100) * 10)[0]
    x2 = torch.sort(torch.rand(batch, 120) * 10)[0] + offset
    return x1, x2


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
def test_ranges_asymmetric(cutoff: float, x_data: tuple[Tensor, Tensor]) -> None:
    block_size = 3
    x1, x2 = x_data
    start, end = ranges.make_ranges(cutoff, x1, x2, block_size=block_size)
    for batch in range(x1.shape[0]):
        rows = int(math.ceil(x1.shape[1] / block_size))
        for i in range(rows):
            block_from = i * block_size
            block_to = min(block_from + block_size, x1.shape[1]) - 1
            for j in range(x2.shape[1]):
                if x1[batch, block_from] - cutoff <= x2[batch, j] <= x1[batch, block_to] + cutoff:
                    assert start[batch, i] <= j
                    assert j < end[batch, i]
                else:
                    assert j < start[batch, i] or end[batch, i] <= j


def test_ranges_no_cutoff(x_data: tuple[Tensor, Tensor]) -> None:
    x1, x2 = x_data
    block_size = 3
    start, end = ranges.make_ranges(None, x1, x2, block_size=block_size)
    rows = int(math.ceil(x1.shape[1] / block_size))
    assert torch.all(start == 0)
    assert torch.all(end == x2.shape[1])
    assert start.shape == (x1.shape[0], rows)
    assert end.shape == (x1.shape[0], rows)


def test_ranges_no_cutoff_symmetric() -> None:
    x = torch.sort(torch.rand(2, 100) * 10)[0]
    block_size = 3
    start, end = ranges.make_ranges(None, x, block_size=block_size)
    rows = int(math.ceil(x.shape[1] / block_size))
    assert torch.all(start == 0)
    assert torch.all(end == x.shape[1])
    assert start.shape == (x.shape[0], rows)
    assert end.shape == (x.shape[0], rows)


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
def test_ranges_symmetric(cutoff: float) -> None:
    x = torch.sort(torch.rand(2, 100) * 10)[0]
    x[:, -1] = 11
    block_size = 3
    start, end = ranges.make_ranges(cutoff, x, block_size=block_size)
    for batch in range(x.shape[0]):
        mask = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.bool)
        for i in range(start.shape[1]):
            mask[i * block_size : (i + 1) * block_size, start[batch, i] : end[batch, i]] = True
        assert torch.all(mask == mask.T)
        for i in range(0, x.shape[1], block_size):
            row_last = min(i + block_size, x.shape[1]) - 1
            for j in range(i, x.shape[1], block_size):
                in_range = x[batch, j] <= x[batch, row_last] + cutoff
                assert torch.all(mask[i : i + block_size, j : j + block_size] == in_range)
