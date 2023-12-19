import math
import torch
import pytest
from torch import Tensor

from kernel_matmul import ranges
from kernel_matmul.util import dict_product


@pytest.fixture(
    # params=[-1.0, 0.0, 1.5], ids=[]
    params=[
        pytest.param(param, id=f"example{i}")
        for i, param in enumerate(
            dict_product(
                [
                    dict(offset=-1.0),
                    dict(offset=0.0),
                    dict(offset=1.5),
                ],
                [
                    dict(gap_x1=None),
                    dict(gap_x1=0),
                    dict(gap_x1=10),
                ],
                [dict(gap_x2=None), dict(gap_x2=0), dict(gap_x2=10)],
            )
        )
    ],
)
def x_data(request) -> tuple[Tensor, Tensor]:
    offset = request.param["offset"]
    gap_x1 = request.param["gap_x1"]
    gap_x2 = request.param["gap_x2"]
    batch = 2
    x1 = torch.sort(torch.rand(batch, 100) * 10)[0]
    x2 = torch.sort(torch.rand(batch, 120) * 10)[0] + offset
    if gap_x1 is not None:
        x1[gap_x1:] += 20
    if gap_x2 is not None:
        x2[gap_x2:] += 20
    return x1, x2


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
@pytest.mark.parametrize("align", [True, False], ids=lambda x: "align" if x else "noalign")
@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_asymmetric(
    cutoff: float, align: bool, block_size: int, x_data: tuple[Tensor, Tensor]
) -> None:
    x1, x2 = x_data
    start, end = ranges.make_ranges(cutoff, x1, x2, block_size=block_size, align=align)
    step = block_size if align else 1
    assert torch.all(start >= 0)
    assert torch.all(start <= end)
    assert torch.all(end <= x2.shape[1])
    for batch in range(x1.shape[0]):
        rows = int(math.ceil(x1.shape[1] / block_size))
        for i in range(rows):
            block_from = i * block_size
            block_to = min(block_from + block_size, x1.shape[1]) - 1
            for j in range(0, x2.shape[1], step):
                if x1[batch, block_from] - cutoff <= x2[batch, j] <= x1[batch, block_to] + cutoff:
                    assert start[batch, i] <= j
                    assert j < end[batch, i]
                else:
                    assert j < start[batch, i] or end[batch, i] <= j


@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_no_cutoff(block_size: int, x_data: tuple[Tensor, Tensor]) -> None:
    x1, x2 = x_data
    start, end = ranges.make_ranges(None, x1, x2, block_size=block_size)
    rows = int(math.ceil(x1.shape[1] / block_size))
    assert torch.all(start == 0)
    assert torch.all(end == x2.shape[1])
    assert start.shape == (x1.shape[0], rows)
    assert end.shape == (x1.shape[0], rows)


@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_no_cutoff_symmetric(block_size: int) -> None:
    x = torch.sort(torch.rand(2, 100) * 10)[0]
    start, end = ranges.make_ranges(None, x, block_size=block_size)
    rows = int(math.ceil(x.shape[1] / block_size))
    assert torch.all(start == 0)
    assert torch.all(end == x.shape[1])
    assert start.shape == (x.shape[0], rows)
    assert end.shape == (x.shape[0], rows)


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_symmetric(cutoff: float, block_size: int) -> None:
    x = torch.sort(torch.rand(2, 100) * 10)[0]
    x[:, -1] = 11
    start, end = ranges.make_ranges(cutoff, x, block_size=block_size)
    assert torch.all(start >= 0)
    assert torch.all(start <= end)
    assert torch.all(end <= x.shape[1])
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


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_transpose(cutoff: float, block_size: int, x_data: tuple[Tensor, Tensor]) -> None:
    x1, x2 = x_data
    start, end = ranges.make_ranges(cutoff, x1, x2, block_size=block_size, align=True)
    start_t, end_t = ranges.transpose_ranges(
        start, end, x1.shape[1], x2.shape[1], block_size=block_size
    )
    assert torch.all(start_t >= 0), "start_t < 0"
    assert torch.all(start_t <= end_t), "start_t > end_t"
    assert torch.all(end_t <= x1.shape[1]), "end_t > x1.shape[1]"
    for batch in range(x1.shape[0]):
        mask = torch.zeros((x1.shape[1], x2.shape[1]), dtype=torch.bool)
        mask_t = torch.zeros((x2.shape[1], x1.shape[1]), dtype=torch.bool)
        for i in range(start.shape[1]):
            mask[i * block_size : (i + 1) * block_size, start[batch, i] : end[batch, i]] = True
        for i in range(start_t.shape[1]):
            mask_t[
                i * block_size : (i + 1) * block_size, start_t[batch, i] : end_t[batch, i]
            ] = True
        assert torch.all(mask == mask_t.T), "mask != mask_t.T"


def test_ranges_transpose_noalign() -> None:
    x1 = torch.tensor([[0.0, 0.0, 2.0]])
    x2 = torch.tensor([[0.0, 2.0, 2.0]])
    block_size = 2
    start, end = ranges.make_ranges(1.0, x1, x2, block_size=block_size, align=False)
    with pytest.raises(ValueError, match="Ranges must be block-aligned"):
        ranges.transpose_ranges(start, end, x1.shape[1], x2.shape[1], block_size=block_size)
