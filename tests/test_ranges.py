import math
import torch
import pytest
from torch import Tensor

from kernel_matmul import ranges
from kernel_matmul.util import dict_product


@pytest.fixture(
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
                [
                    dict(batch_shape=()),
                    dict(batch_shape=(3,)),
                    dict(batch_shape=(2, 3)),
                ],
            )
        )
    ],
)
def x_data(request) -> tuple[Tensor, Tensor]:
    offset = request.param["offset"]
    gap_x1 = request.param["gap_x1"]
    gap_x2 = request.param["gap_x2"]
    batch_shape = request.param["batch_shape"]
    x1 = torch.sort(torch.rand(*batch_shape, 50) * 10)[0]
    x2 = torch.sort(torch.rand(*batch_shape, 55) * 10)[0] + offset
    if gap_x1 is not None:
        x1[..., gap_x1:] += 20
    if gap_x2 is not None:
        x2[..., gap_x2:] += 20
    return x1, x2


@pytest.fixture(
    params=[
        pytest.param(param, id=f"example{i}")
        for i, param in enumerate(
            dict_product(
                [
                    dict(gap_x1=None),
                    dict(gap_x1=0),
                    dict(gap_x1=10),
                ],
                [
                    dict(batch_shape=()),
                    dict(batch_shape=(3,)),
                    dict(batch_shape=(2, 3)),
                ],
            )
        )
    ],
)
def x_data_symmetric(request) -> Tensor:
    gap_x1 = request.param["gap_x1"]
    batch_shape = request.param["batch_shape"]
    x = torch.sort(torch.rand(*batch_shape, 50) * 10)[0]
    if gap_x1 is not None:
        x[..., gap_x1:] += 20
    return x


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
    assert torch.all(end <= x2.shape[-1])
    assert start.shape == end.shape
    assert start.shape[:-1] == x1.shape[:-1]

    if align:
        assert torch.all(torch.logical_or(start % block_size == 0, start == x2.shape[-1]))
        assert torch.all(torch.logical_or(end % block_size == 0, end == x2.shape[-1]))

    x1_ = x1.reshape(-1, x1.shape[-1])
    x2_ = x2.reshape(-1, x2.shape[-1])
    start_ = start.reshape(-1, start.shape[-1])
    end_ = end.reshape(-1, end.shape[-1])
    for batch in range(x1_.shape[0]):
        rows = int(math.ceil(x1_.shape[1] / block_size))
        for i in range(rows):
            block_from = i * block_size
            block_to = min(block_from + block_size, x1_.shape[1]) - 1
            for j in range(0, x2_.shape[1], step):
                if (
                    x1_[batch, block_from] - cutoff
                    <= x2_[batch, j]
                    <= x1_[batch, block_to] + cutoff
                ):
                    assert start_[batch, i] <= j
                    assert j < end_[batch, i]
                else:
                    assert j < start_[batch, i] or end_[batch, i] <= j


@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_no_cutoff(block_size: int, x_data: tuple[Tensor, Tensor]) -> None:
    x1, x2 = x_data
    start, end = ranges.make_ranges(None, x1, x2, block_size=block_size)
    rows = int(math.ceil(x1.shape[-1] / block_size))
    assert torch.all(start == 0)
    assert torch.all(end == x2.shape[-1])
    assert start.shape == (*x1.shape[:-1], rows)
    assert end.shape == (*x1.shape[:-1], rows)


@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_no_cutoff_symmetric(x_data_symmetric: Tensor, block_size: int) -> None:
    x = x_data_symmetric
    start, end = ranges.make_ranges(None, x, block_size=block_size)
    rows = int(math.ceil(x.shape[-1] / block_size))
    assert torch.all(start == 0)
    assert torch.all(end == x.shape[-1])
    assert start.shape == (*x.shape[:-1], rows)
    assert end.shape == (*x.shape[:-1], rows)


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_symmetric(x_data_symmetric: Tensor, cutoff: float, block_size: int) -> None:
    x = x_data_symmetric
    start, end = ranges.make_ranges(cutoff, x, block_size=block_size)
    assert torch.all(start >= 0)
    assert torch.all(start <= end)
    assert torch.all(end <= x.shape[-1])

    x_ = x.reshape(-1, x.shape[-1])
    start_ = start.reshape(-1, start.shape[-1])
    end_ = end.reshape(-1, end.shape[-1])
    for batch in range(x_.shape[0]):
        mask = torch.zeros((x_.shape[1], x_.shape[1]), dtype=torch.bool)
        for i in range(start_.shape[1]):
            mask[i * block_size : (i + 1) * block_size, start_[batch, i] : end_[batch, i]] = True
        assert torch.all(mask == mask.T)
        for i in range(0, x_.shape[1], block_size):
            row_last = min(i + block_size, x_.shape[1]) - 1
            for j in range(i, x_.shape[1], block_size):
                in_range = x_[batch, j] <= x_[batch, row_last] + cutoff
                assert torch.all(mask[i : i + block_size, j : j + block_size] == in_range)


@pytest.mark.parametrize("cutoff", [0.1, 1.0, 11.0], ids=lambda x: f"cutoff_{x}")
@pytest.mark.parametrize("block_size", [1, 3], ids=lambda x: f"block_size_{x}")
def test_ranges_transpose(cutoff: float, block_size: int, x_data: tuple[Tensor, Tensor]) -> None:
    x1, x2 = x_data
    start, end = ranges.make_ranges(cutoff, x1, x2, block_size=block_size, align=True)
    start_t, end_t = ranges.transpose_ranges(
        start, end, x1.shape[-1], x2.shape[-1], block_size=block_size
    )
    assert torch.all(start_t >= 0)
    assert torch.all(start_t <= end_t)
    assert torch.all(end_t <= x1.shape[-1])

    x1_ = x1.reshape(-1, x1.shape[-1])
    x2_ = x2.reshape(-1, x2.shape[-1])
    start_ = start.reshape(-1, start.shape[-1])
    end_ = end.reshape(-1, end.shape[-1])
    start_t_ = start_t.reshape(-1, start_t.shape[-1])
    end_t_ = end_t.reshape(-1, end_t.shape[-1])
    for batch in range(x1_.shape[0]):
        mask = torch.zeros((x1_.shape[1], x2_.shape[1]), dtype=torch.bool)
        mask_t = torch.zeros((x2_.shape[1], x1_.shape[1]), dtype=torch.bool)
        for i in range(start_.shape[1]):
            mask[i * block_size : (i + 1) * block_size, start_[batch, i] : end_[batch, i]] = True
        for i in range(start_t_.shape[1]):
            mask_t[
                i * block_size : (i + 1) * block_size, start_t_[batch, i] : end_t_[batch, i]
            ] = True
        assert torch.all(mask == mask_t.T)


def test_ranges_transpose_noalign() -> None:
    x1 = torch.tensor([[0.0, 0.0, 2.0]])
    x2 = torch.tensor([[0.0, 2.0, 2.0]])
    block_size = 2
    start, end = ranges.make_ranges(1.0, x1, x2, block_size=block_size, align=False)
    with pytest.raises(ValueError, match="Ranges must be block-aligned"):
        ranges.transpose_ranges(start, end, x1.shape[1], x2.shape[1], block_size=block_size)
