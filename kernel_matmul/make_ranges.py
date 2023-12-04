import torch
from torch import Tensor
from gpytorch.kernels.kernel import sq_dist
from tqdm import tqdm

from kernel_matmul import _BLOCK_SIZE


def make_ranges(
    cutoff, x1: Tensor, x2: Tensor, chunk_size: int = 10000**2
) -> tuple[Tensor, Tensor]:
    if x1.shape[-1] * x2.shape[-1] <= chunk_size:
        distances = sq_dist(x1.unsqueeze(-1), x2.unsqueeze(-1))
        mask = (distances <= cutoff**2).to(torch.uint8)
        start = torch.argmax(mask, dim=1).to(torch.int32)
        end = (mask.shape[1] - torch.argmax(mask.flip(1), dim=1)).to(torch.int32)
    else:
        starts = []
        ends = []
        chunk_step = max(1, chunk_size // x2.shape[-1])
        iterator = range(0, x1.shape[-1], chunk_step)
        if len(iterator) > 1000:
            iterator = tqdm(iterator, leave=False, desc="make_ranges")
        for chunk_start in iterator:
            chunk_end = min(chunk_start + chunk_step, x1.shape[-1])
            x2_offset = 0
            if chunk_start > 0:
                x2_offset = starts[-1][-1].item()
            start, end = make_ranges(
                cutoff,
                x1[..., chunk_start:chunk_end],
                x2[..., x2_offset:],
                chunk_size=chunk_step * x2.shape[-1],
            )
            starts.append(start + x2_offset)
            ends.append(end + x2_offset)
        start = torch.cat(starts, dim=-1)
        end = torch.cat(ends, dim=-1)
    return start, end


def make_ranges_block(
    cutoff: float | None,
    x1: Tensor,
    x2: Tensor,
    block_size: int = _BLOCK_SIZE,
    chunk_size: int = 10000**2,
) -> tuple[Tensor, Tensor]:
    if cutoff is None:
        return make_noop_ranges_block(x1, x2, block_size=block_size)
    start, end = make_ranges(cutoff, x1, x2, chunk_size=chunk_size)
    if start.shape[0] % block_size != 0:
        pad = block_size - start.shape[0] % block_size
        start = torch.nn.functional.pad(start, (0, pad), value=x2.shape[0])
        end = torch.nn.functional.pad(end, (0, pad), value=0)
    start = torch.min(start.reshape(-1, block_size), dim=1).values
    end = torch.max(end.reshape(-1, block_size), dim=1).values
    return start.to(torch.int32), end.to(torch.int32)


def make_ranges_block_symmetric(
    cutoff: float | None,
    x: Tensor,
    block_size: int = _BLOCK_SIZE,
) -> tuple[Tensor, Tensor]:
    if cutoff is None:
        return make_noop_ranges_block(x, x, block_size=block_size)
    num_blocks = (x.shape[0] + block_size - 1) // block_size
    mask = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)
    block_start = torch.arange(0, x.shape[0], block_size)
    block_end = torch.cat([block_start[1:], torch.tensor([x.shape[0]])])
    for i in range(num_blocks):
        row_mask = (
            sq_dist(x[block_start[i] : block_end[i], None], x[block_start[i] :, None]) <= cutoff**2
        )
        row_mask = torch.any(row_mask, dim=0)
        if not torch.all(row_mask):
            row_end = torch.argmin(row_mask.to(torch.uint8)).item() + block_start[i] - 1
            row_end_block = row_end // block_size
            mask[i, i : row_end_block + 1] = True
            mask[i : row_end_block + 1, i] = True
        else:
            mask[i, i:] = True
            mask[i:, i] = True
    starts = []
    ends = []
    for i in range(num_blocks):
        starts.append(torch.min(block_start[mask[i]]).item())
        ends.append(torch.max(block_end[mask[i]]).item())
    start = torch.tensor(starts, dtype=torch.int32, device=x.device)
    end = torch.tensor(ends, dtype=torch.int32, device=x.device)
    return start, end


def make_noop_ranges_block(
    x1: Tensor, x2: Tensor, block_size: int = _BLOCK_SIZE
) -> tuple[Tensor, Tensor]:
    num_blocks = (x1.shape[0] + block_size - 1) // block_size
    start = torch.zeros(num_blocks, dtype=torch.int32, device=x1.device)
    end = torch.full_like(start, x2.shape[0], dtype=torch.int32)
    return start, end
