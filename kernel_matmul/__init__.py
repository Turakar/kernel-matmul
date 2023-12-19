import torch


_BLOCK_SIZE = 128


def _seed(seed: int) -> None:
    """
    Seed the random number generator in PyTorch.

    Used by pytest-randomly.
    """
    torch.manual_seed(seed)
