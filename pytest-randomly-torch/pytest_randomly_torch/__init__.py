def seed(seed: int) -> None:
    """
    Seed the random number generator in PyTorch.

    Used by pytest-randomly.
    """
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        import warnings

        warnings.warn("torch not found, cannot seed it")
