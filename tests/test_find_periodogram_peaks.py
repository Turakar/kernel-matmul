import torch
from kernel_matmul.util import find_periodogram_peaks


def test_find_periodogram_peaks():
    x = torch.linspace(0, 1, 1000)
    freq1 = 2.5
    freq2 = 9
    y = torch.sin(2 * torch.pi * freq1 * x + 0.1) * 5 + torch.sin(2 * torch.pi * freq2 * x + 0.1)

    freqs, mags = find_periodogram_peaks(
        x,
        y,
        2,
        10,
    )

    assert torch.allclose(freqs, torch.tensor([freq1, freq2]), atol=0.1)
