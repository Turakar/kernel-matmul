from typing import Any, TypeVar
import itertools
import functools
import torch
from torch import Tensor
from gatspy.periodic import LombScargleFast
import scipy.signal
import math


def format_dict(d: dict[str, Any]) -> str:
    content = ", ".join(f"{k}={v}" for k, v in d.items())
    return f"dict({content})"


K = TypeVar("K")
V = TypeVar("V")


def dict_product(*factors: list[dict[K, V]]) -> list[dict[K, V]]:
    factors = [factor for factor in factors if len(factor) > 0]
    for factor in factors:
        keys0 = set(factor[0])
        if any(set(d.keys()) != keys0 for d in factor):
            raise ValueError("All dicts in a factor must have the same keys")
    return [
        functools.reduce(lambda d1, d2: d1 | d2, combination, {})
        for combination in itertools.product(*factors)
    ]


def find_periodogram_peaks(
    x: Tensor,
    y: Tensor,
    num_components: int,
    max_frequency: float,
    peak_distance: int = 5,
    peak_oversample: int = 1,
    sum_tasks: bool = True,
    min_num_freqs: int = 10000,
    max_num_freqs: int = 1000000,
) -> tuple[Tensor, Tensor]:
    # https://games-uchile.github.io/mogptk/examples.html?q=03_Parameter_Initialization

    single = len(y.shape) == len(x.shape)
    if single:
        y = y.unsqueeze(-1)
        sum_tasks = True
    num_tasks = y.shape[-1]

    # Computation happens on CPU as we will use SciPy.
    x = x.cpu()
    y = y.cpu()

    # Compute the frequency grid
    f_max = torch.tensor(float(max_frequency))
    timespan = x[-1] - x[0]
    f_min = 1 / timespan
    num_freqs = (
        torch.ceil(
            torch.clamp(
                5 * timespan * f_max,
                min=torch.tensor(min_num_freqs),
                max=torch.tensor(float(max_num_freqs)),
            )
        )
        .to(torch.long)
        .item()
    )
    df = (f_max - f_min) / num_freqs
    freqs = torch.arange(num_freqs) * df + f_min

    # Compute the Lomb-Scargle periodogram for each task independently
    pgrams = []
    for i in range(num_tasks):
        ls = LombScargleFast().fit(x.numpy(), y[..., i].numpy())
        pgram_i = ls.score_frequency_grid(f_min.item(), df.item(), num_freqs)
        pgrams.append(pgram_i)

    if sum_tasks:
        pgrams = [sum(pgrams)]

    peak_freqs = []
    peak_mags = []
    for pgram in pgrams:
        # Find peaks in the periodogram.
        peaks, _ = scipy.signal.find_peaks(pgram, distance=peak_distance)
        pgram = torch.from_numpy(pgram)
        peaks = torch.from_numpy(peaks)

        # Select the peaks with the highest magnitude.
        num_peaks = int(math.ceil(num_components / peak_oversample))
        max_idx = torch.argsort(-pgram[peaks])[:num_peaks]
        peak_idx = peaks[max_idx]
        peak_idx = peak_idx.repeat(peak_oversample)[:num_components]
        peak_mag = pgram[peak_idx]
        peak_freq = freqs[peak_idx]

        # If we oversampled peaks, make sure that they have slightly different gradients.
        if peak_oversample > 1:
            peak_freq = peak_freq + torch.rand_like(peak_freq) * df / 10

        # Normalize magnitude of peaks
        peak_mag = peak_mag / torch.sum(peak_mag)

        peak_freqs.append(peak_freq.to(torch.float32))
        peak_mags.append(peak_mag.to(torch.float32))

    if sum_tasks:
        return peak_freqs[0], peak_mags[0]
    else:
        return torch.stack(peak_freqs, dim=0), torch.stack(peak_mags, dim=0)
