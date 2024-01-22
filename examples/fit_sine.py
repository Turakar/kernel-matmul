from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from tqdm import tqdm
import plotly.graph_objects as go

from kernel_matmul.gpytorch.spectral import SpectralKernelMatmulKernel
from gpytorch.means import ZeroMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.constraints import Interval

import torch
from torch import Tensor
import plotly.colors


class MyGP(ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor):
        likelihood = GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = SpectralKernelMatmulKernel(
            epsilon=1e-4, lengthscale_constraint=Interval(1e-2, 1.0)
        )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def main():
    device = torch.device("cuda:0")

    full_x = torch.sort(torch.rand(8000, device=device) * 10)[0]
    full_y = torch.sin((2 * torch.pi * 0.5) * full_x)
    noise = 0.5
    full_y_noisy = full_y + torch.randn_like(full_y) * noise

    test_start = int(0.45 * len(full_x))
    test_end = int(0.55 * len(full_x))
    train_mask = torch.ones_like(full_x, dtype=torch.bool)
    train_mask[test_start:test_end] = False
    train_x = full_x[train_mask]
    train_y = full_y_noisy[train_mask]
    test_x = full_x[~train_mask]
    test_y = full_y[~train_mask]

    model = MyGP(train_x, train_y).to(device)
    model.covar_module.frequency = torch.tensor(0.1)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    progress_bar = tqdm(range(100))
    for _ in progress_bar:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

    print("Fitted parameters:")
    print(f"Frequency: {model.covar_module.frequency.item()}")
    print(f"Lengthscale: {model.covar_module.lengthscale.item()}")
    print(f"Outputscale: {model.covar_module.outputscale.item()}")
    print(f"Noise: {model.likelihood.noise.item()**0.5}")

    model.eval()
    with torch.no_grad():
        pred = model(test_x)
        lower, upper = pred.confidence_region()

    full_x = full_x.cpu()
    full_y = full_y.cpu()
    full_y_noisy = full_y_noisy.cpu()
    train_x = train_x.cpu()
    train_y = train_y.cpu()
    test_x = test_x.cpu()
    test_y = test_y.cpu()
    pred_mean = pred.mean.cpu()
    lower = lower.cpu()
    upper = upper.cpu()

    colors = plotly.colors.qualitative.T10

    fig = go.Figure()
    # true data
    fig.add_trace(
        go.Scatter(x=full_x, y=full_y, mode="lines", name="True", line=dict(color=colors[0]))
    )
    # training data
    fig.add_trace(
        go.Scatter(x=train_x, y=train_y, mode="markers", name="Train", marker=dict(color=colors[1]))
    )
    # prediction
    fig.add_trace(
        go.Scatter(
            x=test_x,
            y=pred_mean,
            mode="lines",
            name="Prediction",
            line=dict(color=colors[2]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=torch.cat([test_x, test_x.flip(0)]),
            y=torch.cat([upper, lower.flip(0)]),
            fill="toself",
            fillcolor=transparent_color(colors[2]),
            line=dict(width=0),
            name="Confidence Interval",
        )
    )
    fig.show()


def transparent_color(color, alpha=0.3):
    rgb = plotly.colors.hex_to_rgb(color)
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"


if __name__ == "__main__":
    main()
