import click
import multiprocessing
import torch
from kernel_matmul.configurations import MatmulAutotuneConfiguration
from kernel_matmul.native_function import NativeFunction
from kernel_matmul.ranges import make_ranges
from kernel_matmul.compile import load_native


@click.group()
def cli():
    pass


@cli.command()
def autotune():
    x, start, end, parameters, rhs = make_example()

    with multiprocessing.get_context("spawn").Pool(processes=10) as pool:
        matmul = NativeFunction(
            "matmul",
            MatmulAutotuneConfiguration("rbf"),
            compile_pool=pool,
            verbose=True,
            num_measurements=10,
        )
        matmul(x, x, rhs, parameters, start, end)

    print(matmul.defines)


@cli.command()
def time():
    defines = {
        "BLOCK_SIZE": 128,
        "BATCH_DIM": 1,
        "KERNEL_RBF": None,
        "MATMUL_THREADS": 64,
        "MATMUL_PER_THREAD": 2,
        "MATMUL_COL_BLOCKS": 4,
        "MATMUL_USE_SHM": 1,
        "MATMUL_K_BLOCK_SIZE": 1,
    }
    x, start, end, parameters, rhs = make_example()
    matmul = load_native("matmul", defines)
    matmul.call(x, x, rhs, parameters, start, end)
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    timings = []
    for i in range(10):
        event_start.record()
        matmul.call(x, x, rhs, parameters, start, end)
        event_end.record()
        torch.cuda.synchronize()
        timings.append(event_start.elapsed_time(event_end))
    print(
        f"Wall time: {torch.tensor(timings).median().item():.3f} ms [{', '.join(f'{t:.3f}' for t in sorted(timings))}]"
    )


def make_example():
    dt = 1 / 24
    num_samples = 65536
    tkwargs = dict(dtype=torch.float32, device=torch.device("cuda:0"))
    batch_size = 5
    cutoff = 100.0
    rhs_columns = 1

    torch.manual_seed(0)
    x = torch.sort(torch.rand(num_samples, **tkwargs) * dt * num_samples)[0]
    start, end = make_ranges(cutoff, x)
    x = x.unsqueeze(0).expand(batch_size, -1)
    start = start.unsqueeze(0).expand(batch_size, -1)
    end = end.unsqueeze(0).expand(batch_size, -1)

    torch.manual_seed(0)
    parameters = torch.ones(batch_size, 2, **tkwargs) + torch.randn(batch_size, 2, **tkwargs) * 0.01

    torch.manual_seed(0)
    rhs = torch.randn(batch_size, num_samples, rhs_columns, **tkwargs)
    rhs = rhs / rhs.norm(dim=0, keepdim=True)
    return x, start, end, parameters, rhs


if __name__ == "__main__":
    cli()
