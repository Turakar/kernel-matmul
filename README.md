# KernelMatmul

Accelerated multiplications with kernel matrices.

## Structure
- `native/`: CUDA/C++ code, separated into a `common/` folder and several operators. `matmul/` contains the main KernelMatmul implementation.
- `kernel_matmul/`: Python library. This code is responsible for the on-demand compilation and autotuning of the CUDA/C++ operators (`native_function.py`, `configurations.py`), the sparsity pattern calculation (`ranges.py`), and contains the implementation of the GPyTorch functionality (`gpytorch/` and `linear_operator.py`).
- `examples/`: Contains two simple examples demonstrating the use of the library.
- `tests/`: Contains the extensive unit test suite for this library. The tests in `native/` ensure correct functionality of the CUDA/C++ implementations, while the `test_linear_operator.py` ensures correctness with the LinearOperator test suite from GPyTorch.

Note that the GPyTorch integration is available only if the corresponding extra (`gpytorch` or just `linear_operator`) is activated.

## Development
This project has the following dependencies:
- NVIDIA CUDA 12.1
- Python 3.11
- GPyTorch 1.11
- PyTorch 2.2.2
- And many other Python packages (see below)

### Development Environment
The development environment is powered by [devcontainers](https://containers.dev/).
With [VSCode](https://code.visualstudio.com/), [Docker](https://www.docker.com/) and the [devcontainer extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), you can just open this folder with VSCode and will be prompted to download, build and open the container.
The environment is defined by the files in the `.devcontainer` directory.
This config assumes a working [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installation.

**Note:** Developing inside a container provides us with a reproducible environment which is similar on all platforms (e.g., with the correct CUDA and Python version).
This reduces compatibility issues once the initial devcontainer installation is completed.

### Python Dependencies
The project's dependencies are managed with [Poetry](https://python-poetry.org/).
Poetry is already installed in the container.
You can install the Python dependencies using the following command:

```bash
poetry install
```

### Pre-Commit
This project uses [pre-commit](https://pre-commit.com/) (configured in `.pre-commit-config.yml`) for running formatters and linters.
To install it as a Git commit hook, run

```
pre-commit install
```

inside the container.
You can also invoke it manually with the following command:

```
pre-commit run
```

### Add C++/CUDA IntelliSense information
We provide a default VSCode configuration in `.vscode`.
This refers to a folder of include paths named `include` in the project root.
You can create this folder using the provided script `find_include_paths.py`:

```python
python find_include_paths.py
```

This script creates the `include` directory, queries Torch and Python for the include paths and symlinks them to the `include` directory.
You can then use the VSCode configuration to work with the C++/CUDA code.

### Unit tests
We use [pytest](https://docs.pytest.org/en/stable/) for unit tests.
Just run it:

```bash
pytest
```

This will likely take a long time to complete, because of the compilation times for all the operator variants.
