# kernel-matmul

Accelerated multiplications with kernel matrices.

## Add C++/CUDA IntelliSense information
We provide a default VSCode configuration in `.vscode`.
This refers to a folder of include paths named `include` in the project root.
You can create this folder using the provided script `find_include_paths.py`:

```python
python find_include_paths.py
```

This script creates the `include` directory, queries Torch and Python for the include paths and symlinks them to the `include` directory.
You can then use the VSCode configuration to work with the C++/CUDA code.
