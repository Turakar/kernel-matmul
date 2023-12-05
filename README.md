# kernel-matmul

Accelerated multiplications with kernel matrices.

## Add C++/CUDA debug information
You can add debug information for VSCode using the `compile_commands.json` file.
This file contains the compile commands for a Torch extension build with Ninja.
The provided VSCode configurations expect this file in the root directory of the project.
As the file is specific to one extension, we provide a utility script `compile_commands.py` which creates a `compile_commands.json` which contains the concatenation of compilation commands for all extensions cached.

```python
python compile_commands.py
```
