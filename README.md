# kernel-matmul

Accelerated multiplications with kernel matrices.

## Add C++/CUDA debug information
You can add debug information for VSCode using the `compile_commands.json` file.
This file contains the compile commands for a Torch extension build with Ninja.
The provided VSCode configurations expect this file in the root directory of the project.
To generate it, follow these steps:

1. Run the program to build a Torch extension.
2. Locate the extension. It should be in `~/.cache/torch_extensions/<python_version>/<extension_name>`.
3. Run `ninja -C <extension_path> -t compdb > compile_commands.json` to generate the file.

Note that the `compile_commands.json` file is specific to each extension.
