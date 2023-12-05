import json
import os
import subprocess
import warnings
from os import path

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=DeprecationWarning, message="pkg_resources is deprecated as an API"
    )
    import torch.utils.cpp_extension


def main():
    build_path = torch.utils.cpp_extension._get_build_directory("km___ranges", verbose=False)
    build_dir = path.dirname(path.normpath(build_path))
    print(f"Looking for extensions in {build_dir}")
    compile_commands = []
    for extension in os.listdir(build_dir):
        if extension.startswith("km___"):
            print(f"Found extension {extension}")
            output = subprocess.check_output(
                ["ninja", "-C", os.path.join(build_dir, extension), "-t", "compdb"]
            )
            compile_commands += json.loads(output.decode("utf-8"))
    with open("compile_commands.json", "w") as f:
        json.dump(compile_commands, f, indent=2)


if __name__ == "__main__":
    main()
