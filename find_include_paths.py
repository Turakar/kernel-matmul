import os
import sysconfig

import torch.utils.cpp_extension


def main():
    os.makedirs("include", exist_ok=False)
    with open("include/.gitignore", "w") as fd:
        fd.write("*\n")
    for i, path in enumerate(torch.utils.cpp_extension.include_paths(cuda=True)):
        if os.path.exists(path):
            print(path)
            os.symlink(path, f"include/torch_{i}")
    python_path = sysconfig.get_path(
        "include", scheme="nt" if torch.utils.cpp_extension.IS_WINDOWS else "posix_prefix"
    )
    print(python_path)
    os.symlink(python_path, "include/python")


if __name__ == "__main__":
    main()
