import multiprocessing.pool
from kernel_matmul.compile import find_best, load_native
from kernel_matmul.configurations import Configuration
from kernel_matmul.util import format_dict


from types import ModuleType
from typing import Any


class NativeFunction:
    _cache_key: str | None = None
    _module: ModuleType | None = None

    def __init__(
        self,
        name: str,
        config: Configuration,
        num_measurements: int = 1,
        verbose: bool = False,
        compile_pool: multiprocessing.pool.Pool | None = None,
    ):
        super().__init__()
        self.name = name
        self.config = config
        self.num_measurements = num_measurements
        self.verbose = verbose
        self.compile_pool = compile_pool

    def __call__(self, *args: Any) -> Any:
        cache_key = self.config.cache_key(args)
        if self._module is None or cache_key != self._cache_key:
            candidates = self.config.make_candidates(args)
            if len(candidates) > 1:
                if self.verbose:
                    print(f"Autotuning {self.name}...")
                best, timings = find_best(
                    self.name,
                    args,
                    candidates,
                    num_measurements=self.num_measurements,
                    return_timings=True,
                    compile_pool=self.compile_pool,
                )
                if self.verbose:
                    for i, timing in enumerate(timings):
                        print(f"  - Candidate {i}: {timing:.3f} ms ({format_dict(candidates[i])})")
                    print(f"Best: {format_dict(best)}")
            else:
                best = candidates[0]
            self._module = load_native(self.name, defines=best)
            self._cache_key = cache_key
        return self._module.call(*args)
