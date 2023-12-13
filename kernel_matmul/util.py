from typing import Any, TypeVar
import itertools
import functools


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
