import time
import functools
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timer] {func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def truncate_text(text: str, max_chars: int = 200) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
