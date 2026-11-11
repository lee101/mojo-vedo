"""Build and load the Mojo shared library used by :mod:`mojovedo`."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOVEDO_LIB") or os.path.join(ROOT, "dist", "libmojo-vedo.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mvd_point_distances": ([I, I, I, I, I], None),
    "mvd_point_distances_chunk": ([I, I, I, I, I, I, I], None),
    "mvd_surface_distances": ([I, I, I, I, I, I], None),
    "mvd_densify_once": ([I, I, F, I, I, I], I),
    "mvd_clip_plane": ([I, I, I, F, F, F, F, F, F, I, I, I], I),
    "mvd_cell_metrics": ([I, I, I, I, I], None),
}

# The point-to-point distance kernel is O(na*nb) with the inner loop in
# registers: three loads and eight flops per point pair, all of it streaming
# from cache once nb is small enough to stay resident. That is far above two
# flops per byte, so large inputs split across a thread pool. Measured on this
# box the crossover is around 8M point pairs: below it the serial call wins,
# above it the fan-out reaches 5x.
PARALLEL_PAIR_THRESHOLD = 8_000_000
PARALLEL_WORKERS = min(32, os.cpu_count() or 1)


def _row_chunks(count: int, workers: int) -> list[tuple[int, int]]:
    step = (count + workers - 1) // workers
    return [
        (lo, min(lo + step, count)) for lo in range(0, count, step) if lo < count
    ]


def _distance_pass(
    src: np.ndarray,
    src_count: int,
    target: np.ndarray,
    target_count: int,
    out: np.ndarray,
    workers: int,
) -> None:
    if workers <= 1 or src_count * target_count < PARALLEL_PAIR_THRESHOLD:
        lib().mvd_point_distances(
            addr(src), src_count, addr(target), target_count, addr(out)
        )
        return
    src_addr, target_addr, out_addr = addr(src), addr(target), addr(out)
    chunk = lib().mvd_point_distances_chunk
    parts = _row_chunks(src_count, min(src_count, workers))

    def run(bounds: tuple[int, int]) -> None:
        chunk(src_addr, src_count, target_addr, target_count, out_addr, *bounds)

    with ThreadPoolExecutor(max_workers=len(parts)) as pool:
        list(pool.map(run, parts))


def point_distances(
    a: np.ndarray, na: int, b: np.ndarray, nb: int, dst: np.ndarray
) -> None:
    """Distance from each of the first ``na`` rows of ``a`` to the nearest of ``b``."""
    _distance_pass(a, na, b, nb, dst, PARALLEL_WORKERS)


def chamfer_distances(
    a: np.ndarray,
    na: int,
    b: np.ndarray,
    nb: int,
    a_dst: np.ndarray,
    b_dst: np.ndarray,
) -> None:
    """Directed nearest-neighbour distances for both point sets."""
    workers = max(1, min(PARALLEL_WORKERS, max(na, nb)))
    _distance_pass(a, na, b, nb, a_dst, workers)
    _distance_pass(b, nb, a, na, b_dst, workers)


class BuildError(RuntimeError):
    pass


def _mojo() -> list[str]:
    if override := os.environ.get("MOJOVEDO_MOJO"):
        return override.split()
    if found := shutil.which("mojo"):
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOVEDO_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "capi.mojo")
    if os.environ.get("MOJOVEDO_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    proc = subprocess.run(
        _mojo() + ["build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
        _loaded = loaded
    return _loaded


def f64(a, *, ndim: int | None = None) -> np.ndarray:
    """Return a C-contiguous Float64 array without lossy numeric conversion."""
    raw = np.asarray(a)
    if raw.dtype.kind not in "fiu":
        raise TypeError("expected real numeric values")
    if raw.dtype.kind == "f" and raw.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("float values wider than float64 are not supported")
    if raw.dtype.kind in "iu" and raw.size:
        info = np.iinfo(raw.dtype)
        if info.min < -(1 << 53) or info.max > (1 << 53):
            # Check the actual range, not merely the storage type: int64 coordinate
            # arrays are common and are safe when their values are exactly representable.
            if raw.min() < -(1 << 53) or raw.max() > (1 << 53):
                raise ValueError("integer coordinates must be exactly representable as float64")
    arr = np.ascontiguousarray(raw, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"expected a {ndim}D array, got {arr.ndim}D")
    if not np.isfinite(arr).all():
        raise ValueError("coordinates must be finite")
    return arr


def i64(a, *, ndim: int | None = None) -> np.ndarray:
    """Return C-contiguous Int64 indices, rejecting lossy casts."""
    raw = np.asarray(a)
    if raw.dtype.kind not in "iu":
        raise TypeError("indices must use an integer dtype")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError("index exceeds int64 range")
    arr = np.ascontiguousarray(raw, dtype=np.int64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"expected a {ndim}D array, got {arr.ndim}D")
    return arr


def addr(a: np.ndarray) -> int:
    return int(a.ctypes.data)


if __name__ == "__main__":
    print(build(force="--force" in sys.argv))
