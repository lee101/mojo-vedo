"""Build and load the Mojo shared library used by :mod:`mojovedo`."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOVEDO_LIB") or os.path.join(ROOT, "dist", "libmojo-vedo.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mvd_point_distances": ([I, I, I, I, I], None),
    "mvd_chamfer_distances": ([I, I, I, I, I, I], None),
    "mvd_surface_distances": ([I, I, I, I, I, I], None),
    "mvd_densify_once": ([I, I, F, I, I, I], I),
    "mvd_clip_plane": ([I, I, I, F, F, F, F, F, F, I, I, I], I),
    "mvd_cell_metrics": ([I, I, I, I, I], None),
}


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
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_loaded, name)
            fn.argtypes, fn.restype = argtypes, restype
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
