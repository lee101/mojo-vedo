"""Measured comparison with vedo's VTK-backed reference implementation."""

from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
from vedo import Points as VedoPoints

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from mojovedo import Points  # noqa: E402


def measure(fn, repeat=3):
    fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples)


def main():
    rng = np.random.default_rng(7)
    a = np.ascontiguousarray(rng.normal(size=(12_000, 3)))
    b = np.ascontiguousarray(rng.normal(size=(3_000, 3)))
    ours_a, ours_b = Points(a), Points(b)
    vedo_a, vedo_b = VedoPoints(a), VedoPoints(b)
    cases = [
        ("point distance (12k x 3k)", lambda: ours_a.distance_to(ours_b), lambda: vedo_a.distance_to(vedo_b)),
        ("Chamfer distance (12k, 3k)", lambda: ours_a.chamfer_distance(ours_b), lambda: vedo_a.chamfer_distance(vedo_b)),
    ]
    print(f"Machine: {platform.platform()} | Python {platform.python_version()}")
    print("| operation | mojo-vedo | vedo | ratio | result |")
    print("|---|---:|---:|---:|---|")
    for name, ours, theirs in cases:
        mo = measure(ours)
        ve = measure(theirs)
        ratio = ve / mo
        result = "faster" if mo < ve else "slower"
        print(f"| {name} | {mo * 1e3:.1f} ms | {ve * 1e3:.1f} ms | {ratio:.2f}x | {result} |")


if __name__ == "__main__":
    main()
