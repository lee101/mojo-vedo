# mojo-vedo

`mojo-vedo` is a standalone Mojo port of the compute-heavy mesh-analysis portion of
[vedo](https://vedo.embl.es/). It supplies a compact NumPy-facing `Mesh` and `Points`
API for workloads where keeping geometry in contiguous arrays is more useful than the
full VTK scene graph.

The current covered subset is:

- point-cloud and point-to-triangle-surface distances, including signed distance for
  closed triangular meshes, Chamfer distance, and Hausdorff distance;
- point-cloud densification with nearest-neighbour neighborhoods;
- triangular mesh clipping with `cut_with_plane`;
- `compute_cell_size`, `area`, `volume`, and VTK-compatible triangle quality metrics
  0 (edge ratio), 2 (radius ratio), 6 (minimum angle), 8 (maximum angle), and 28 (area);
- `boolean` for fully-contained closed triangular surfaces and disjoint closed surfaces
  whose axis-aligned bounding boxes do not overlap.

This is intentionally not the whole of vedo: rendering, VTK datasets, non-triangular
cells, point attributes during topology changes, radius-based densification, and general
CSG are not covered. Boolean inputs with intersecting or otherwise ambiguous overlapping
surfaces raise instead of returning an invalid approximate mesh. The methods and argument names above mirror
their vedo counterparts; `Mesh([points, faces])` is the usual construction form.

## Install and use

```bash
pixi install
pixi run build
```

`pixi` sets `PYTHONPATH=python`, so this runs directly from the checkout:

```python
import numpy as np
from mojovedo import Mesh, Points

triangle = Mesh([
    np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
    np.array([[0, 1, 2]]),
])
distances = Points([[0.2, 0.2, 1.0]]).distance_to(triangle)
print(distances)                    # [1.]
print(triangle.compute_quality(6).celldata["Quality"])  # [45.]
```

Run the verified suite with `pixi run test`; it compares the covered operations to the
installed `vedo` reference wherever VTK defines a direct result. The suite also covers
FFI input validation and the SIMD and parallel-loop boundaries.

## Benchmark

Measured with `pixi run bench` on Linux 6.8.0-136-generic x86_64, glibc 2.39, Python
3.13.14, on 2026-08-27. Times are the best of three complete calls; library loading is
warmed up before timing.

| operation | mojo-vedo | vedo | ratio | result |
|---|---:|---:|---:|---|
| point distance (12k x 3k) | 8.8 ms | 17.6 ms | 1.99x | faster |
| Chamfer distance (12k, 3k) | 13.0 ms | 24.6 ms | 1.89x | faster |

Point scans use SIMD strided loads over the contiguous `(n, 3)` NumPy layout, with a
scalar remainder, and distribute only sufficiently large independent row chunks across
CPU workers. Chamfer batches its two directions into one balanced parallel schedule.
Each point comparison performs about eight floating-point operations for 24 bytes of
target coordinates (roughly 0.33 flop/byte), so it is below the arithmetic-intensity
threshold where a GPU path is justified. There is no GPU path.

## How it works

The hot loops live together in `src/capi.mojo` and build into
`dist/libmojo-vedo.so`. Python passes C-contiguous `float64` point arrays and `int64`
triangle indices through `ctypes` as raw integer addresses. Mojo reconstructs typed
pointers internally, writes into Python-owned output buffers, and never allocates across
the FFI boundary. Points are `(n, 3)` row-major arrays and triangle cells are `(m, 3)`
row-major index arrays.
