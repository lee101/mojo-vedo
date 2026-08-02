"""A compact, NumPy-friendly mesh API matching vedo's covered methods."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ._lib import addr, f64, i64, lib


def _triangles(faces: np.ndarray | None) -> np.ndarray:
    if faces is None:
        return np.empty((0, 3), dtype=np.int64)
    f = i64(faces)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("only triangular faces with shape (n, 3) are supported")
    return f


def _inside(points: np.ndarray, mesh: "Mesh", eps: float = 1e-10) -> np.ndarray:
    """Odd-even ray test along a skew direction, for closed triangular surfaces."""
    if not mesh.ncells:
        return np.zeros(len(points), dtype=bool)
    tris = mesh.coordinates[mesh.cells]
    direction = np.array([1.0, 0.371390676, 0.173205081])
    direction /= np.linalg.norm(direction)
    verdict = np.zeros(len(points), dtype=bool)
    for k, p in enumerate(points):
        e1 = tris[:, 1] - tris[:, 0]
        e2 = tris[:, 2] - tris[:, 0]
        h = np.cross(np.broadcast_to(direction, e2.shape), e2)
        det = np.einsum("ij,ij->i", e1, h)
        usable = np.abs(det) > eps
        inv = np.zeros_like(det)
        inv[usable] = 1.0 / det[usable]
        s = p - tris[:, 0]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * q.dot(direction)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hits = usable & (u > eps) & (v > eps) & (u + v < 1.0 - eps) & (t > eps)
        verdict[k] = bool(np.count_nonzero(hits) % 2)
    return verdict


class Mesh:
    """Triangular surface mesh with vedo-compatible analysis methods.

    Parameters follow the common ``vedo.Mesh([points, faces])`` convention.  Arrays
    are exposed as ``coordinates`` and ``cells`` and attributes live in ``pointdata``
    and ``celldata``.
    """

    def __init__(self, inputobj, faces=None, c=None, alpha=1.0):
        if isinstance(inputobj, Mesh):
            points, faces = inputobj.coordinates, inputobj.cells
        elif faces is None and isinstance(inputobj, (tuple, list)) and len(inputobj) == 2:
            points, faces = inputobj
        else:
            points = inputobj
        self.coordinates = f64(points, ndim=2)
        if self.coordinates.shape[1] != 3:
            raise ValueError("points must have shape (n, 3)")
        self.cells = _triangles(faces)
        if self.cells.size and (self.cells.min() < 0 or self.cells.max() >= len(self.coordinates)):
            raise ValueError("face index outside points array")
        self.pointdata: dict[str, np.ndarray] = {}
        self.celldata: dict[str, np.ndarray] = {}
        self.name = "Mesh"

    @property
    def points(self):
        return self.coordinates

    @property
    def npoints(self) -> int:
        return len(self.coordinates)

    @property
    def ncells(self) -> int:
        return len(self.cells)

    def copy(self):
        copied = Mesh(self.coordinates.copy(), self.cells.copy())
        copied.pointdata = {k: v.copy() for k, v in self.pointdata.items()}
        copied.celldata = {k: v.copy() for k, v in self.celldata.items()}
        copied.name = self.name
        return copied

    def distance_to(self, pcloud, signed=False, invert=False, name="Distance") -> np.ndarray:
        target = as_mesh(pcloud)
        result = np.empty(self.npoints, dtype=np.float64)
        if not target.npoints:
            raise ValueError("distance target has no points")
        if target.ncells:
            lib().mvd_surface_distances(addr(self.coordinates), self.npoints, addr(target.coordinates),
                                        addr(target.cells), target.ncells, addr(result))
            if signed:
                result[_inside(self.coordinates, target)] *= -1.0
        else:
            lib().mvd_point_distances(addr(self.coordinates), self.npoints, addr(target.coordinates),
                                      target.npoints, addr(result))
        if invert:
            result *= -1.0
        self.pointdata[name] = result
        return result

    def hausdorff_distance(self, points) -> float:
        target = as_mesh(points)
        return float(max(np.max(self.distance_to(target)), np.max(target.distance_to(self))))

    def chamfer_distance(self, pcloud) -> float:
        target = as_mesh(pcloud)
        if not self.npoints or not target.npoints:
            raise ValueError("Chamfer distance requires two non-empty point sets")
        a = np.empty(self.npoints, dtype=np.float64)
        b = np.empty(target.npoints, dtype=np.float64)
        lib().mvd_chamfer_distances(addr(self.coordinates), self.npoints, addr(target.coordinates), target.npoints,
                                    addr(a), addr(b))
        return float((a.mean() + b.mean()) / 2.0)

    def densify(self, target_distance=0.1, nclosest=6, radius=None, niter=1, nmax=None):
        if radius is not None:
            raise NotImplementedError("radius neighborhoods are not yet supported; use nclosest")
        if target_distance <= 0 or nclosest < 1 or niter < 1:
            raise ValueError("target_distance, nclosest and niter must be positive")
        current = self.coordinates
        if nmax is not None:
            if not isinstance(nmax, (int, np.integer)) or nmax < self.npoints:
                raise ValueError("nmax must be an integer at least as large as the input point count")
            cap = int(nmax)
        else:
            cap = max(self.npoints, min(1_000_000, self.npoints * (self.npoints + 1) // 2))
        for _ in range(niter):
            generated = np.empty((cap, 3), dtype=np.float64)
            count = int(lib().mvd_densify_once(addr(current), len(current), float(target_distance),
                                               int(nclosest), cap, addr(generated)))
            if count < 0:
                raise RuntimeError(f"densify exceeded nmax={cap}; increase nmax")
            current = generated[:count].copy()
        result = Points(current)
        result.name = "DensifiedCloud"
        return result

    def cut_with_plane(self, origin=(0, 0, 0), normal=(1, 0, 0), invert=False):
        if not self.ncells:
            keep = ((self.coordinates - np.asarray(origin)) @ _normal(normal) >= 0)
            self.coordinates = self.coordinates[~keep if invert else keep].copy()
            return self
        o, n = f64(origin), _normal(normal)
        verts = np.empty((self.ncells * 6, 3), dtype=np.float64)
        faces = np.empty((self.ncells * 2, 3), dtype=np.int64)
        count = int(lib().mvd_clip_plane(addr(self.coordinates), addr(self.cells), self.ncells,
                                         *map(float, o), *map(float, n), int(invert), addr(verts), addr(faces)))
        faces = faces[:count].copy()
        used = int(faces.max()) + 1 if count else 0
        self.coordinates, self.cells = verts[:used].copy(), faces
        self.pointdata.clear()
        self.celldata.clear()
        return self

    def compute_cell_size(self):
        quality = np.empty(self.ncells, dtype=np.float64)
        lib().mvd_cell_metrics(addr(self.coordinates), addr(self.cells), self.ncells, 28, addr(quality))
        self.celldata["Area"] = quality
        self.celldata["Volume"] = np.zeros(self.ncells)
        self.celldata["Length"] = np.zeros(self.ncells)
        return self

    def area(self) -> float:
        self.compute_cell_size()
        return float(self.celldata["Area"].sum())

    def volume(self) -> float:
        if not self.ncells:
            return 0.0
        tri = self.coordinates[self.cells]
        return float(abs(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0))

    def compute_quality(self, metric=6):
        if metric not in (0, 2, 6, 8, 28):
            raise NotImplementedError(
                "supported triangular VTK quality metrics are 0 (edge ratio), 2 "
                "(radius ratio), 6 (minimum angle), 8 (maximum angle), and 28 (area)"
            )
        values = np.empty(self.ncells, dtype=np.float64)
        lib().mvd_cell_metrics(addr(self.coordinates), addr(self.cells), self.ncells, int(metric), addr(values))
        self.celldata["Quality"] = values
        return self

    def boolean(self, operation: str, mesh2, method=0, tol=None):
        """Exact CSG for disjoint or fully-contained closed triangle meshes.

        Intersecting surfaces need face splitting and are rejected explicitly, which is
        preferable to returning a plausible but invalid manifold.
        """
        if method not in (0, 1):
            raise ValueError(f"Unknown method={method}")
        other = as_mesh(mesh2)
        if not _is_closed(self) or not _is_closed(other):
            raise ValueError("boolean requires two closed triangular surface meshes")
        a_in_b = _inside(self.coordinates, other)
        b_in_a = _inside(other.coordinates, self)
        if (a_in_b.any() and not a_in_b.all()) or (b_in_a.any() and not b_in_a.all()):
            raise NotImplementedError("intersecting surfaces require exact face splitting")
        op = operation.lower()
        if op in ("plus", "+"):
            if a_in_b.all(): return other.copy()
            if b_in_a.all(): return self.copy()
            if _boxes_overlap(self, other):
                raise NotImplementedError("overlapping surfaces require exact face splitting")
            return _join(self, other)
        if op == "intersect":
            if a_in_b.all(): return self.copy()
            if b_in_a.all(): return other.copy()
            if _boxes_overlap(self, other):
                raise NotImplementedError("overlapping surfaces require exact face splitting")
            return Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
        if op in ("minus", "-"):
            if a_in_b.all(): return Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64))
            if b_in_a.all():
                cavity = other.copy()
                cavity.cells = cavity.cells[:, ::-1]
                return _join(self, cavity)
            if _boxes_overlap(self, other):
                raise NotImplementedError("overlapping surfaces require exact face splitting")
            return self.copy()
        raise ValueError("operation must be 'plus', 'intersect', or 'minus'")


class Points(Mesh):
    def __init__(self, inputobj, r=4, c=None, alpha=1.0):
        super().__init__(np.asarray(inputobj), None, c=c, alpha=alpha)
        self.name = "Points"


def _normal(value) -> np.ndarray:
    if isinstance(value, str):
        names = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}
        value = names[value.lower()]
    n = f64(value)
    if n.shape != (3,) or not np.linalg.norm(n):
        raise ValueError("normal must be a nonzero 3-vector")
    return n / np.linalg.norm(n)


def _join(a: Mesh, b: Mesh) -> Mesh:
    return Mesh(np.vstack((a.coordinates, b.coordinates)), np.vstack((a.cells, b.cells + a.npoints)))


def _is_closed(mesh: Mesh) -> bool:
    """A closed triangular 2-manifold has every undirected edge twice."""
    if not mesh.ncells:
        return False
    edges = np.sort(
        np.vstack((mesh.cells[:, [0, 1]], mesh.cells[:, [1, 2]], mesh.cells[:, [2, 0]])),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def _boxes_overlap(a: Mesh, b: Mesh) -> bool:
    """Conservative ambiguity check for CSG cases without contained vertices."""
    return bool(np.all(a.coordinates.min(axis=0) <= b.coordinates.max(axis=0))
                and np.all(b.coordinates.min(axis=0) <= a.coordinates.max(axis=0)))


def as_mesh(value) -> Mesh:
    return value if isinstance(value, Mesh) else Points(value)
