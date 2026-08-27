"""C ABI kernels for the supported vedo mesh-analysis subset."""

from std.math import acos, sqrt
from std.runtime.asyncrt import TaskGroup, initialize_runtime, parallelism_level
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime PARALLEL_DISTANCE_THRESHOLD = 262_144
comptime PARALLEL_DISTANCE_WORKERS = 36
comptime PARALLEL_CHAMFER_WORKERS = 72


@always_inline
def parallelize[
    origins: OriginSet, //, func: def(Int) capturing[origins] -> None
](num_work_items: Int, max_workers: Int):
    var worker_count = min(num_work_items, min(max_workers, parallelism_level()))
    var chunk_size, extra_items = divmod(num_work_items, worker_count)

    async def work_chunk(worker: Int, chunk_size: Int, extra_items: Int):
        var begin = worker * chunk_size + min(worker, extra_items)
        var count = chunk_size + Int(worker < extra_items)
        for item in range(begin, begin + count):
            func(item)

    var tasks = TaskGroup()
    for worker in range(worker_count):
        tasks.create_task(work_chunk(worker, chunk_size, extra_items))
    tasks.wait()


def sqdist3(a: FPtr, ai: Int, b: FPtr, bi: Int) -> Float64:
    var x = a[ai] - b[bi]
    var y = a[ai + 1] - b[bi + 1]
    var z = a[ai + 2] - b[bi + 2]
    return x * x + y * y + z * z


def point_min_sqdist(a: FPtr, ai: Int, b: FPtr, nb: Int) -> Float64:
    comptime W = simd_width_of[DType.float64]()
    var best_lanes = SIMD[DType.float64, W](1e300)
    var j = 0
    var ax = SIMD[DType.float64, W](a[ai])
    var ay = SIMD[DType.float64, W](a[ai + 1])
    var az = SIMD[DType.float64, W](a[ai + 2])
    while j + W <= nb:
        var base = 3 * j
        var dx = ax - (b + base).strided_load[width=W](3)
        var dy = ay - (b + base + 1).strided_load[width=W](3)
        var dz = az - (b + base + 2).strided_load[width=W](3)
        best_lanes = min(best_lanes, dx * dx + dy * dy + dz * dz)
        j += W
    var best = best_lanes.reduce_min()
    while j < nb:
        var d = sqdist3(a, ai, b, 3 * j)
        if d < best:
            best = d
        j += 1
    return best


def point_distance_range(a: FPtr, start: Int, stop: Int, b: FPtr, nb: Int, dst: FPtr):
    for i in range(start, stop):
        dst[i] = sqrt(point_min_sqdist(a, 3 * i, b, nb))


def point_triangle_sqdist(p: FPtr, pi: Int, v: FPtr, ai: Int, bi: Int, ci: Int) -> Float64:
    var ax = v[ai]
    var ay = v[ai + 1]
    var az = v[ai + 2]
    var abx = v[bi] - ax
    var aby = v[bi + 1] - ay
    var abz = v[bi + 2] - az
    var acx = v[ci] - ax
    var acy = v[ci + 1] - ay
    var acz = v[ci + 2] - az
    var apx = p[pi] - ax
    var apy = p[pi + 1] - ay
    var apz = p[pi + 2] - az
    var d1 = abx * apx + aby * apy + abz * apz
    var d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz
    var bpx = p[pi] - v[bi]
    var bpy = p[pi + 1] - v[bi + 1]
    var bpz = p[pi + 2] - v[bi + 2]
    var d3 = abx * bpx + aby * bpy + abz * bpz
    var d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz
    var vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        var q = d1 / (d1 - d3)
        var dx = apx - q * abx
        var dy = apy - q * aby
        var dz = apz - q * abz
        return dx * dx + dy * dy + dz * dz
    var cpx = p[pi] - v[ci]
    var cpy = p[pi + 1] - v[ci + 1]
    var cpz = p[pi + 2] - v[ci + 2]
    var d5 = abx * cpx + aby * cpy + abz * cpz
    var d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz
    var vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        var q = d2 / (d2 - d6)
        var dx = apx - q * acx
        var dy = apy - q * acy
        var dz = apz - q * acz
        return dx * dx + dy * dy + dz * dz
    var va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        var q = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        var bcx = v[ci] - v[bi]
        var bcy = v[ci + 1] - v[bi + 1]
        var bcz = v[ci + 2] - v[bi + 2]
        var dx = bpx - q * bcx
        var dy = bpy - q * bcy
        var dz = bpz - q * bcz
        return dx * dx + dy * dy + dz * dz
    var denom = 1.0 / (va + vb + vc)
    var q = vb * denom
    var r = vc * denom
    var dx = apx - abx * q - acx * r
    var dy = apy - aby * q - acy * r
    var dz = apz - abz * q - acz * r
    return dx * dx + dy * dy + dz * dz


@export("mvd_point_distances")
def mvd_point_distances(a_addr: Int, na: Int, b_addr: Int, nb: Int, dst_addr: Int) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var b = FPtr(unsafe_from_address=b_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)

    @parameter
    def process_row(i: Int):
        dst[i] = sqrt(point_min_sqdist(a, 3 * i, b, nb))

    if na > 1 and na * nb >= PARALLEL_DISTANCE_THRESHOLD:
        initialize_runtime()
        parallelize[process_row](na, min(na, PARALLEL_DISTANCE_WORKERS))
    else:
        point_distance_range(a, 0, na, b, nb, dst)


@export("mvd_chamfer_distances")
def mvd_chamfer_distances(a_addr: Int, na: Int, b_addr: Int, nb: Int, a_dst_addr: Int, b_dst_addr: Int) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var b = FPtr(unsafe_from_address=b_addr)
    var a_dst = FPtr(unsafe_from_address=a_dst_addr)
    var b_dst = FPtr(unsafe_from_address=b_dst_addr)
    var rows = na + nb

    @parameter
    def process_chunk(task: Int):
        var chunk = task // 2
        if task % 2 == 0:
            var start = chunk * na // PARALLEL_DISTANCE_WORKERS
            var stop = (chunk + 1) * na // PARALLEL_DISTANCE_WORKERS
            point_distance_range(a, start, stop, b, nb, a_dst)
        else:
            var start = chunk * nb // PARALLEL_DISTANCE_WORKERS
            var stop = (chunk + 1) * nb // PARALLEL_DISTANCE_WORKERS
            point_distance_range(b, start, stop, a, na, b_dst)

    if rows > 1 and 2 * na * nb >= PARALLEL_DISTANCE_THRESHOLD:
        initialize_runtime()
        parallelize[process_chunk](
            2 * PARALLEL_DISTANCE_WORKERS, PARALLEL_CHAMFER_WORKERS
        )
    else:
        point_distance_range(a, 0, na, b, nb, a_dst)
        point_distance_range(b, 0, nb, a, na, b_dst)


@export("mvd_surface_distances")
def mvd_surface_distances(a_addr: Int, na: Int, v_addr: Int, f_addr: Int, nf: Int, dst_addr: Int) abi("C"):
    var a = FPtr(unsafe_from_address=a_addr)
    var v = FPtr(unsafe_from_address=v_addr)
    var faces = IPtr(unsafe_from_address=f_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(na):
        var best = 1e300
        for j in range(nf):
            var k = 3 * j
            var d = point_triangle_sqdist(a, 3 * i, v, 3 * Int(faces[k]), 3 * Int(faces[k + 1]), 3 * Int(faces[k + 2]))
            if d < best:
                best = d
        dst[i] = sqrt(best)


@export("mvd_densify_once")
def mvd_densify_once(p_addr: Int, n: Int, target: Float64, nclosest: Int, limit: Int, dst_addr: Int) abi("C") -> Int:
    var p = FPtr(unsafe_from_address=p_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(n):
        dst[3 * i] = p[3 * i]
        dst[3 * i + 1] = p[3 * i + 1]
        dst[3 * i + 2] = p[3 * i + 2]
    var count = n
    for i in range(n):
        for j in range(i + 1, n):
            var closer = 0
            var dij = sqdist3(p, 3 * i, p, 3 * j)
            for q in range(n):
                if q != i and sqdist3(p, 3 * i, p, 3 * q) < dij:
                    closer += 1
            if closer < nclosest and dij > target * target:
                if count >= limit:
                    return -count
                dst[3 * count] = 0.5 * (p[3 * i] + p[3 * j])
                dst[3 * count + 1] = 0.5 * (p[3 * i + 1] + p[3 * j + 1])
                dst[3 * count + 2] = 0.5 * (p[3 * i + 2] + p[3 * j + 2])
                count += 1
    return count


def emit_vertex(dst: FPtr, count: Int, x: Float64, y: Float64, z: Float64) -> Int:
    dst[3 * count] = x
    dst[3 * count + 1] = y
    dst[3 * count + 2] = z
    return count


@export("mvd_clip_plane")
def mvd_clip_plane(v_addr: Int, f_addr: Int, nf: Int, ox: Float64, oy: Float64, oz: Float64, nx: Float64, ny: Float64, nz: Float64, invert: Int, vdst_addr: Int, fdst_addr: Int) abi("C") -> Int:
    var v = FPtr(unsafe_from_address=v_addr)
    var faces = IPtr(unsafe_from_address=f_addr)
    var vd = FPtr(unsafe_from_address=vdst_addr)
    var fd = IPtr(unsafe_from_address=fdst_addr)
    var vc = 0
    var fc = 0
    for fi in range(nf):
        var ids0 = Int(faces[3 * fi])
        var ids1 = Int(faces[3 * fi + 1])
        var ids2 = Int(faces[3 * fi + 2])
        var q0 = 0
        var q1 = 0
        var q2 = 0
        var q3 = 0
        var pn = 0
        for e in range(3):
            var ia = ids0
            var ib = ids1
            if e == 1:
                ia = ids1
                ib = ids2
            elif e == 2:
                ia = ids2
                ib = ids0
            var ax = v[3 * ia]
            var ay = v[3 * ia + 1]
            var az = v[3 * ia + 2]
            var bx = v[3 * ib]
            var by = v[3 * ib + 1]
            var bz = v[3 * ib + 2]
            var da = (ax - ox) * nx + (ay - oy) * ny + (az - oz) * nz
            var db = (bx - ox) * nx + (by - oy) * ny + (bz - oz) * nz
            var ina = da >= 0.0
            var inb = db >= 0.0
            if invert != 0:
                ina = not ina
                inb = not inb
            if ina:
                var q = emit_vertex(vd, vc, ax, ay, az)
                if pn == 0: q0 = q
                elif pn == 1: q1 = q
                elif pn == 2: q2 = q
                else: q3 = q
                vc += 1
                pn += 1
            if ina != inb:
                var t = da / (da - db)
                var q = emit_vertex(vd, vc, ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az))
                if pn == 0: q0 = q
                elif pn == 1: q1 = q
                elif pn == 2: q2 = q
                else: q3 = q
                vc += 1
                pn += 1
        if pn >= 3:
            fd[3 * fc] = Int64(q0)
            fd[3 * fc + 1] = Int64(q1)
            fd[3 * fc + 2] = Int64(q2)
            fc += 1
            if pn == 4:
                fd[3 * fc] = Int64(q0)
                fd[3 * fc + 1] = Int64(q2)
                fd[3 * fc + 2] = Int64(q3)
                fc += 1
    return fc


@export("mvd_cell_metrics")
def mvd_cell_metrics(v_addr: Int, f_addr: Int, nf: Int, metric: Int, dst_addr: Int) abi("C"):
    var v = FPtr(unsafe_from_address=v_addr)
    var faces = IPtr(unsafe_from_address=f_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)
    for i in range(nf):
        var a = 3 * Int(faces[3 * i])
        var b = 3 * Int(faces[3 * i + 1])
        var c = 3 * Int(faces[3 * i + 2])
        var ab = sqrt(sqdist3(v, a, v, b))
        var bc = sqrt(sqdist3(v, b, v, c))
        var ca = sqrt(sqdist3(v, c, v, a))
        var ex = v[b] - v[a]
        var ey = v[b + 1] - v[a + 1]
        var ez = v[b + 2] - v[a + 2]
        var fx = v[c] - v[a]
        var fy = v[c + 1] - v[a + 1]
        var fz = v[c + 2] - v[a + 2]
        var sx = ey * fz - ez * fy
        var sy = ez * fx - ex * fz
        var sz = ex * fy - ey * fx
        var area = 0.5 * sqrt(sx * sx + sy * sy + sz * sz)
        var mn = min(ab, min(bc, ca))
        var mx = max(ab, max(bc, ca))
        if area <= 1e-300 or mn <= 1e-300:
            dst[i] = 0.0
        elif metric == 0:
            dst[i] = mx / mn
        elif metric == 1:
            dst[i] = mx * mx / (2.0 * sqrt(3.0) * area)
        elif metric == 2:
            var sem = 0.5 * (ab + bc + ca)
            var rin = area / sem
            var rout = ab * bc * ca / (4.0 * area)
            dst[i] = rout / (2.0 * rin)
        elif metric == 6 or metric == 8:
            var aa = acos((ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca))
            var bb = acos((ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc))
            var cc = 3.141592653589793 - aa - bb
            dst[i] = (min(aa, min(bb, cc)) if metric == 6 else max(aa, max(bb, cc))) * 57.29577951308232
        elif metric == 19 or metric == 28:
            dst[i] = area if metric == 28 else 0.0
        else:
            dst[i] = mx / mn
