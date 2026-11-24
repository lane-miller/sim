"""Test-only reference (slow, pure numpy) cut-cell fractions: clip + sweep + divergence-theorem beta.

Conventions: cell (i,j,k) spans [x0+i h, x0+(i+1)h] etc.; alpha[a] has shape with n_a+1 along axis a
(face planes 0..n_a); face normals = solid OUTWARD (pointing into fluid).
"""
import numpy as np


def clip_poly(poly, axis, val, keep_ge):
    """Sutherland-Hodgman clip of polygon (list of 3-vectors) against half-space x_axis >= val (or <=)."""
    out = []
    n = len(poly)
    if n == 0:
        return out
    for i in range(n):
        P, Q = poly[i], poly[(i + 1) % n]
        dp = (P[axis] - val) if keep_ge else (val - P[axis])
        dq = (Q[axis] - val) if keep_ge else (val - Q[axis])
        if dp >= 0:
            out.append(P)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append(P + t * (Q - P))
    return out


def poly_area_centroid(poly):
    """Area and centroid of a planar 3-D polygon via fan triangulation."""
    if len(poly) < 3:
        return 0.0, np.zeros(3)
    p0 = poly[0]
    A = 0.0
    C = np.zeros(3)
    for i in range(1, len(poly) - 1):
        a = 0.5 * np.linalg.norm(np.cross(poly[i] - p0, poly[i + 1] - p0))
        A += a
        C += a * (p0 + poly[i] + poly[i + 1]) / 3.0
    return A, (C / A if A > 0 else C)


def fractions(verts, faces, x0, h, shape, plane_tol=1e-9):
    x0 = np.asarray(x0, float)
    shape = tuple(shape)
    N = np.zeros(shape + (3,))       # sum A n
    M = np.zeros(shape)              # sum A (xbar - xc) . n
    Asum = np.zeros(shape)
    Ssum = np.zeros(shape + (3,))    # sum A x̄ (world)
    area_check = []
    for f in faces:
        P = verts[f]
        nvec = np.cross(P[1] - P[0], P[2] - P[0])
        A_tri = 0.5 * np.linalg.norm(nvec)
        if A_tri == 0:
            continue
        n = nvec / (2 * A_tri)
        # on-plane detection: all 3 vertices on the same grid plane of some axis
        onplane_axis = None
        for a in range(3):
            s = (P[:, a] - x0[a]) / h
            r = np.round(s)
            if np.all(np.abs(s - r) < plane_tol) and r[0] == r[1] == r[2]:
                onplane_axis = (a, int(r[0]))
                break
        lo = np.floor((P.min(0) - x0) / h).astype(int)
        hi = np.floor((P.max(0) - x0) / h).astype(int)
        lo = np.clip(lo, 0, np.array(shape) - 1)
        hi = np.clip(hi, 0, np.array(shape) - 1)
        if onplane_axis is not None:
            a, pi = onplane_axis
            layer = pi if n[a] > 0 else pi - 1
            lo[a] = hi[a] = layer
        tot = 0.0
        for i in range(lo[0], hi[0] + 1):
            for j in range(lo[1], hi[1] + 1):
                for k in range(lo[2], hi[2] + 1):
                    idx = (i, j, k)
                    cmin = x0 + h * np.array(idx)
                    cmax = cmin + h
                    poly = [p.copy() for p in P]
                    for a in range(3):
                        if onplane_axis is not None and a == onplane_axis[0]:
                            continue
                        poly = clip_poly(poly, a, cmin[a], True)
                        poly = clip_poly(poly, a, cmax[a], False)
                        if not poly:
                            break
                    A, C = poly_area_centroid(poly)
                    if A <= 0:
                        continue
                    xc = cmin + 0.5 * h
                    N[idx] += A * n
                    M[idx] += A * np.dot(C - xc, n)
                    Asum[idx] += A
                    Ssum[idx] += A * C
                    tot += A
        area_check.append(abs(tot - A_tri) / A_tri)
    # sweep
    alpha = []
    resid = 0.0
    for a in range(3):
        s = list(shape)
        s[a] += 1
        al = np.ones(s)
        for i in range(shape[a]):
            sl_c = [slice(None)] * 3
            sl_c[a] = i
            sl_f = [slice(None)] * 3
            sl_f[a] = i
            sl_f1 = [slice(None)] * 3
            sl_f1[a] = i + 1
            al[tuple(sl_f1)] = al[tuple(sl_f)] + N[tuple(sl_c) + (a,)] / h**2
        last = [slice(None)] * 3
        last[a] = shape[a]
        resid = max(resid, np.max(np.abs(al[tuple(last)] - 1.0)))
        alpha.append(al)
    sum_alpha = np.zeros(shape)
    for a in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[a] = slice(0, shape[a])
        sl1[a] = slice(1, shape[a] + 1)
        sum_alpha += alpha[a][tuple(sl0)] + alpha[a][tuple(sl1)]
    beta = ((h / 2) * h**2 * sum_alpha - M) / (3 * h**3)
    wall_area_eps = 1e-12 * h**2
    wall_mask = Asum > wall_area_eps
    flat_ids = np.flatnonzero(wall_mask.ravel())
    wall_area = Asum.ravel()[flat_ids]
    wall_area_vec = N.reshape(-1, 3)[flat_ids]
    wall_centroid = (Ssum.reshape(-1, 3)[flat_ids] / wall_area[:, np.newaxis])
    return dict(
        alpha=alpha,
        beta=beta,
        N=N,
        M=M,
        Asum=Asum,
        Ssum=Ssum,
        resid=resid,
        max_area_err=max(area_check) if area_check else 0.0,
        wall_cell=flat_ids.astype(np.int64),
        wall_area=wall_area,
        wall_centroid=wall_centroid,
        wall_area_vec=wall_area_vec,
    )


def box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return V, F


def cylinder_mesh(r, z0, z1, n):
    th = 2 * np.pi * np.arange(n) / n
    ring = np.c_[r * np.cos(th), r * np.sin(th)]
    V = np.vstack([np.c_[ring, np.full(n, z0)], np.c_[ring, np.full(n, z1)],
                   [[0, 0, z0]], [[0, 0, z1]]])
    cb, ct = 2 * n, 2 * n + 1
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i]]   # side, outward
        F += [[cb, j, i]]                           # bottom, -z
        F += [[ct, n + i, n + j]]                   # top, +z
    return V, np.array(F)


def mesh_volume(V, F):
    P = V[F]
    return np.sum(np.einsum('ij,ij->i', P[:, 0], np.cross(P[:, 1], P[:, 2]))) / 6.0
