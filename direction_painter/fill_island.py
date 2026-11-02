from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve, zoom
from .scene import Mesh, DirectionLayer


def build_island_mask(mesh: Mesh, width: int, height: int) -> np.ndarray:
    """
    Rasterise all UV triangles into a boolean mask of shape (height, width).
    A True cell means that texel lies inside at least one UV triangle.
    """
    mask = np.zeros((height, width), dtype=bool)
    uvs = mesh.uvs

    for face in mesh.indices:
        uv0 = uvs[face[0]]
        uv1 = uvs[face[1]]
        uv2 = uvs[face[2]]
        _rasterize_triangle(mask, uv0, uv1, uv2, width, height)

    return mask


def fill_island(
    layer: DirectionLayer,
    mesh: Mesh,
    painted_mask: np.ndarray,
) -> None:
    """
    Propagate painted directions across the UV island using multi-scale
    Jacobi diffusion. Painted texels act as fixed boundary conditions.

    painted_mask: (H, W) bool — True where the user has painted.
    """
    H, W = layer.height, layer.width

    if mesh.island_mask is None:
        mesh.island_mask = build_island_mask(mesh, W, H)

    island    = mesh.island_mask.astype(bool)   # (H, W)
    free_mask = island & ~painted_mask           # texels to be filled

    # Work in complex space: z = R + i*G = e^(iθ)
    field = layer.texture.copy().astype(np.float64)   # (H, W, 3)

    scales = sorted({max(1, W // 16), max(1, W // 8),
                     max(1, W // 4),  max(1, W // 2), W})

    avg_kernel = np.ones((3, 3), dtype=np.float64) / 9.0

    prev_field = None

    for s in scales:
        scale_factor = s / W

        if prev_field is not None:
            # Upsample warm-start from previous scale
            field_s = zoom(prev_field, (s / prev_field.shape[0],
                                        s / prev_field.shape[1], 1.0),
                           order=1)
        else:
            field_s = zoom(field, (scale_factor, scale_factor, 1.0), order=1)

        island_s   = zoom(island.astype(float),   scale_factor, order=0).astype(bool)
        painted_s  = zoom(painted_mask.astype(float), scale_factor, order=0).astype(bool)
        free_s     = island_s & ~painted_s

        n_iters = 50 if s < W else 200
        for _ in range(n_iters):
            for ch in range(2):
                smoothed = convolve(field_s[..., ch], avg_kernel, mode="constant")
                field_s[..., ch] = np.where(free_s, smoothed, field_s[..., ch])

        prev_field = field_s

    # Upsample final result back to full resolution
    final = zoom(prev_field, (H / prev_field.shape[0], W / prev_field.shape[1], 1.0),
                 order=1)

    # Normalise only the direction channels (R, G) to unit vectors;
    # the B channel (intensity) is not part of the direction vector.
    mag = np.linalg.norm(final[..., :2], axis=2, keepdims=True)
    final[..., :2] /= np.maximum(mag, 1e-8)

    # Write back only the free texels
    layer.texture[free_mask] = final[free_mask].astype(np.float32)
    layer.dirty = True


# ─────────────────────────────────────────────────────────────────────────────
# UV triangle rasterisation
# ─────────────────────────────────────────────────────────────────────────────

def _rasterize_triangle(
    mask: np.ndarray,
    uv0: np.ndarray,
    uv1: np.ndarray,
    uv2: np.ndarray,
    W: int,
    H: int,
) -> None:
    """Fill texels that lie inside the UV triangle into *mask*."""
    # Convert UV [0,1] to pixel coordinates
    p0 = np.array([uv0[0] * W, uv0[1] * H])
    p1 = np.array([uv1[0] * W, uv1[1] * H])
    p2 = np.array([uv2[0] * W, uv2[1] * H])

    xmin = max(0,   int(np.floor(min(p0[0], p1[0], p2[0]))))
    xmax = min(W-1, int(np.ceil( max(p0[0], p1[0], p2[0]))))
    ymin = max(0,   int(np.floor(min(p0[1], p1[1], p2[1]))))
    ymax = min(H-1, int(np.ceil( max(p0[1], p1[1], p2[1]))))

    if xmin > xmax or ymin > ymax:
        return

    xs = np.arange(xmin, xmax + 1, dtype=np.float64)
    ys = np.arange(ymin, ymax + 1, dtype=np.float64)
    gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)   # sample at texel centres

    # Edge function signs
    def edge(ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    e0 = edge(p0[0], p0[1], p1[0], p1[1], gx, gy)
    e1 = edge(p1[0], p1[1], p2[0], p2[1], gx, gy)
    e2 = edge(p2[0], p2[1], p0[0], p0[1], gx, gy)

    inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
    # Handle clockwise winding
    if not inside.any():
        inside = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)

    mask[ymin : ymax + 1, xmin : xmax + 1] |= inside
