"""
spline_painter.py — Catmull-Rom spline-based flow map painting.

The user places control points on the mesh surface (UV space).
On commit the spline is sampled densely and its tangent direction is
painted into the flow map texture with a configurable radius of influence
— same blending used by Painter (doubled-angle, smooth falloff).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .scene import Scene


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplineParams:
    radius:   float = 0.05   # brush radius in UV space [0..1]
    strength: float = 0.90   # blend weight per painted sample


class SplinePainter:
    def __init__(self, scene: Scene) -> None:
        self.scene  = scene
        self.params = SplineParams()

        self._ctrl_uvs: List[np.ndarray] = []
        self._preview_uvs:  Optional[np.ndarray] = None   # (N, 2) sampled curve
        self._preview_tans: Optional[np.ndarray] = None   # (N, 2) unit tangents

    # ──────────────────────────────────────────────────────────────────────

    @property
    def has_points(self) -> bool:
        return len(self._ctrl_uvs) > 0

    @property
    def ctrl_uvs(self) -> List[np.ndarray]:
        return self._ctrl_uvs

    @property
    def preview_uvs(self) -> Optional[np.ndarray]:
        return self._preview_uvs

    def add_point(self, uv: np.ndarray) -> None:
        self._ctrl_uvs.append(uv.copy())
        self._rebuild_preview()

    def cancel(self) -> None:
        self._ctrl_uvs     = []
        self._preview_uvs  = None
        self._preview_tans = None

    def commit(self, renderer) -> None:
        """Rasterize the current spline into the active layer."""
        layer = self.scene.active_layer
        if layer is None or len(self._ctrl_uvs) < 2:
            self.cancel()
            return

        if self._preview_uvs is None:
            self._rebuild_preview()

        pts  = self._preview_uvs
        tans = self._preview_tans
        if pts is None or len(pts) == 0:
            self.cancel()
            return

        H, W     = layer.height, layer.width
        radius   = self.params.radius
        strength = self.params.strength
        dirty: Optional[Tuple[int, int, int, int]] = None

        for i in range(len(pts)):
            uv    = pts[i]
            tan   = tans[i]
            angle = float(np.arctan2(tan[1], tan[0]))

            cx = int(uv[0] * W)
            cy = int(uv[1] * H)
            rx = max(1, int(radius * W))
            ry = max(1, int(radius * H))

            x0 = max(0, cx - rx);  x1 = min(W, cx + rx + 1)
            y0 = max(0, cy - ry);  y1 = min(H, cy + ry + 1)

            ty_idx, tx_idx = np.mgrid[y0:y1, x0:x1]
            du   = (tx_idx - cx) / (rx + 1e-6)
            dv   = (ty_idx - cy) / (ry + 1e-6)
            dist = np.sqrt(du * du + dv * dv)
            weight = strength * (1.0 - dist) ** 2 * (dist <= 1.0)

            sub         = layer.texture[y0:y1, x0:x1, :]
            existing_a2 = np.arctan2(sub[..., 1], sub[..., 0])
            new_a2      = np.full_like(existing_a2, 2.0 * angle)
            diff        = (new_a2 - existing_a2 + np.pi) % (2 * np.pi) - np.pi
            blended_a2  = existing_a2 + diff * weight
            layer.texture[y0:y1, x0:x1, 0] = np.cos(blended_a2)
            layer.texture[y0:y1, x0:x1, 1] = np.sin(blended_a2)

            dirty = _union_rect(dirty, (x0, y0, x1 - x0, y1 - y0))

        if dirty is not None:
            dx, dy, dw, dh = dirty
            renderer.upload_dirty_region(layer, dx, dy, dw, dh)
            if self.scene.mesh is not None:
                renderer.update_ticks(self.scene.mesh, layer)

        self.cancel()

    # ──────────────────────────────────────────────────────────────────────
    # Internal — Catmull-Rom sampling
    # ──────────────────────────────────────────────────────────────────────

    def _rebuild_preview(self) -> None:
        pts = self._ctrl_uvs
        n   = len(pts)

        if n == 0:
            self._preview_uvs  = None
            self._preview_tans = None
            return

        if n == 1:
            self._preview_uvs  = np.array([pts[0]], dtype=np.float32)
            self._preview_tans = np.array([[1.0, 0.0]], dtype=np.float32)
            return

        steps_per_seg = max(30, 200 // max(n - 1, 1))
        sampled:  list = []
        tangents: list = []

        for seg in range(n - 1):
            p0 = pts[max(0, seg - 1)]
            p1 = pts[seg]
            p2 = pts[seg + 1]
            p3 = pts[min(n - 1, seg + 2)]

            for j in range(steps_per_seg):
                t   = j / steps_per_seg
                pos = _catmull_rom(p0, p1, p2, p3, t)
                tan = _catmull_rom_tan(p0, p1, p2, p3, t)
                mag = np.linalg.norm(tan)
                tan = tan / mag if mag > 1e-8 else np.array([1.0, 0.0])
                sampled.append(pos)
                tangents.append(tan)

        # Final endpoint
        p0 = pts[max(0, n - 3)]
        p1 = pts[max(0, n - 2)]
        p2 = pts[n - 1]
        pos = _catmull_rom(p0, p1, p2, p2, 1.0)
        tan = _catmull_rom_tan(p0, p1, p2, p2, 1.0)
        mag = np.linalg.norm(tan)
        tan = tan / mag if mag > 1e-8 else np.array([1.0, 0.0])
        sampled.append(pos)
        tangents.append(tan)

        self._preview_uvs  = np.array(sampled,  dtype=np.float32)
        self._preview_tans = np.array(tangents, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _catmull_rom(p0, p1, p2, p3, t: float) -> np.ndarray:
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
    )


def _catmull_rom_tan(p0, p1, p2, p3, t: float) -> np.ndarray:
    return 0.5 * (
        (-p0 + p2)
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * 2 * t
        + (-p0 + 3 * p1 - 3 * p2 + p3) * 3 * t ** 2
    )


def _union_rect(
    a: Optional[Tuple[int, int, int, int]],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    if a is None:
        return b
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx);  y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw);  y1 = max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0
