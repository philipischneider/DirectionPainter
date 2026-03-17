"""
painter.py — Brush painting logic for all brush modes.

Stroke:   direction = mouse-drag direction projected onto surface
Radial:   direction = outward/inward from a centre point
Circular: direction = tangent to circles around a centre point
Comb:     direction = weighted average of UV-space neighbours (smoothing)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

from .scene import Scene, Mesh, DirectionLayer
from .undo_stack import UndoStack, UndoEntry
from .camera import OrbitCamera

# Minimum squared screen-space displacement (pixels²) before updating the
# direction anchor.  Using a large enough delta makes the computed angle
# robust to sub-pixel cursor jitter that would otherwise corrupt the EMA.
_MIN_DIR_DIST_SQ = 100  # 10 px


@dataclass
class BrushParams:
    mode:     str   = "Stroke"   # "Stroke" | "Radial" | "Circular" | "Comb"
    radius:   float = 0.05       # brush radius in UV space [0..1]
    strength: float = 0.8        # blend weight per paint event [0..1]
    inward:   bool  = False      # for Radial mode: inward instead of outward


class Painter:
    def __init__(self, scene: Scene, undo_stack: UndoStack) -> None:
        self.scene  = scene
        self.undo   = undo_stack
        self.brush  = BrushParams()

        self._active          = False
        self._center_uv:  Optional[np.ndarray] = None   # Radial / Circular
        self._prev_screen: Optional[Tuple[int, int]] = None
        self._dirty_region: Optional[Tuple[int, int, int, int]] = None
        # Pre-stroke full snapshot — small enough when stored per dirty region
        self._pre_snapshot: Optional[UndoEntry] = None
        # Smoothed stroke direction (EMA) — avoids concentric artifacts at
        # stroke start/end where instantaneous delta is zero or degenerate
        self._smooth_angle: Optional[float] = None
        # Anchor position used to compute stroke direction.  Only advances
        # when the cursor has moved ≥ √_MIN_DIR_DIST_SQ pixels, so the
        # direction vector is always long enough to be numerically stable.
        self._dir_anchor: Optional[Tuple[int, int]] = None

    # ──────────────────────────────────────────────────────────────────────
    # Stroke lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def begin_stroke(
        self,
        tri_id: int,
        bary_u: float,
        bary_v: float,
        camera: OrbitCamera,
        screen_pos: Tuple[int, int],
        viewport_size: Tuple[int, int],
    ) -> None:
        if not self.scene.layers:
            return
        layer = self.scene.active_layer
        if layer is None:
            return

        self._active       = True
        self._dirty_region = None
        self._prev_screen  = screen_pos
        self._smooth_angle = None   # reset EMA each stroke
        self._dir_anchor   = None   # reset anchor; will be set on first move

        # For Radial / Circular, record UV centre at click position
        uv_hit = self._interpolate_uv(tri_id, bary_u, bary_v)
        self._center_uv = uv_hit.copy()

        # Take a full-layer snapshot for undo.
        # We'll trim it to the dirty region in end_stroke.
        self._pre_snapshot = self.undo.snapshot_region(
            self.scene.active_layer_idx,
            layer.texture,
            (0, 0, layer.width, layer.height),
        )

        # Stroke mode needs movement to determine direction — skip first point.
        # All other modes (Radial, Circular, Comb) paint immediately.
        if self.brush.mode != "Stroke":
            self.paint_point(tri_id, bary_u, bary_v, camera, screen_pos, viewport_size)

    def paint_point(
        self,
        tri_id: int,
        bary_u: float,
        bary_v: float,
        camera: OrbitCamera,
        screen_pos: Tuple[int, int],
        viewport_size: Tuple[int, int],
    ) -> None:
        if not self._active:
            return
        layer = self.scene.active_layer
        if layer is None:
            return

        mesh = self.scene.mesh
        H, W = layer.height, layer.width

        uv_hit   = self._interpolate_uv(tri_id, bary_u, bary_v)
        pos_hit  = self._interpolate_attr(tri_id, bary_u, bary_v, mesh.vertices)
        norm_hit = _normalize(self._interpolate_attr(tri_id, bary_u, bary_v, mesh.normals))
        tan_hit  = _normalize(self._interpolate_attr(tri_id, bary_u, bary_v, mesh.tangents[:, :3]))
        bit_hit  = _normalize(self._interpolate_attr(tri_id, bary_u, bary_v, mesh.bitangents))

        # Compute the paint angle for this point
        if self.brush.mode == "Stroke":
            # Use a stable anchor that only advances when the cursor has moved
            # ≥ √_MIN_DIR_DIST_SQ px.  Computing the angle over a guaranteed-
            # large delta prevents noisy micro-movements (1-2 px at 60 fps)
            # from injecting random directions that cause circular artefacts.
            anchor = self._dir_anchor if self._dir_anchor is not None \
                     else self._prev_screen
            if anchor is not None:
                adx = screen_pos[0] - anchor[0]
                ady = screen_pos[1] - anchor[1]
                far_enough = (adx * adx + ady * ady) >= _MIN_DIR_DIST_SQ
            else:
                far_enough = False

            if far_enough:
                raw = self._stroke_angle(
                    screen_pos, anchor, camera, viewport_size,
                    pos_hit, norm_hit, tan_hit, bit_hit,
                )
                if raw is not None:
                    if self._smooth_angle is None:
                        self._smooth_angle = raw
                    else:
                        diff = (raw - self._smooth_angle + np.pi) % (2 * np.pi) - np.pi
                        self._smooth_angle = self._smooth_angle + diff * 0.15
                # Advance anchor regardless (even if ray missed, keep moving)
                self._dir_anchor = screen_pos

            if self._smooth_angle is None:
                # No valid direction established yet — skip this sample.
                # Do NOT advance _prev_screen here: it acts as the initial
                # anchor (= begin_stroke position) so the displacement
                # accumulates until the cursor moves ≥ √_MIN_DIR_DIST_SQ px.
                return
            angle: Optional[float] = self._smooth_angle
        elif self.brush.mode == "Radial":
            angle = self._radial_angle(uv_hit, self._center_uv, outward=not self.brush.inward)
        elif self.brush.mode == "Circular":
            angle = self._circular_angle(uv_hit, self._center_uv)
        else:
            angle = None   # Comb: no fixed angle

        # Brush footprint in texture space
        cx = int(uv_hit[0] * W)
        cy = int(uv_hit[1] * H)
        rx = max(1, int(self.brush.radius * W))
        ry = max(1, int(self.brush.radius * H))

        x0 = max(0, cx - rx);  x1 = min(W, cx + rx + 1)
        y0 = max(0, cy - ry);  y1 = min(H, cy + ry + 1)

        ty_idx, tx_idx = np.mgrid[y0:y1, x0:x1]                    # (h, w)
        du = (tx_idx - cx) / (rx + 1e-6)
        dv = (ty_idx - cy) / (ry + 1e-6)
        dist = np.sqrt(du * du + dv * dv)
        within = dist <= 1.0
        weight = self.brush.strength * (1.0 - dist) ** 2 * within   # smooth falloff

        sub = layer.texture[y0:y1, x0:x1, :]                        # (h, w, 2)
        # Texture stores (cos 2θ, sin 2θ) — doubled-angle representation.
        # Blend in the doubled space so that orientations θ and θ+π are
        # treated as identical (π-periodic orientation field).
        existing_a2 = np.arctan2(sub[..., 1], sub[..., 0])          # (h, w) = 2θ

        if self.brush.mode == "Comb":
            # _comb_angle averages (R,G) of the doubled-angle texture, so
            # its arctan2 result is already a doubled angle — use directly.
            new_a2 = self._comb_angle(layer, y0, y1, x0, x1)
        else:
            new_a2 = np.full_like(existing_a2, 2.0 * angle)

        blended_a2 = _angle_slerp_arr(existing_a2, new_a2, weight)
        layer.texture[y0:y1, x0:x1, 0] = np.cos(blended_a2)
        layer.texture[y0:y1, x0:x1, 1] = np.sin(blended_a2)

        self._dirty_region = _union_rect(self._dirty_region, (x0, y0, x1 - x0, y1 - y0))
        self._prev_screen  = screen_pos

    def end_stroke(self, renderer) -> None:
        """Finalise the stroke: trim undo entry, upload GPU texture, refresh ticks."""
        if not self._active:
            return
        self._active = False

        layer = self.scene.active_layer
        if layer is None or self._dirty_region is None:
            return

        # Trim the pre-stroke snapshot to the actual dirty region
        dx, dy, dw, dh = self._dirty_region
        if self._pre_snapshot is not None:
            sx, sy, sw, sh = self._pre_snapshot.region
            # Extract the sub-region of the full snapshot that corresponds to dirty_region
            snap = self._pre_snapshot.snapshot[dy:dy+dh, dx:dx+dw, :]
            entry = type(self._pre_snapshot)(
                layer_index=self.scene.active_layer_idx,
                region=(dx, dy, dw, dh),
                snapshot=snap.copy(),
            )
            self.undo.push(entry)

        # Upload changed region to GPU
        renderer.upload_dirty_region(layer, dx, dy, dw, dh)

        # Refresh tick overlay
        if self.scene.mesh is not None:
            renderer.update_ticks(self.scene.mesh, layer)

        self._dirty_region  = None
        self._pre_snapshot  = None

    # ──────────────────────────────────────────────────────────────────────
    # Per-mode angle computations
    # ──────────────────────────────────────────────────────────────────────

    def _stroke_angle(
        self,
        curr_screen: Tuple[int, int],
        prev_screen: Optional[Tuple[int, int]],
        camera: OrbitCamera,
        viewport_size: Tuple[int, int],
        pos_hit: np.ndarray,
        norm_hit: np.ndarray,
        tan_hit: np.ndarray,
        bit_hit: np.ndarray,
    ) -> Optional[float]:
        """Return the angle in tangent space for the current movement delta.

        Returns None when there is no meaningful movement, so the caller can
        preserve the last smoothed direction instead of falling back to 0.0.
        """
        if prev_screen is None or prev_screen == curr_screen:
            return None

        vw, vh = viewport_size
        o1, d1 = camera.unproject_ray(prev_screen[0], prev_screen[1], vw, vh)
        o2, d2 = camera.unproject_ray(curr_screen[0], curr_screen[1], vw, vh)

        p1 = _ray_plane_intersect(o1, d1, pos_hit, norm_hit)
        p2 = _ray_plane_intersect(o2, d2, pos_hit, norm_hit)

        if p1 is None or p2 is None:
            # Fallback: use screen-space delta projected onto camera plane
            dx = curr_screen[0] - prev_screen[0]
            dy = curr_screen[1] - prev_screen[1]
            right = camera._right()
            up    = camera._cam_up()
            world_dir = _normalize(right * dx - up * dy)
            p1 = pos_hit
            p2 = pos_hit + world_dir * 0.01

        stroke_world = p2 - p1
        mag = np.linalg.norm(stroke_world)
        if mag < 1e-8:
            return None   # degenerate delta — caller keeps last smooth angle
        stroke_world /= mag

        dir_t = float(np.dot(stroke_world, tan_hit))
        dir_b = float(np.dot(stroke_world, bit_hit))
        return float(np.arctan2(dir_b, dir_t))

    @staticmethod
    def _radial_angle(uv_hit: np.ndarray, center_uv: np.ndarray,
                      outward: bool = True) -> float:
        delta = uv_hit - center_uv
        if np.linalg.norm(delta) < 1e-8:
            return 0.0
        if not outward:
            delta = -delta
        return float(np.arctan2(delta[1], delta[0]))

    @staticmethod
    def _circular_angle(uv_hit: np.ndarray, center_uv: np.ndarray) -> float:
        delta = uv_hit - center_uv
        if np.linalg.norm(delta) < 1e-8:
            return 0.0
        # Rotate 90° CCW: (x, y) → (−y, x)
        perp = np.array([-delta[1], delta[0]])
        return float(np.arctan2(perp[1], perp[0]))

    @staticmethod
    def _comb_angle(
        layer: DirectionLayer,
        y0: int, y1: int, x0: int, x1: int,
    ) -> np.ndarray:
        """Return the weighted-average angle of the neighbourhood."""
        from scipy.ndimage import convolve
        kernel = np.ones((3, 3), dtype=np.float32) / 9.0
        r_avg = convolve(layer.texture[..., 0], kernel, mode="nearest")[y0:y1, x0:x1]
        g_avg = convolve(layer.texture[..., 1], kernel, mode="nearest")[y0:y1, x0:x1]
        return np.arctan2(g_avg, r_avg)

    # ──────────────────────────────────────────────────────────────────────
    # Mesh attribute interpolation
    # ──────────────────────────────────────────────────────────────────────

    def _interpolate_uv(self, tri_id: int, u: float, v: float) -> np.ndarray:
        return self._interpolate_attr(tri_id, u, v, self.scene.mesh.uvs)

    def _interpolate_attr(
        self,
        tri_id: int,
        u: float,
        v: float,
        attr: np.ndarray,
    ) -> np.ndarray:
        face = self.scene.mesh.indices[tri_id]
        bary = np.array([u, v, 1.0 - u - v], dtype=np.float32)
        return bary @ attr[face]


# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _angle_slerp_arr(
    a: np.ndarray, b: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Shortest-arc angle interpolation, vectorised."""
    diff = (b - a + np.pi) % (2 * np.pi) - np.pi   # wrap to (−π, π]
    return a + diff * t


def _ray_plane_intersect(
    origin: np.ndarray,
    direction: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[np.ndarray]:
    denom = float(np.dot(plane_normal, direction))
    if abs(denom) < 1e-8:
        return None
    t = float(np.dot(plane_normal, plane_point - origin)) / denom
    if t < 0:
        return None
    return origin + t * direction


def _union_rect(
    a: Optional[Tuple[int, int, int, int]],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    if a is None:
        return b
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0 = min(ax, bx)
    y0 = min(ay, by)
    x1 = max(ax + aw, bx + bw)
    y1 = max(ay + ah, by + bh)
    return x0, y0, x1 - x0, y1 - y0
