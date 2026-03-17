from __future__ import annotations
import math
import numpy as np


class OrbitCamera:
    def __init__(self) -> None:
        self.target   = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 3.0
        self.yaw      = 0.0    # radians, horizontal
        self.pitch    = 0.3    # radians, vertical (clamped)
        self.fov_y    = math.radians(45.0)
        self.near     = 0.001
        self.far      = 1000.0

    # ------------------------------------------------------------------ #
    # Navigation                                                           #
    # ------------------------------------------------------------------ #

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw   += dx
        self.pitch  = float(np.clip(self.pitch + dy,
                                    -math.pi / 2 + 0.01,
                                     math.pi / 2 - 0.01))

    def pan(self, dx: float, dy: float) -> None:
        speed = self.distance * 0.001
        right = self._right()
        up    = self._cam_up()
        self.target += right * (-dx * speed) + up * (dy * speed)

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance * (1.0 - delta * 0.1),
                                      0.001, 10000.0))

    def fit_to_mesh(self, vertices: np.ndarray) -> None:
        center = vertices.mean(axis=0)
        self.target = center.astype(np.float32)
        extents = vertices.max(axis=0) - vertices.min(axis=0)
        self.distance = float(np.linalg.norm(extents)) * 1.5

    # ------------------------------------------------------------------ #
    # Derived geometry                                                     #
    # ------------------------------------------------------------------ #

    def eye(self) -> np.ndarray:
        x = math.cos(self.pitch) * math.sin(self.yaw)
        y = math.sin(self.pitch)
        z = math.cos(self.pitch) * math.cos(self.yaw)
        return (self.target + np.array([x, y, z], dtype=np.float32)
                * self.distance)

    def _forward(self) -> np.ndarray:
        e = self.eye()
        d = self.target - e
        n = np.linalg.norm(d)
        return d / n if n > 1e-8 else np.array([0, 0, -1], dtype=np.float32)

    def _right(self) -> np.ndarray:
        fwd       = self._forward()
        world_up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(fwd, world_up)
        n = np.linalg.norm(r)
        if n < 1e-6:
            r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            r /= n
        return r

    def _cam_up(self) -> np.ndarray:
        return np.cross(self._right(), self._forward())

    # ------------------------------------------------------------------ #
    # Matrices (row-major, must transpose before uploading to OpenGL)     #
    # ------------------------------------------------------------------ #

    def view_matrix(self) -> np.ndarray:
        e   = self.eye()
        fwd = self._forward()
        r   = self._right()
        u   = np.cross(r, fwd)   # camera up

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = r
        m[1, :3] = u
        m[2, :3] = -fwd
        m[0, 3]  = -float(np.dot(r, e))
        m[1, 3]  = -float(np.dot(u, e))
        m[2, 3]  =  float(np.dot(fwd, e))
        return m

    def proj_matrix(self, aspect: float) -> np.ndarray:
        f = 1.0 / math.tan(self.fov_y / 2.0)
        nf = self.near - self.far
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (self.far + self.near) / nf
        m[2, 3] = 2.0 * self.far * self.near / nf
        m[3, 2] = -1.0
        return m

    # ------------------------------------------------------------------ #
    # Picking helpers                                                      #
    # ------------------------------------------------------------------ #

    def unproject_ray(self, sx: float, sy: float,
                      viewport_w: int, viewport_h: int
                      ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (ray_origin, ray_direction) in world space for a screen pixel.
        sx, sy are pixel coords with (0,0) at top-left.
        """
        aspect  = viewport_w / max(viewport_h, 1)
        V       = self.view_matrix()
        P       = self.proj_matrix(aspect)
        VP      = P @ V
        inv_VP  = np.linalg.inv(VP)

        ndc_x =  2.0 * sx / viewport_w - 1.0
        ndc_y = -(2.0 * sy / viewport_h - 1.0)   # flip y (screen y-down → NDC y-up)

        near_h = inv_VP @ np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        far_h  = inv_VP @ np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)
        near_w = near_h[:3] / near_h[3]
        far_w  = far_h[:3]  / far_h[3]

        direction = far_w - near_w
        n = np.linalg.norm(direction)
        direction = direction / n if n > 1e-8 else direction
        return near_w.astype(np.float32), direction.astype(np.float32)
