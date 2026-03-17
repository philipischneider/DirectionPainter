from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Mesh:
    vertices:   np.ndarray   # (N, 3) float32
    normals:    np.ndarray   # (N, 3) float32
    tangents:   np.ndarray   # (N, 4) float32  — w is handedness sign
    uvs:        np.ndarray   # (N, 2) float32
    indices:    np.ndarray   # (F, 3) uint32
    bitangents: np.ndarray   # (N, 3) float32  — precomputed
    uv_aspect:  np.ndarray   # (N,)   float32  — |∂pos/∂u| / |∂pos/∂v| per vertex

    # Cached UV-space lookup data, populated by Renderer on mesh upload
    face_uv_centroids:  Optional[np.ndarray] = None  # (F, 2) float32
    face_pos_centroids: Optional[np.ndarray] = None  # (F, 3) float32
    face_tangents:      Optional[np.ndarray] = None  # (F, 3) float32
    face_bitangents:    Optional[np.ndarray] = None  # (F, 3) float32

    # UV island mask, computed on demand by fill_island
    island_mask: Optional[np.ndarray] = None   # (H, W) uint8

    @property
    def face_count(self) -> int:
        return self.indices.shape[0]

    @property
    def vertex_count(self) -> int:
        return self.vertices.shape[0]


@dataclass
class DirectionLayer:
    name:    str
    width:   int
    height:  int
    texture: np.ndarray   # (H, W, 2) float32  R=cos(θ), G=sin(θ)
    # moderngl.Texture set by Renderer; None until mesh is loaded
    gpu_texture: object = None
    dirty: bool = False

    @staticmethod
    def create(name: str, width: int, height: int) -> DirectionLayer:
        """Create a layer pre-filled with the neutral direction (cos0, sin0) = (1, 0)."""
        tex = np.zeros((height, width, 2), dtype=np.float32)
        tex[..., 0] = 1.0   # cos(0) = 1 → along +U axis
        return DirectionLayer(name=name, width=width, height=height, texture=tex)


@dataclass
class Scene:
    mesh: Optional[Mesh] = None
    layers: List[DirectionLayer] = field(default_factory=list)
    active_layer_idx: int = 0

    @property
    def active_layer(self) -> Optional[DirectionLayer]:
        if not self.layers:
            return None
        idx = max(0, min(self.active_layer_idx, len(self.layers) - 1))
        return self.layers[idx]
