"""
renderer.py — All OpenGL / moderngl rendering for DirectionPainter.

Two main passes:
  1. Picking pass  (offscreen RGBA32F FBO) — encodes triangle_id + barycentric
  2. Viewport pass (offscreen RGBA8 FBO)  — diffuse shading + direction tick overlay

The viewport FBO's colour texture (.glo) is passed to imgui.image() each frame.
"""
from __future__ import annotations
import struct
import numpy as np
import moderngl
from scipy.spatial import cKDTree

from .scene import Mesh, DirectionLayer, Scene
from .camera import OrbitCamera


# ─────────────────────────────────────────────────────────────────────────────
# GLSL sources
# ─────────────────────────────────────────────────────────────────────────────

_PICK_VERT = """
#version 330 core
in vec3  in_position;
in vec2  in_bary_uv;   // (1,0) / (0,1) / (0,0) per vertex
in float in_tri_id;

flat out float v_tri_id;
out vec2       v_bary_uv;

uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_tri_id   = in_tri_id;
    v_bary_uv  = in_bary_uv;
}
"""

_PICK_FRAG = """
#version 330 core
flat in float v_tri_id;
in vec2       v_bary_uv;

out vec4 out_color;

void main() {
    // R = triangle_id (exact float up to 2^24), G = bary_u, B = bary_v
    out_color = vec4(v_tri_id, v_bary_uv.x, v_bary_uv.y, 1.0);
}
"""

_VIEW_VERT = """
#version 330 core
in vec3  in_position;
in vec3  in_normal;
in vec2  in_uv;
in float in_uv_aspect;   // |∂pos/∂u| / |∂pos/∂v| per vertex

out vec3  v_normal;
out vec2  v_uv;
out float v_uv_aspect;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;

void main() {
    gl_Position = u_proj * u_view * u_model * vec4(in_position, 1.0);
    v_normal    = normalize(u_normal_mat * in_normal);
    v_uv        = in_uv;
    v_uv_aspect = in_uv_aspect;
}
"""

_VIEW_FRAG = """
#version 330 core
in vec3  v_normal;
in vec2  v_uv;
in float v_uv_aspect;   // |∂pos/∂u| / |∂pos/∂v|

out vec4 out_color;

uniform vec3      u_light_dir;
uniform sampler2D u_dir_tex;
uniform bool      u_has_dir_tex;
uniform bool      u_show_hatching;
uniform float     u_hatch_freq;       // lines per UV unit

void main() {
    vec3  n          = normalize(v_normal);
    float ndl        = max(dot(n, normalize(u_light_dir)), 0.0);
    float brightness = 0.15 + 0.85 * ndl;

    if (u_show_hatching) {
        // ── Hatching pass ─────────────────────────────────────────────
        // Texture stores (cos 2θ, sin 2θ): doubled-angle representation.
        // Decode to the actual orientation θ = atan2(G,R) / 2.
        vec2  rg   = u_has_dir_tex ? texture(u_dir_tex, v_uv).rg : vec2(1.0, 0.0);
        float a2   = atan(rg.g, rg.r);
        vec2  dir  = vec2(cos(a2 * 0.5), sin(a2 * 0.5));
        vec2  perp = vec2(-dir.y, dir.x);   // perpendicular = stripe-spacing axis

        // Normalise for UV anisotropy: scale V so that 1 UV unit in every
        // direction corresponds to the same world-space distance.
        // norm_factor = length of perp in world-metric UV space.
        // This is purely derived from vertex attributes — fully view-independent.
        float aspect      = max(v_uv_aspect, 1e-4);
        float norm_factor = length(vec2(perp.x, perp.y / aspect));
        float s   = dot(v_uv, perp) / max(norm_factor, 1e-7) * u_hatch_freq;
        float fw  = fwidth(s) * 1.5;

        // When fwidth is large the direction is changing faster than the stripe
        // period — the effective local frequency is too high to render clearly.
        // Fade out the hatching in those regions instead of showing aliased
        // thin lines at the boundary between two painted directions.
        float line_width = 0.45;
        float fade = 1.0 - smoothstep(line_width, 1.0, fw);

        float b   = fract(s);
        float ink = (1.0 - smoothstep(line_width - fw, line_width + fw, b)) * fade;

        vec3 paper_col = vec3(0.95, 0.93, 0.88);
        vec3 ink_col   = vec3(0.07, 0.06, 0.05);
        out_color = vec4(mix(paper_col, ink_col, ink), 1.0);

    } else {
        // ── Standard diffuse + direction colour overlay ───────────────
        vec3 base = vec3(0.65, 0.67, 0.72);
        vec3 col  = base * (0.25 + 0.75 * ndl);

        if (u_has_dir_tex) {
            vec2  rg    = texture(u_dir_tex, v_uv).rg;
            // Decode doubled-angle: stored as (cos 2θ, sin 2θ).
            // Normalise θ to [0, π) so the full hue wheel covers 180°.
            float angle = mod(atan(rg.g, rg.r) * 0.5, 3.14159265);
            float h     = fract(angle / 3.14159265);
            float sector = fract(h * 6.0);
            vec3  tint;
            int   si = int(h * 6.0) % 6;
            if      (si == 0) tint = vec3(1.0,          sector,        0.0);
            else if (si == 1) tint = vec3(1.0 - sector,  1.0,          0.0);
            else if (si == 2) tint = vec3(0.0,            1.0,          sector);
            else if (si == 3) tint = vec3(0.0,            1.0 - sector,  1.0);
            else if (si == 4) tint = vec3(sector,          0.0,          1.0);
            else              tint = vec3(1.0,              0.0,          1.0 - sector);
            float painted = 1.0 - step(0.999, rg.r) * step(-0.001, -abs(rg.g));
            col = mix(col, col * tint, 0.55 * painted);
        }

        out_color = vec4(col, 1.0);
    }
}
"""

_TICK_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_TICK_FRAG = """
#version 330 core
out vec4 out_color;
uniform vec4 u_color;
void main() {
    out_color = u_color;
}
"""


def _mat4_bytes(m: np.ndarray) -> bytes:
    """Upload a row-major numpy 4×4 to OpenGL column-major bytes."""
    return m.T.astype(np.float32).tobytes()


def _mat3_bytes(m: np.ndarray) -> bytes:
    """Upload a row-major numpy 3×3 to OpenGL column-major bytes."""
    return m.T.astype(np.float32).tobytes()


# ─────────────────────────────────────────────────────────────────────────────

class Renderer:
    MAX_TICKS = 8192   # maximum tick line segments

    def __init__(self) -> None:
        self.ctx: moderngl.Context = moderngl.create_context()
        self.ctx.gc_mode = "auto"

        # Shader programs
        self._pick_prog = self.ctx.program(
            vertex_shader=_PICK_VERT,
            fragment_shader=_PICK_FRAG,
        )
        self._view_prog = self.ctx.program(
            vertex_shader=_VIEW_VERT,
            fragment_shader=_VIEW_FRAG,
        )
        self._tick_prog = self.ctx.program(
            vertex_shader=_TICK_VERT,
            fragment_shader=_TICK_FRAG,
        )

        # FBOs / textures — created on first resize
        self.pick_fbo:  moderngl.Framebuffer | None = None
        self.view_fbo:  moderngl.Framebuffer | None = None
        self.pick_tex:  moderngl.Texture | None     = None
        self.view_tex:  moderngl.Texture | None     = None
        self._fbo_size  = (0, 0)

        # Mesh VAOs — created on upload_mesh
        self._pick_vao:    moderngl.VertexArray | None = None
        self._view_vao:    moderngl.VertexArray | None = None
        self._face_count   = 0

        # Tick VBO / VAO
        self._tick_vbo: moderngl.Buffer = self.ctx.buffer(
            reserve=self.MAX_TICKS * 2 * 12   # 2 vertices × 3 floats × 4 bytes
        )
        self._tick_vao: moderngl.VertexArray = self.ctx.vertex_array(
            self._tick_prog, [(self._tick_vbo, "3f", "in_position")]
        )
        self._tick_count = 0

        # UV → world KD-tree (populated on mesh upload)
        self._uv_tree:        cKDTree | None    = None
        self._face_positions: np.ndarray | None = None
        self._face_tangents:  np.ndarray | None = None
        self._face_bitangents: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────
    # FBO management
    # ──────────────────────────────────────────────────────────────────────

    def resize_fbos(self, size: tuple[int, int]) -> None:
        w, h = max(1, size[0]), max(1, size[1])
        if (w, h) == self._fbo_size:
            return
        self._fbo_size = (w, h)

        # Release old objects
        for attr in ("pick_fbo", "view_fbo", "pick_tex", "view_tex"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()

        # Picking FBO: RGBA32F
        self.pick_tex = self.ctx.texture((w, h), 4, dtype="f4")
        pick_depth    = self.ctx.depth_renderbuffer((w, h))
        self.pick_fbo = self.ctx.framebuffer([self.pick_tex], pick_depth)

        # Viewport FBO: RGBA8
        self.view_tex  = self.ctx.texture((w, h), 4)
        view_depth     = self.ctx.depth_renderbuffer((w, h))
        self.view_fbo  = self.ctx.framebuffer([self.view_tex], view_depth)

    # ──────────────────────────────────────────────────────────────────────
    # Mesh upload
    # ──────────────────────────────────────────────────────────────────────

    def upload_mesh(self, mesh: Mesh) -> None:
        """Build VAOs from a loaded Mesh."""
        # Release old VAOs if any
        if self._pick_vao:
            self._pick_vao.release()
        if self._view_vao:
            self._view_vao.release()

        # ── Viewport VAO (indexed) ─────────────────────────────────────
        vbo_pos    = self.ctx.buffer(mesh.vertices.tobytes())
        vbo_norm   = self.ctx.buffer(mesh.normals.tobytes())
        vbo_uv     = self.ctx.buffer(mesh.uvs.tobytes())
        vbo_aspect = self.ctx.buffer(mesh.uv_aspect.astype(np.float32).tobytes())
        ibo        = self.ctx.buffer(mesh.indices.tobytes())

        self._view_vao = self.ctx.vertex_array(
            self._view_prog,
            [
                (vbo_pos,    "3f", "in_position"),
                (vbo_norm,   "3f", "in_normal"),
                (vbo_uv,     "2f", "in_uv"),
                (vbo_aspect, "1f", "in_uv_aspect"),
            ],
            index_buffer=ibo,
            index_element_size=4,
        )

        # ── Picking VAO (non-indexed, expanded per-face) ───────────────
        expanded = _expand_for_picking(mesh)

        vbo_pick_pos   = self.ctx.buffer(expanded["pos"].tobytes())
        vbo_pick_bary  = self.ctx.buffer(expanded["bary"].tobytes())   # (F*3, 2) float32
        vbo_pick_id    = self.ctx.buffer(expanded["ids"].tobytes())    # (F*3,)   float32

        self._pick_vao = self.ctx.vertex_array(
            self._pick_prog,
            [
                (vbo_pick_pos,  "3f", "in_position"),
                (vbo_pick_bary, "2f", "in_bary_uv"),
                (vbo_pick_id,   "1f", "in_tri_id"),
            ],
        )
        self._face_count = mesh.face_count

        # ── UV KD-tree for tick sampling ───────────────────────────────
        if mesh.face_uv_centroids is not None:
            self._uv_tree         = cKDTree(mesh.face_uv_centroids)
            self._face_positions  = mesh.face_pos_centroids
            self._face_tangents   = mesh.face_tangents
            self._face_bitangents = mesh.face_bitangents

    # ──────────────────────────────────────────────────────────────────────
    # Layer texture management
    # ──────────────────────────────────────────────────────────────────────

    def create_layer_texture(self, layer: DirectionLayer) -> None:
        if layer.gpu_texture is not None:
            layer.gpu_texture.release()
        tex = self.ctx.texture(
            (layer.width, layer.height), 2,
            data=layer.texture.tobytes(), dtype="f4"
        )
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        layer.gpu_texture = tex
        layer.dirty = False

    def upload_dirty_region(
        self,
        layer: DirectionLayer,
        x: int, y: int, w: int, h: int,
    ) -> None:
        if layer.gpu_texture is None:
            self.create_layer_texture(layer)
            return
        region = layer.texture[y : y + h, x : x + w, :]
        layer.gpu_texture.write(region.astype(np.float32).tobytes(),
                                 viewport=(x, y, w, h))
        layer.dirty = False

    # ──────────────────────────────────────────────────────────────────────
    # Tick overlay
    # ──────────────────────────────────────────────────────────────────────

    def update_ticks(self, mesh: Mesh, layer: DirectionLayer,
                     spacing: int = 16) -> None:
        """Rebuild the GL_LINES tick buffer from the current direction texture."""
        if self._uv_tree is None:
            return

        H, W     = layer.height, layer.width
        tex      = layer.texture
        endpoints: list[np.ndarray] = []

        # Sample a regular UV grid
        for tv_px in range(0, H, spacing):
            for tu_px in range(0, W, spacing):
                r, g = tex[tv_px, tu_px]
                if r == 1.0 and g == 0.0:
                    continue   # neutral / unpainted — skip

                tu = tu_px / W
                tv = tv_px / H

                # Find nearest face centroid in UV space
                dist, fi = self._uv_tree.query([tu, tv])
                if dist > 0.05:   # no nearby geometry
                    continue

                pos   = self._face_positions[fi]
                tan   = self._face_tangents[fi]
                bitan = self._face_bitangents[fi]

                theta = float(np.arctan2(g, r)) * 0.5   # texture stores 2θ
                d     = np.cos(theta) * tan + np.sin(theta) * bitan
                half  = 0.015   # world-space half-length of tick

                endpoints.append(pos - d * half)
                endpoints.append(pos + d * half)

                if len(endpoints) >= self.MAX_TICKS * 2:
                    break
            if len(endpoints) >= self.MAX_TICKS * 2:
                break

        self._tick_count = len(endpoints)
        if endpoints:
            data = np.array(endpoints, dtype=np.float32).tobytes()
            self._tick_vbo.write(data)

    # ──────────────────────────────────────────────────────────────────────
    # Render passes
    # ──────────────────────────────────────────────────────────────────────

    def render_picking(self, camera: OrbitCamera, size: tuple[int, int]) -> None:
        if self._pick_vao is None or self.pick_fbo is None:
            return

        w, h  = size
        model = np.eye(4, dtype=np.float32)
        V     = camera.view_matrix()
        P     = camera.proj_matrix(w / max(h, 1))
        mvp   = P @ V @ model

        self.pick_fbo.use()
        self.pick_fbo.viewport = (0, 0, w, h)
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        self._pick_prog["u_mvp"].write(_mat4_bytes(mvp))
        self._pick_vao.render(moderngl.TRIANGLES)

    def render_viewport(
        self,
        camera: OrbitCamera,
        size: tuple[int, int],
        layer: DirectionLayer | None,
        show_hatching: bool = False,
        hatch_freq: float = 30.0,
    ) -> None:
        if self._view_vao is None or self.view_fbo is None:
            return

        w, h  = size
        model = np.eye(4, dtype=np.float32)
        V     = camera.view_matrix()
        P     = camera.proj_matrix(w / max(h, 1))
        mvp   = P @ V @ model

        normal_mat = np.linalg.inv(model[:3, :3]).T

        self.view_fbo.use()
        self.view_fbo.viewport = (0, 0, w, h)
        self.ctx.clear(0.10, 0.10, 0.11, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        prog = self._view_prog
        prog["u_model"].write(_mat4_bytes(model))
        prog["u_view"].write(_mat4_bytes(V))
        prog["u_proj"].write(_mat4_bytes(P))
        prog["u_normal_mat"].write(_mat3_bytes(normal_mat))
        light = np.array([0.6, 1.0, 0.8], dtype=np.float32)
        light /= np.linalg.norm(light)
        prog["u_light_dir"].value = tuple(light)

        # Bind direction texture (unit 0)
        if layer is not None and layer.gpu_texture is not None:
            layer.gpu_texture.use(location=0)
            prog["u_dir_tex"].value = 0
            prog["u_has_dir_tex"].value = True
        else:
            prog["u_has_dir_tex"].value = False

        prog["u_show_hatching"].value = show_hatching
        prog["u_hatch_freq"].value    = hatch_freq

        self._view_vao.render(moderngl.TRIANGLES)

        # Tick overlay
        if self._tick_count > 0:
            self.ctx.disable(moderngl.DEPTH_TEST)
            self._tick_prog["u_mvp"].write(_mat4_bytes(mvp))
            self._tick_prog["u_color"].value = (1.0, 0.85, 0.2, 1.0)
            self._tick_vao.render(moderngl.LINES, vertices=self._tick_count)

        # Restore default framebuffer so Dear ImGui can render
        self.ctx.screen.use()

    # ──────────────────────────────────────────────────────────────────────
    # Picking readback
    # ──────────────────────────────────────────────────────────────────────

    def read_pick_pixel(
        self, x: int, y: int
    ) -> tuple[int, float, float]:
        """
        Read the picking buffer at screen pixel (x, y) — (0,0) = top-left.
        Returns (triangle_id, bary_u, bary_v).  triangle_id == -1 on miss.
        """
        if self.pick_fbo is None:
            return -1, 0.0, 0.0

        W, H  = self._fbo_size
        gl_y  = H - 1 - y   # flip: screen y-down → OpenGL y-up

        if not (0 <= x < W and 0 <= gl_y < H):
            return -1, 0.0, 0.0

        self.pick_fbo.use()
        raw = self.pick_fbo.read(viewport=(x, gl_y, 1, 1), components=4, dtype="f4")
        self.ctx.screen.use()

        r, g, b, a = struct.unpack("4f", raw)

        # triangle_id is stored as a float (exact for ints up to 2^24)
        tri_id = int(round(r))
        if tri_id < 0 or tri_id >= self._face_count:
            return -1, 0.0, 0.0

        return tri_id, float(g), float(b)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _expand_for_picking(mesh: Mesh) -> dict:
    """
    Build a non-indexed (expanded) vertex buffer for the picking pass.
    Each triangle occupies 3 consecutive vertices with:
      - positions from the original mesh
      - barycentric coords (1,0), (0,1), (0,0) per vertex
      - triangle id repeated 3×
    """
    F = mesh.face_count
    flat_idx = mesh.indices.reshape(-1)                   # (F*3,)

    pos  = mesh.vertices[flat_idx].astype(np.float32)     # (F*3, 3)

    # Barycentric: for vertex 0 in each face → (1,0), vertex 1 → (0,1), vertex 2 → (0,0)
    bary_pattern = np.tile(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32),
        (F, 1)
    )                                                      # (F*3, 2)

    ids = np.repeat(np.arange(F, dtype=np.float32), 3)    # (F*3,)

    return {"pos": pos, "bary": bary_pattern, "ids": ids}
